"""Named-view CLIP classifier with the paper implementation's DS fusion."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor, nn


def _tensor_from_output(output):
    if isinstance(output, Tensor):
        return output
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    raise TypeError(f"Unsupported Transformers output type: {type(output)!r}")


class MultiViewCLIPClassifier(nn.Module):
    def __init__(
        self,
        clip_model: nn.Module,
        input_order: Sequence[str],
        fusion_pairs: Sequence[Sequence[str]],
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.input_order = tuple(input_order)
        self.fusion_pairs = tuple(tuple(pair) for pair in fusion_pairs)
        self.num_classes = num_classes
        self.softplus = nn.Softplus()
        expected = {"L_CC", "L_MLO", "R_CC", "R_MLO"}
        if len(self.input_order) != 4 or set(self.input_order) != expected:
            raise ValueError("input_order must name all four canonical views")
        flattened = [view for pair in self.fusion_pairs for view in pair]
        if len(self.fusion_pairs) != 2 or len(flattened) != 4 or set(flattened) != expected:
            raise ValueError("fusion_pairs must partition the four canonical views")

    def _text_features(self, input_ids: Tensor) -> Tensor:
        output = self.clip_model.get_text_features(input_ids=input_ids)
        features = _tensor_from_output(output)
        return features / features.norm(p=2, dim=-1, keepdim=True)

    def encode_views(
        self, views: Mapping[str, Tensor]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        batch_size = self._validate_views(views)
        images = torch.cat([views[name] for name in self.input_order], dim=0)
        vision_output = self.clip_model.vision_model(pixel_values=images)
        projected = self.clip_model.visual_projection(vision_output.pooler_output)
        projected_chunks = projected.split(batch_size, dim=0)
        hidden_chunks = vision_output.last_hidden_state.split(batch_size, dim=0)
        return dict(zip(self.input_order, projected_chunks)), dict(
            zip(self.input_order, hidden_chunks)
        )

    def _validate_views(self, views: Mapping[str, Tensor]) -> int:
        missing = set(self.input_order).difference(views)
        if missing:
            raise ValueError(f"Missing input views: {sorted(missing)}")
        sizes = {int(views[name].shape[0]) for name in self.input_order}
        if len(sizes) != 1:
            raise ValueError("All views must use the same batch size")
        return sizes.pop()

    def _combine_two(self, alpha1: Tensor, alpha2: Tensor) -> tuple[Tensor, Tensor]:
        if alpha1.shape != alpha2.shape or alpha1.shape[1] != self.num_classes:
            raise ValueError("DS fusion expects equal [batch, num_classes] tensors")
        strength1 = alpha1.sum(dim=1, keepdim=True)
        strength2 = alpha2.sum(dim=1, keepdim=True)
        evidence1 = alpha1 - 1
        evidence2 = alpha2 - 1
        belief1 = evidence1 / strength1.expand_as(evidence1)
        belief2 = evidence2 / strength2.expand_as(evidence2)
        uncertainty1 = self.num_classes / strength1
        uncertainty2 = self.num_classes / strength2

        products = torch.bmm(belief1.unsqueeze(2), belief2.unsqueeze(1))
        conflict = products.sum(dim=(1, 2)) - torch.diagonal(products, dim1=-2, dim2=-1).sum(-1)
        denominator = (1 - conflict).unsqueeze(1)
        fused_belief = (
            belief1 * belief2
            + belief1 * uncertainty2.expand_as(belief1)
            + belief2 * uncertainty1.expand_as(belief2)
        ) / denominator
        fused_uncertainty = uncertainty1 * uncertainty2 / denominator
        fused_strength = self.num_classes / fused_uncertainty
        fused_alpha = fused_belief * fused_strength.expand_as(fused_belief) + 1
        return fused_alpha, fused_uncertainty

    def fuse_logits(self, logits_by_view: Mapping[str, Tensor]) -> Tensor:
        alphas = {view: self.softplus(logits) + 1 for view, logits in logits_by_view.items()}
        first, _ = self._combine_two(
            alphas[self.fusion_pairs[0][0]], alphas[self.fusion_pairs[0][1]]
        )
        second, _ = self._combine_two(
            alphas[self.fusion_pairs[1][0]], alphas[self.fusion_pairs[1][1]]
        )
        fused, _ = self._combine_two(first, second)
        # Kept for historical protocol compatibility with the accepted-paper code.
        return self.softplus(fused)

    def classify_embeddings(self, embeddings: Mapping[str, Tensor], input_ids: Tensor) -> Tensor:
        text = self._text_features(input_ids)
        scale = self.clip_model.logit_scale.exp()
        logits: dict[str, Tensor] = {}
        for view, image in embeddings.items():
            normalized = image / image.norm(p=2, dim=-1, keepdim=True)
            logits[view] = scale * normalized @ text.t()
        return self.fuse_logits(logits)

    def forward_with_hidden_states(
        self, views: Mapping[str, Tensor], input_ids: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        embeddings, hidden = self.encode_views(views)
        return self.classify_embeddings(embeddings, input_ids), hidden

    def forward(self, views: Mapping[str, Tensor], input_ids: Tensor) -> Tensor:
        batch_size = self._validate_views(views)
        images = torch.cat([views[name] for name in self.input_order], dim=0)
        output = self.clip_model.get_image_features(pixel_values=images)
        features = _tensor_from_output(output)
        chunks = features.split(batch_size, dim=0)
        return self.classify_embeddings(dict(zip(self.input_order, chunks)), input_ids)
