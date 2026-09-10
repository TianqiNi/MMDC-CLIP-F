"""Frozen, named-view feature extraction for the view-risk study.

The lower-level extractor in :class:`VerifiedFrozenEncoder` accepts tensors that
have already received the legacy ImageNet normalization.  The
``extract_realized_parent`` convenience method is the pre-normalization entry
point for :class:`~.perturbations.RealizedParent` tensors in ``[0, 1]`` and
applies that normalization exactly once.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor

from mmdc_clip_f.backbones import get_backbone
from mmdc_clip_f.model import MultiViewCLIPClassifier
from mmdc_clip_f.provenance import sha256_file

from .fusion import FusionPairs, fuse_view_logits, validate_fusion_pairs
from .inputs import NUM_CLASSES, canonicalize_observed_views
from .perturbations import RealizedParent


LEGACY_IMAGE_MEAN = (0.485, 0.456, 0.406)
LEGACY_IMAGE_STD = (0.229, 0.224, 0.225)
LEGACY_PREPROCESSING = "resize-rgb-float-0-1_then_imagenet-normalize/v1"


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _tensor_bytes(tensor: Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return header + b"\0" + value.view(torch.uint8).numpy().tobytes()


def tensor_sha256(tensor: Tensor) -> str:
    """Hash a tensor's dtype, shape, and exact CPU byte representation."""

    if not isinstance(tensor, Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    return hashlib.sha256(_tensor_bytes(tensor)).hexdigest()


def _mapping_tensors(
    values: Mapping[str, Tensor], views: tuple[str, ...], field: str
) -> Mapping[str, Tensor]:
    if not isinstance(values, Mapping) or set(values) != set(views):
        raise ValueError(f"{field} must contain exactly the observed named views")
    return MappingProxyType({view: values[view] for view in views})


def _validate_float_tensor(tensor: Tensor, field: str, ndim: int) -> None:
    if not isinstance(tensor, Tensor) or not tensor.is_floating_point():
        raise TypeError(f"{field} must be a floating torch.Tensor")
    if tensor.ndim != ndim:
        raise ValueError(f"{field} must have rank {ndim}")
    if tensor.requires_grad:
        raise ValueError(f"{field} must be detached")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{field} must be finite")


@dataclass(frozen=True)
class FrozenEncoderIdentity:
    """Provenance derived by loading a tensor-only classifier checkpoint."""

    checkpoint_sha256: str
    backbone: str
    hf_model: str
    backbone_revision: str
    image_size: int
    hidden_size: int
    preprocessing: str
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    prompt_order: tuple[str, ...]
    text_input_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "backbone": self.backbone,
            "hf_model": self.hf_model,
            "backbone_revision": self.backbone_revision,
            "image_size": self.image_size,
            "hidden_size": self.hidden_size,
            "preprocessing": self.preprocessing,
            "image_mean": list(self.image_mean),
            "image_std": list(self.image_std),
            "prompt_order": list(self.prompt_order),
            "text_input_sha256": self.text_input_sha256,
        }


@dataclass(frozen=True)
class FrozenViewFeatures:
    """Label-free frozen evidence for one observed mask and an ordered batch."""

    logits_by_view: Mapping[str, Tensor]
    hidden_by_view: Mapping[str, Tensor]
    projected_by_view: Mapping[str, Tensor]
    normalized_text_embeddings: Tensor
    scores: Tensor
    probabilities: Tensor
    observed_views: tuple[str, ...]
    fusion_pairs: FusionPairs

    def __post_init__(self) -> None:
        observed = canonicalize_observed_views(self.observed_views)
        if tuple(self.observed_views) != observed:
            raise ValueError("observed_views must be unique and in canonical order")
        pairs = validate_fusion_pairs(self.fusion_pairs)
        logits = _mapping_tensors(self.logits_by_view, observed, "logits_by_view")
        hidden = _mapping_tensors(self.hidden_by_view, observed, "hidden_by_view")
        projected = _mapping_tensors(self.projected_by_view, observed, "projected_by_view")

        batch_size: int | None = None
        projection_size: int | None = None
        common_device: torch.device | None = None
        for view in observed:
            _validate_float_tensor(logits[view], f"logits for {view}", 2)
            _validate_float_tensor(hidden[view], f"hidden features for {view}", 3)
            _validate_float_tensor(projected[view], f"projected features for {view}", 2)
            if logits[view].shape[1] != NUM_CLASSES:
                raise ValueError("per-view logits must have four class columns")
            if hidden[view].shape[1] < 1:
                raise ValueError("hidden features must retain at least one token")
            current_batch = int(logits[view].shape[0])
            if current_batch < 1:
                raise ValueError("feature batches cannot be empty")
            if hidden[view].shape[0] != current_batch or projected[view].shape[0] != current_batch:
                raise ValueError("all per-view features must share a batch axis")
            if batch_size is None:
                batch_size = current_batch
                projection_size = int(projected[view].shape[1])
                common_device = logits[view].device
            elif current_batch != batch_size:
                raise ValueError("all observed views must share a batch size")
            if int(projected[view].shape[1]) != projection_size:
                raise ValueError("all projected features must share a width")
            if not (
                logits[view].device
                == hidden[view].device
                == projected[view].device
                == common_device
            ):
                raise ValueError("all feature tensors must share a device")

        _validate_float_tensor(self.normalized_text_embeddings, "normalized_text_embeddings", 2)
        _validate_float_tensor(self.scores, "scores", 2)
        _validate_float_tensor(self.probabilities, "probabilities", 2)
        if self.normalized_text_embeddings.shape != (NUM_CLASSES, projection_size):
            raise ValueError("text and projected feature widths/classes must agree")
        if self.scores.shape != (batch_size, NUM_CLASSES):
            raise ValueError("scores must have shape [batch, four classes]")
        if self.probabilities.shape != self.scores.shape:
            raise ValueError("probabilities must match score shape")
        if not (
            self.normalized_text_embeddings.device
            == self.scores.device
            == self.probabilities.device
            == common_device
        ):
            raise ValueError("text, score, and per-view tensors must share a device")
        if not torch.allclose(
            self.normalized_text_embeddings.norm(dim=1),
            torch.ones(NUM_CLASSES, device=common_device),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError("class text embeddings must be L2-normalized")
        expected_probabilities = torch.softmax(self.scores, dim=1)
        if not torch.allclose(self.probabilities, expected_probabilities, atol=1e-6, rtol=1e-6):
            raise ValueError("probabilities must be softmax of the stored classifier scores")

        object.__setattr__(self, "observed_views", observed)
        object.__setattr__(self, "fusion_pairs", pairs)
        object.__setattr__(self, "logits_by_view", logits)
        object.__setattr__(self, "hidden_by_view", hidden)
        object.__setattr__(self, "projected_by_view", projected)

    @property
    def batch_size(self) -> int:
        return int(self.scores.shape[0])

    @property
    def prediction(self) -> Tensor:
        return self.scores.argmax(dim=1)

    def all_tensors(self) -> tuple[Tensor, ...]:
        """Return every inference tensor, useful for gradient/finiteness audits."""

        return (
            *(self.logits_by_view[view] for view in self.observed_views),
            *(self.hidden_by_view[view] for view in self.observed_views),
            *(self.projected_by_view[view] for view in self.observed_views),
            self.normalized_text_embeddings,
            self.scores,
            self.probabilities,
        )


_VERIFIED_FACTORY_TOKEN = object()


class VerifiedFrozenEncoder:
    """A frozen classifier whose identity comes from a strictly loaded checkpoint."""

    def __init__(
        self,
        classifier: MultiViewCLIPClassifier,
        input_ids: Tensor,
        identity: FrozenEncoderIdentity,
        *,
        _factory_token: object,
    ) -> None:
        if _factory_token is not _VERIFIED_FACTORY_TOKEN:
            raise RuntimeError("VerifiedFrozenEncoder must be created by its checkpoint factory")
        self.classifier = classifier
        self.input_ids = input_ids.detach()
        self.identity = identity
        self._state_versions = {
            f"parameter:{name}": value._version for name, value in classifier.named_parameters()
        }
        self._state_versions.update(
            {f"buffer:{name}": value._version for name, value in classifier.named_buffers()}
        )

    def _assert_weights_unchanged(self) -> None:
        current = {
            f"parameter:{name}": value._version
            for name, value in self.classifier.named_parameters()
        }
        current.update(
            {f"buffer:{name}": value._version for name, value in self.classifier.named_buffers()}
        )
        if current != self._state_versions:
            raise RuntimeError("verified frozen classifier weights changed after checkpoint load")
        if any(parameter.requires_grad for parameter in self.classifier.parameters()):
            raise RuntimeError("verified frozen classifier parameters must remain frozen")

    def extract_normalized(self, normalized_views: Mapping[str, Tensor]) -> FrozenViewFeatures:
        """Encode only supplied named tensors, which must already be normalized."""

        views = _validate_normalized_views(normalized_views)
        observed = tuple(views)
        self._assert_weights_unchanged()
        self.classifier.eval()
        with torch.no_grad():
            batch_size = next(iter(views.values())).shape[0]
            images = torch.cat([views[view] for view in observed], dim=0)
            vision = self.classifier.clip_model.vision_model(pixel_values=images)
            if not hasattr(vision, "pooler_output") or not hasattr(vision, "last_hidden_state"):
                raise TypeError("vision encoder must expose pooler_output and last_hidden_state")
            projected_all = self.classifier.clip_model.visual_projection(vision.pooler_output)
            hidden_all = vision.last_hidden_state
            if hidden_all.ndim != 3 or hidden_all.shape[0] != images.shape[0]:
                raise ValueError("vision encoder returned invalid full token sequences")
            if hidden_all.shape[2] != self.identity.hidden_size:
                raise ValueError("vision hidden width disagrees with the pinned backbone")
            if projected_all.ndim != 2 or projected_all.shape[0] != images.shape[0]:
                raise ValueError("visual projection returned an invalid batch")

            projected = dict(zip(observed, projected_all.split(batch_size, dim=0)))
            hidden = dict(zip(observed, hidden_all.split(batch_size, dim=0)))
            text = self.classifier._text_features(self.input_ids)
            if text.ndim != 2 or text.shape[0] != NUM_CLASSES:
                raise ValueError("classifier must provide four class text embeddings")
            scale = self.classifier.clip_model.logit_scale.exp()
            logits: dict[str, Tensor] = {}
            for view in observed:
                image = projected[view]
                normalized_image = image / image.norm(p=2, dim=-1, keepdim=True)
                logits[view] = scale * normalized_image @ text.t()
            fused = fuse_view_logits(logits, observed, fusion_pairs=self.classifier.fusion_pairs)
            scores = fused.scores
            probabilities = torch.softmax(scores, dim=1)

        self._assert_weights_unchanged()
        return FrozenViewFeatures(
            logits_by_view={view: logits[view].detach() for view in observed},
            hidden_by_view={view: hidden[view].detach() for view in observed},
            projected_by_view={view: projected[view].detach() for view in observed},
            normalized_text_embeddings=text.detach(),
            scores=scores.detach(),
            probabilities=probabilities.detach(),
            observed_views=observed,
            fusion_pairs=fused.fusion_pairs,
        )

    def extract_realized_parent(self, parent: RealizedParent) -> FrozenViewFeatures:
        """Normalize one pre-normalization ``[0,1]`` parent exactly once and encode it."""

        if not isinstance(parent, RealizedParent):
            raise TypeError("parent must be a RealizedParent")
        normalized: dict[str, Tensor] = {}
        for view in parent.observed_views:
            image = parent.images[view]
            if not isinstance(image, Tensor) or not image.is_floating_point():
                raise TypeError("realized parent images must be floating torch tensors")
            if image.ndim != 3 or image.shape[0] != 3:
                raise ValueError("realized parent images must have shape [3, H, W]")
            if not torch.isfinite(image).all() or ((image < 0) | (image > 1)).any():
                raise ValueError("realized parent image pixels must be finite and in [0, 1]")
            mean = image.new_tensor(LEGACY_IMAGE_MEAN)[:, None, None]
            std = image.new_tensor(LEGACY_IMAGE_STD)[:, None, None]
            normalized[view] = ((image - mean) / std).unsqueeze(0)
        return self.extract_normalized(normalized)

    def select_parent_views(
        self, parent: FrozenViewFeatures, observed: Iterable[str]
    ) -> FrozenViewFeatures:
        """Select already-realized parent features and recompute only subset fusion."""

        if not isinstance(parent, FrozenViewFeatures):
            raise TypeError("parent must be FrozenViewFeatures")
        selected = canonicalize_observed_views(observed)
        if not set(selected).issubset(parent.observed_views):
            raise ValueError("selected child contains a view absent from its parent")
        logits = {view: parent.logits_by_view[view] for view in selected}
        fused = fuse_view_logits(logits, selected, fusion_pairs=parent.fusion_pairs)
        scores = fused.scores.detach()
        return FrozenViewFeatures(
            logits_by_view=logits,
            hidden_by_view={view: parent.hidden_by_view[view] for view in selected},
            projected_by_view={view: parent.projected_by_view[view] for view in selected},
            normalized_text_embeddings=parent.normalized_text_embeddings,
            scores=scores,
            probabilities=torch.softmax(scores, dim=1).detach(),
            observed_views=selected,
            fusion_pairs=fused.fusion_pairs,
        )

    def omit_parent_view(self, parent: FrozenViewFeatures, removed_view: str) -> FrozenViewFeatures:
        if removed_view not in parent.observed_views:
            raise ValueError("removed_view must name an observed parent view")
        if len(parent.observed_views) == 1:
            raise ValueError("omission child must remain nonempty")
        return self.select_parent_views(
            parent, tuple(view for view in parent.observed_views if view != removed_view)
        )


def _validate_normalized_views(values: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    if not isinstance(values, Mapping):
        raise TypeError("normalized_views must be a named mapping")
    supplied = dict(values)
    observed = canonicalize_observed_views(supplied)
    if set(supplied) != set(observed):
        raise ValueError("normalized views contain invalid names")
    batch: int | None = None
    shape: tuple[int, int, int] | None = None
    dtype: torch.dtype | None = None
    device: torch.device | None = None
    for view in observed:
        tensor = supplied[view]
        if not isinstance(tensor, Tensor) or not tensor.is_floating_point():
            raise TypeError(f"normalized image for {view} must be a floating tensor")
        if tensor.ndim != 4 or tensor.shape[1] != 3:
            raise ValueError("normalized images must have shape [batch, 3, H, W]")
        if tensor.shape[0] < 1 or tensor.shape[2] < 1 or tensor.shape[3] < 1:
            raise ValueError("normalized image dimensions cannot be empty")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"normalized image for {view} must be finite")
        current_shape = (int(tensor.shape[1]), int(tensor.shape[2]), int(tensor.shape[3]))
        if batch is None:
            batch, shape, dtype, device = (
                int(tensor.shape[0]),
                current_shape,
                tensor.dtype,
                tensor.device,
            )
        elif (int(tensor.shape[0]), current_shape, tensor.dtype, tensor.device) != (
            batch,
            shape,
            dtype,
            device,
        ):
            raise ValueError("normalized views must share batch, shape, dtype, and device")
    return MappingProxyType({view: supplied[view] for view in observed})


def load_verified_frozen_encoder(
    classifier: MultiViewCLIPClassifier,
    input_ids: Tensor,
    checkpoint_path: str | Path,
    *,
    backbone: str,
    prompts: Sequence[str],
) -> VerifiedFrozenEncoder:
    """Strictly load tensor-only weights, then freeze and provenance-bind an encoder."""

    from safetensors.torch import load_file

    if not isinstance(classifier, MultiViewCLIPClassifier):
        raise TypeError("classifier must be a MultiViewCLIPClassifier")
    if (
        not isinstance(input_ids, Tensor)
        or input_ids.ndim != 2
        or input_ids.shape[0] != NUM_CLASSES
    ):
        raise ValueError("input_ids must have one row for each of four ordered prompts")
    prompt_order = tuple(_required_string(prompt, "prompt") for prompt in prompts)
    if len(prompt_order) != NUM_CLASSES:
        raise ValueError("prompts must contain exactly four ordered class prompts")
    spec = get_backbone(_required_string(backbone, "backbone"))
    checkpoint = Path(checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint)
    try:
        device = next(classifier.parameters()).device
    except StopIteration as exc:
        raise ValueError("classifier must contain parameters") from exc
    state = load_file(str(checkpoint), device=str(device))
    classifier.load_state_dict(state, strict=True)
    if sha256_file(checkpoint) != checkpoint_sha256:
        raise RuntimeError("classifier checkpoint changed while it was being loaded")
    classifier.requires_grad_(False)
    classifier.eval()
    bound_input_ids = input_ids.detach().to(device)
    text_material = hashlib.sha256()
    text_material.update(_tensor_bytes(bound_input_ids))
    text_material.update(
        json.dumps(prompt_order, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    )
    identity = FrozenEncoderIdentity(
        checkpoint_sha256=checkpoint_sha256,
        backbone=spec.name,
        hf_model=spec.hf_model,
        backbone_revision=spec.revision,
        image_size=spec.image_size,
        hidden_size=spec.hidden_size,
        preprocessing=LEGACY_PREPROCESSING,
        image_mean=LEGACY_IMAGE_MEAN,
        image_std=LEGACY_IMAGE_STD,
        prompt_order=prompt_order,
        text_input_sha256=text_material.hexdigest(),
    )
    return VerifiedFrozenEncoder(
        classifier,
        bound_input_ids,
        identity,
        _factory_token=_VERIFIED_FACTORY_TOKEN,
    )
