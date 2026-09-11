"""Label-free, canonical inputs for relation-aware view-risk heads.

Raw preparation is deliberately parameter-free and is safe to share as a
specification.  :class:`ViewRiskInputPreparation` owns every learned projection;
each candidate or control must instantiate its own module.  The default scaler
is an identity operation with no fitted state.  A fitted replacement belongs to
the later training phase and may only be fitted on confidence-fit data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mmdc_clip_f.backbones import get_backbone

from .features import FrozenViewFeatures
from .fusion import FusionPairs, fuse_view_logits
from .inputs import CANONICAL_VIEWS, NUM_CLASSES


HeadInputMode = Literal["combined", "hidden_only", "evidence_only"]
TOKEN_POLICY_BY_BACKBONE = {
    "vit_b_32": "exclude_cls",
    "vit_l_14_336": "include_cls",
}


def _check_tensor(
    value: Tensor,
    name: str,
    shape: tuple[int | None, ...],
    *,
    floating: bool | None = None,
) -> None:
    if not isinstance(value, Tensor) or value.ndim != len(shape):
        raise ValueError(f"{name} must be a rank-{len(shape)} tensor")
    if any(expected is not None and actual != expected for actual, expected in zip(value.shape, shape)):
        raise ValueError(f"{name} has an invalid shape")
    if floating is True and not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if floating is False and value.is_floating_point():
        raise TypeError(f"{name} must not be floating point")
    if value.requires_grad:
        raise ValueError(f"{name} must be detached from the frozen encoder")
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class RawHeadInputs:
    """Reusable label-free evidence in the four canonical named slots.

    Invalid slots are masked rather than assigned semantic values.  Consumers
    must apply the supplied masks, so even adversarial values stored in an
    invalid slot cannot influence a valid output.
    """

    pooled_tokens: Tensor
    current_scores: Tensor
    current_probabilities: Tensor
    fused_evidence: Tensor
    fused_vacuity: Tensor
    view_evidence: Tensor
    view_vacuity: Tensor
    omission_score_differences: Tensor
    observed_mask: Tensor
    removal_valid_mask: Tensor
    current_prediction: Tensor
    omission_prediction: Tensor
    prediction_agreement: Tensor
    no_removal: Tensor
    backbone: str
    token_policy: str
    fusion_pairs: FusionPairs
    view_order: tuple[str, ...] = field(default=CANONICAL_VIEWS, init=False)

    def __post_init__(self) -> None:
        batch, slots, width = self.pooled_tokens.shape
        if batch < 1 or slots != len(CANONICAL_VIEWS) or width < 1:
            raise ValueError("pooled_tokens must have shape [batch, 4, hidden_width]")
        shapes = {
            "pooled_tokens": (batch, 4, width),
            "current_scores": (batch, NUM_CLASSES),
            "current_probabilities": (batch, NUM_CLASSES),
            "fused_evidence": (batch, NUM_CLASSES),
            "fused_vacuity": (batch, 1),
            "view_evidence": (batch, 4, NUM_CLASSES),
            "view_vacuity": (batch, 4, 1),
            "omission_score_differences": (batch, 4, NUM_CLASSES),
        }
        for name, shape in shapes.items():
            _check_tensor(getattr(self, name), name, shape, floating=True)
        for name, shape in {
            "observed_mask": (batch, 4),
            "removal_valid_mask": (batch, 4),
            "current_prediction": (batch,),
            "omission_prediction": (batch, 4),
            "prediction_agreement": (batch, 4),
            "no_removal": (batch, 1),
        }.items():
            _check_tensor(getattr(self, name), name, shape, floating=False)
        for name in ("observed_mask", "removal_valid_mask", "prediction_agreement", "no_removal"):
            if getattr(self, name).dtype != torch.bool:
                raise TypeError(f"{name} must use boolean dtype")
        if self.current_prediction.dtype != torch.long or self.omission_prediction.dtype != torch.long:
            raise TypeError("prediction metadata must use torch.long")
        device = self.pooled_tokens.device
        if any(tensor.device != device for tensor in self.all_tensors()):
            raise ValueError("all raw head inputs must share one device")
        if not torch.equal(self.removal_valid_mask & ~self.observed_mask, torch.zeros_like(self.observed_mask)):
            raise ValueError("only observed views can be valid removals")
        if not torch.equal(self.no_removal[:, 0], ~self.removal_valid_mask.any(dim=1)):
            raise ValueError("no_removal must identify parents without a valid omission")

    @property
    def hidden_width(self) -> int:
        return int(self.pooled_tokens.shape[-1])

    @property
    def batch_size(self) -> int:
        return int(self.pooled_tokens.shape[0])

    def all_tensors(self) -> tuple[Tensor, ...]:
        return (
            self.pooled_tokens,
            self.current_scores,
            self.current_probabilities,
            self.fused_evidence,
            self.fused_vacuity,
            self.view_evidence,
            self.view_vacuity,
            self.omission_score_differences,
            self.observed_mask,
            self.removal_valid_mask,
            self.current_prediction,
            self.omission_prediction,
            self.prediction_agreement,
            self.no_removal,
        )


class IdentityFeatureScaler(nn.Module):
    """A documented no-fit scaler used until P4 introduces fit-only scaling."""

    def forward(self, value: Tensor) -> Tensor:
        return value

    def fit(self, value: Tensor) -> None:
        del value
        raise RuntimeError("IdentityFeatureScaler does not fit or retain data")


@dataclass(frozen=True)
class PreparedHeadInputs:
    """Candidate-owned projections plus raw validity/reporting metadata."""

    view_features: Tensor
    omission_features: Tensor
    global_features: Tensor
    observed_mask: Tensor
    removal_valid_mask: Tensor
    current_prediction: Tensor
    omission_prediction: Tensor
    prediction_agreement: Tensor
    no_removal: Tensor


def prepare_raw_head_inputs(
    features: FrozenViewFeatures,
    *,
    backbone: str,
) -> RawHeadInputs:
    """Derive label-free parent and leave-one-view-out evidence.

    Children are recomputed from the parent's exact frozen per-view logits using
    its accepted fusion tree.  No image is re-encoded and no label, correctness,
    TCP, clean counterpart, or corruption descriptor is accepted by this API.
    """

    if not isinstance(features, FrozenViewFeatures):
        raise TypeError("features must be FrozenViewFeatures")
    spec = get_backbone(backbone)
    try:
        token_policy = TOKEN_POLICY_BY_BACKBONE[spec.name]
    except KeyError as exc:  # defensive if a future backbone is added without protocol review
        raise ValueError(f"backbone {spec.name!r} has no accepted token policy") from exc
    observed = features.observed_views
    batch = features.batch_size
    device = features.scores.device
    dtype = features.scores.dtype
    with torch.no_grad():
        parent = fuse_view_logits(
            features.logits_by_view,
            observed,
            fusion_pairs=features.fusion_pairs,
        )
        if not torch.allclose(features.scores, parent.scores, atol=1e-6, rtol=1e-6):
            raise ValueError("stored classifier scores disagree with accepted fusion")
        if not torch.equal(features.prediction, parent.scores.argmax(dim=1)):
            raise ValueError("stored classifier prediction disagrees with accepted fusion")

        pooled_tokens = torch.zeros(batch, 4, spec.hidden_size, dtype=dtype, device=device)
        view_evidence = torch.zeros(batch, 4, NUM_CLASSES, dtype=dtype, device=device)
        view_vacuity = torch.zeros(batch, 4, 1, dtype=dtype, device=device)
        observed_mask = torch.zeros(batch, 4, dtype=torch.bool, device=device)
        removal_valid = torch.zeros_like(observed_mask)
        omission_differences = torch.zeros_like(view_evidence)
        omission_prediction = torch.full(
            (batch, 4), -1, dtype=torch.long, device=device
        )
        agreement = torch.zeros_like(observed_mask)

        for column, view in enumerate(CANONICAL_VIEWS):
            if view not in observed:
                continue
            hidden = features.hidden_by_view[view]
            if hidden.shape[-1] != spec.hidden_size:
                raise ValueError(
                    f"{view} hidden width {hidden.shape[-1]} does not match backbone hidden width {spec.hidden_size}"
                )
            if token_policy == "exclude_cls":
                if hidden.shape[1] < 2:
                    raise ValueError("ViT-B token policy cannot exclude CLS from a singleton sequence")
                pooled = hidden[:, 1:].mean(dim=1)
            else:
                pooled = hidden.mean(dim=1)
            pooled_tokens[:, column] = pooled
            alpha = F.softplus(features.logits_by_view[view]) + 1
            view_evidence[:, column] = alpha - 1
            view_vacuity[:, column] = NUM_CLASSES / alpha.sum(dim=1, keepdim=True)
            observed_mask[:, column] = True

            if len(observed) > 1:
                child_views = tuple(candidate for candidate in observed if candidate != view)
                child = fuse_view_logits(
                    features.logits_by_view,
                    child_views,
                    fusion_pairs=features.fusion_pairs,
                )
                child_prediction = child.scores.argmax(dim=1)
                removal_valid[:, column] = True
                omission_differences[:, column] = features.scores - child.scores
                omission_prediction[:, column] = child_prediction
                agreement[:, column] = features.prediction == child_prediction

        raw = RawHeadInputs(
            pooled_tokens=pooled_tokens.detach(),
            current_scores=features.scores.detach(),
            current_probabilities=features.probabilities.detach(),
            fused_evidence=parent.stats.evidence.detach(),
            fused_vacuity=parent.stats.uncertainty.detach(),
            view_evidence=view_evidence.detach(),
            view_vacuity=view_vacuity.detach(),
            omission_score_differences=omission_differences.detach(),
            observed_mask=observed_mask,
            removal_valid_mask=removal_valid,
            current_prediction=features.prediction.detach().to(torch.long),
            omission_prediction=omission_prediction,
            prediction_agreement=agreement,
            no_removal=~removal_valid.any(dim=1, keepdim=True),
            backbone=spec.name,
            token_policy=token_policy,
            fusion_pairs=features.fusion_pairs,
        )
    return raw


class ViewRiskInputPreparation(nn.Module):
    """Own the learned projection for exactly one candidate or control."""

    def __init__(
        self,
        hidden_width: int,
        *,
        input_mode: HeadInputMode = "combined",
        hidden_dim: int = 128,
        use_omission_features: bool = True,
    ) -> None:
        super().__init__()
        if hidden_width <= 0 or hidden_dim <= 0:
            raise ValueError("hidden_width and hidden_dim must be positive")
        if input_mode not in ("combined", "hidden_only", "evidence_only"):
            raise ValueError("input_mode must be combined, hidden_only, or evidence_only")
        self.hidden_width = hidden_width
        self.hidden_dim = hidden_dim
        self.input_mode = input_mode
        self.use_omission_features = bool(use_omission_features)
        # Scaling is intentionally fixed to identity in P3A.  P4 may introduce
        # a provenance-bound confidence-fit-only scaler rather than injecting a
        # silently fitted tune/pilot transform here.
        self.scaler = IdentityFeatureScaler()
        self.view_name_embeddings = nn.Parameter(torch.empty(4, hidden_dim))
        nn.init.normal_(self.view_name_embeddings, std=0.02)
        if input_mode != "evidence_only":
            self.hidden_projection = nn.Linear(hidden_width, hidden_dim)
        if input_mode != "hidden_only":
            self.evidence_projection = nn.Linear(NUM_CLASSES + 1, hidden_dim)

    @property
    def global_feature_dim(self) -> int:
        return 0 if self.input_mode == "hidden_only" else (NUM_CLASSES * 3 + 1)

    @property
    def omission_feature_dim(self) -> int:
        if self.input_mode == "hidden_only" or not self.use_omission_features:
            return 0
        return NUM_CLASSES

    def forward(self, raw: RawHeadInputs) -> PreparedHeadInputs:
        if not isinstance(raw, RawHeadInputs):
            raise TypeError("raw must be RawHeadInputs")
        if raw.hidden_width != self.hidden_width:
            raise ValueError("raw hidden width does not match preparation module")
        mask = raw.observed_mask.unsqueeze(-1)
        view_features = self.view_name_embeddings.unsqueeze(0).expand(raw.batch_size, -1, -1)
        if self.input_mode != "evidence_only":
            hidden = raw.pooled_tokens.masked_fill(~mask, 0)
            view_features = view_features + self.hidden_projection(hidden)
        if self.input_mode != "hidden_only":
            evidence = torch.cat((raw.view_evidence, raw.view_vacuity), dim=-1)
            evidence = self.scaler(evidence.masked_fill(~mask, 0))
            view_features = view_features + self.evidence_projection(evidence)
        view_features = view_features.masked_fill(~mask, 0)

        if self.omission_feature_dim:
            removal_mask = raw.removal_valid_mask.unsqueeze(-1)
            omission = self.scaler(
                raw.omission_score_differences.masked_fill(~removal_mask, 0)
            )
        else:
            omission = raw.pooled_tokens.new_zeros(raw.batch_size, 4, 0)

        if self.global_feature_dim:
            global_features = self.scaler(
                torch.cat(
                    (
                        raw.current_scores,
                        raw.current_probabilities,
                        raw.fused_evidence,
                        raw.fused_vacuity,
                    ),
                    dim=1,
                )
            )
        else:
            global_features = raw.pooled_tokens.new_zeros(raw.batch_size, 0)
        return PreparedHeadInputs(
            view_features=view_features,
            omission_features=omission,
            global_features=global_features,
            observed_mask=raw.observed_mask,
            removal_valid_mask=raw.removal_valid_mask,
            current_prediction=raw.current_prediction,
            omission_prediction=raw.omission_prediction,
            prediction_agreement=raw.prediction_agreement,
            no_removal=raw.no_removal,
        )


__all__ = [
    "HeadInputMode",
    "IdentityFeatureScaler",
    "PreparedHeadInputs",
    "RawHeadInputs",
    "TOKEN_POLICY_BY_BACKBONE",
    "ViewRiskInputPreparation",
    "prepare_raw_head_inputs",
]
