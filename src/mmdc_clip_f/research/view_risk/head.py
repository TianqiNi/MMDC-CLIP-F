"""Relation-aware confidence head and parent-normalized scientific losses."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mmdc_clip_f.backbones import get_backbone

from .head_inputs import HeadInputMode, RawHeadInputs, ViewRiskInputPreparation
from .inputs import CANONICAL_VIEWS
from .targets import InterventionTargets


AuxiliaryTask = Literal["signed", "magnitude", "corruption"]
SIGNED_EFFECT_CLASS_ORDER = (-1, 0, 1)
MAGNITUDE_EFFECT_CLASS_ORDER = (0, 1)


@dataclass(frozen=True)
class RelationAwareHeadConfig:
    backbone: str
    input_mode: HeadInputMode = "combined"
    use_omission_features: bool = True
    use_relation_biases: bool = True
    auxiliary_task: AuxiliaryTask = "signed"
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    head_dim: int = 32
    ff_dim: int = 256
    dropout: float = 0.2

    def __post_init__(self) -> None:
        get_backbone(self.backbone)
        if self.input_mode not in ("combined", "hidden_only", "evidence_only"):
            raise ValueError("unsupported input_mode")
        if self.auxiliary_task not in ("signed", "magnitude", "corruption"):
            raise ValueError("unsupported auxiliary_task")
        if self.hidden_dim <= 0 or self.num_layers <= 0 or self.num_heads <= 0:
            raise ValueError("hidden and attention dimensions must be positive")
        if self.head_dim <= 0 or self.hidden_dim != self.num_heads * self.head_dim:
            raise ValueError("hidden_dim must equal num_heads * head_dim")
        if self.ff_dim <= 0:
            raise ValueError("ff_dim must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


def _relation_codes() -> Tensor:
    codes = torch.zeros(4, 4, 3, dtype=torch.bool)
    for query, query_name in enumerate(CANONICAL_VIEWS):
        for key, key_name in enumerate(CANONICAL_VIEWS):
            codes[query, key, 0] = query_name[0] == key_name[0]
            codes[query, key, 1] = query_name.split("_", 1)[1] == key_name.split("_", 1)[1]
            codes[query, key, 2] = query == key
    return codes


class RelationAttentionLayer(nn.Module):
    """Masked self-attention with learned named-view relation biases."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        ff_dim: int,
        dropout: float,
        use_relation_biases: bool,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.register_buffer("relation_codes", _relation_codes(), persistent=True)
        if use_relation_biases:
            # Rows are same-breast, same-projection, and explicit self; columns
            # are attention heads.  This is 12 learned scalars per layer.
            self.relation_bias = nn.Parameter(torch.zeros(3, num_heads))
        else:
            self.register_parameter("relation_bias", None)
        self.dropout = nn.Dropout(dropout)
        self.attention_normalization = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.feed_forward_normalization = nn.LayerNorm(hidden_dim)

    def _heads(self, value: Tensor) -> Tensor:
        batch, slots, _ = value.shape
        return value.reshape(batch, slots, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, value: Tensor, observed_mask: Tensor) -> Tensor:
        if value.ndim != 3 or value.shape[1:] != (4, self.hidden_dim):
            raise ValueError("relation input must have shape [batch, 4, hidden_dim]")
        if observed_mask.shape != value.shape[:2] or observed_mask.dtype != torch.bool:
            raise ValueError("observed_mask must be boolean with shape [batch, 4]")
        if not observed_mask.any(dim=1).all():
            raise ValueError("each parent must contain at least one observed view")

        query = self._heads(self.query(value))
        key = self._heads(self.key(value))
        attended_value = self._heads(self.value(value))
        scores = torch.matmul(query, key.transpose(-1, -2)) / sqrt(self.head_dim)
        if self.relation_bias is not None:
            named_bias = torch.einsum(
                "qkr,rh->hqk", self.relation_codes.to(value.dtype), self.relation_bias
            )
            scores = scores + named_bias.unsqueeze(0)
        scores = scores.masked_fill(~observed_mask[:, None, None, :], -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        # Absent queries have valid keys but are explicitly erased after attention.
        attended = torch.matmul(weights, attended_value).transpose(1, 2).reshape_as(value)
        query_mask = observed_mask.unsqueeze(-1)
        attended = self.output(attended).masked_fill(~query_mask, 0)
        value = self.attention_normalization(value + self.dropout(attended))
        value = value.masked_fill(~query_mask, 0)
        value = self.feed_forward_normalization(value + self.feed_forward(value))
        return value.masked_fill(~query_mask, 0)


@dataclass(frozen=True)
class RelationAwareHeadOutput:
    error_logit: Tensor
    confidence: Tensor
    effect_representations: Tensor
    raw_auxiliary_logits: Tensor
    reported_auxiliary_probabilities: Tensor
    auxiliary_valid_mask: Tensor
    auxiliary_task: AuxiliaryTask
    classifier_prediction: Tensor | None = None


class RelationAwareConfidenceHead(nn.Module):
    """Two-layer named-relation head that never changes classifier predictions."""

    def __init__(self, config: RelationAwareHeadConfig) -> None:
        super().__init__()
        if not isinstance(config, RelationAwareHeadConfig):
            raise TypeError("config must be RelationAwareHeadConfig")
        self.config = config
        spec = get_backbone(config.backbone)
        self.preparation = ViewRiskInputPreparation(
            spec.hidden_size,
            input_mode=config.input_mode,
            hidden_dim=config.hidden_dim,
            use_omission_features=config.use_omission_features,
        )
        self.relation_layers = nn.ModuleList(
            [
                RelationAttentionLayer(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    head_dim=config.head_dim,
                    ff_dim=config.ff_dim,
                    dropout=config.dropout,
                    use_relation_biases=config.use_relation_biases,
                )
                for _ in range(config.num_layers)
            ]
        )
        effect_input_dim = config.hidden_dim + self.preparation.omission_feature_dim
        self.effect_representation = nn.Sequential(
            nn.Linear(effect_input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        auxiliary_classes = 3 if config.auxiliary_task == "signed" else 2
        self.auxiliary_classifier = nn.Linear(config.hidden_dim, auxiliary_classes)
        risk_input_dim = (
            config.hidden_dim * 2 + self.preparation.global_feature_dim + 1
        )
        self.risk = nn.Sequential(
            nn.Linear(risk_input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    @staticmethod
    def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
        weights = mask.unsqueeze(-1).to(value.dtype)
        count = weights.sum(dim=1).clamp_min(1)
        return (value * weights).sum(dim=1) / count

    def forward(self, raw: RawHeadInputs) -> RelationAwareHeadOutput:
        if raw.backbone != self.config.backbone:
            raise ValueError("raw input backbone does not match head configuration")
        prepared = self.preparation(raw)
        interacted = prepared.view_features
        for layer in self.relation_layers:
            interacted = layer(interacted, prepared.observed_mask)

        effect_input = torch.cat((interacted, prepared.omission_features), dim=-1)
        auxiliary_valid = (
            prepared.observed_mask
            if self.config.auxiliary_task == "corruption"
            else prepared.removal_valid_mask
        )
        effect_representations = self.effect_representation(effect_input)
        effect_representations = effect_representations.masked_fill(
            ~auxiliary_valid.unsqueeze(-1), 0
        )
        raw_auxiliary_logits = self.auxiliary_classifier(effect_representations)
        raw_probabilities = torch.softmax(raw_auxiliary_logits, dim=-1)
        reported = raw_probabilities.masked_fill(~auxiliary_valid.unsqueeze(-1), 0)
        if self.config.auxiliary_task in ("signed", "magnitude"):
            forced = reported.new_tensor(
                [0.0, 1.0, 0.0]
                if self.config.auxiliary_task == "signed"
                else [1.0, 0.0]
            )
            force_mask = auxiliary_valid & prepared.prediction_agreement
            reported = torch.where(force_mask.unsqueeze(-1), forced, reported)

        pooled_views = self._masked_mean(interacted, prepared.observed_mask)
        pooled_effects = self._masked_mean(effect_representations, auxiliary_valid)
        risk_features = torch.cat(
            (
                pooled_views,
                pooled_effects,
                prepared.global_features,
                prepared.no_removal.to(interacted.dtype),
            ),
            dim=1,
        )
        error_logit = self.risk(risk_features).squeeze(1)
        confidence = 1 - torch.sigmoid(error_logit)
        return RelationAwareHeadOutput(
            error_logit=error_logit,
            confidence=confidence,
            effect_representations=effect_representations,
            raw_auxiliary_logits=raw_auxiliary_logits,
            reported_auxiliary_probabilities=reported,
            auxiliary_valid_mask=auxiliary_valid,
            auxiliary_task=self.config.auxiliary_task,
            classifier_prediction=prepared.current_prediction,
        )


@dataclass(frozen=True)
class ViewRiskLoss:
    total: Tensor
    error_bce: Tensor
    auxiliary_ce: Tensor
    per_parent_error_bce: Tensor
    per_parent_auxiliary_ce: Tensor


def _error_target(
    targets: InterventionTargets | Tensor,
    output: RelationAwareHeadOutput,
) -> Tensor:
    value = targets.observed_error if isinstance(targets, InterventionTargets) else targets
    if not isinstance(value, Tensor) or value.ndim != 1 or value.shape != output.error_logit.shape:
        raise ValueError("observed error target must have shape [batch]")
    if value.device != output.error_logit.device:
        raise ValueError("targets and output must share a device")
    if value.is_floating_point() or ((value != 0) & (value != 1)).any():
        raise ValueError("observed error target must contain integer 0 or 1")
    return value.to(output.error_logit.dtype)


def compute_view_risk_loss(
    output: RelationAwareHeadOutput,
    targets: InterventionTargets | Tensor,
    *,
    auxiliary_weight: float = 1.0,
    corruption_targets: Tensor | None = None,
) -> ViewRiskLoss:
    """Average BCE plus each parent's mean valid raw-logit auxiliary CE.

    The deterministic reporting overwrite is never used for optimization.
    Corruption labels are accepted only here, never by raw preparation or the
    head forward path.
    """

    if not isinstance(output, RelationAwareHeadOutput):
        raise TypeError("output must be RelationAwareHeadOutput")
    if not isinstance(auxiliary_weight, (float, int)) or auxiliary_weight < 0:
        raise ValueError("auxiliary_weight must be nonnegative")
    error_target = _error_target(targets, output)
    per_parent_bce = F.binary_cross_entropy_with_logits(
        output.error_logit, error_target, reduction="none"
    )
    batch, slots, classes = output.raw_auxiliary_logits.shape
    if output.auxiliary_valid_mask.shape != (batch, slots):
        raise ValueError("auxiliary validity mask shape is inconsistent")

    if output.auxiliary_task == "corruption":
        if corruption_targets is None:
            raise ValueError("corruption_targets are required for corruption auxiliary loss")
        if corruption_targets.shape != (batch, slots) or corruption_targets.device != error_target.device:
            raise ValueError("corruption_targets must have shape [batch, 4] on the output device")
        if corruption_targets.is_floating_point() or (
            (corruption_targets != -1)
            & (corruption_targets != 0)
            & (corruption_targets != 1)
        ).any():
            raise ValueError("corruption_targets must contain 0, 1, or -1")
        if (corruption_targets[~output.auxiliary_valid_mask] != -1).any():
            raise ValueError("corruption_targets must use -1 for absent views")
        auxiliary_target = corruption_targets
    else:
        if corruption_targets is not None:
            raise ValueError("corruption_targets are only valid for the corruption auxiliary")
        if not isinstance(targets, InterventionTargets):
            raise TypeError("signed and magnitude losses require InterventionTargets")
        target_valid = targets.valid_removal_mask.to(output.error_logit.device)
        if target_valid.ndim == 1:
            target_valid = target_valid.unsqueeze(0).expand(batch, -1)
        if not torch.equal(target_valid, output.auxiliary_valid_mask):
            raise ValueError("target and head valid-removal masks disagree")
        auxiliary_target = (
            targets.omission_labels
            if output.auxiliary_task == "signed"
            else targets.omission_effects.abs()
        )
        if auxiliary_target.shape != (batch, slots):
            raise ValueError("auxiliary targets must have shape [batch, 4]")

    valid = output.auxiliary_valid_mask
    per_slot_ce = output.error_logit.new_zeros(batch, slots)
    if valid.any():
        per_slot_ce[valid] = F.cross_entropy(
            output.raw_auxiliary_logits[valid], auxiliary_target[valid].long(), reduction="none"
        )
    valid_counts = valid.sum(dim=1)
    per_parent_auxiliary = per_slot_ce.sum(dim=1) / valid_counts.clamp_min(1)
    per_parent_auxiliary = torch.where(
        valid_counts > 0, per_parent_auxiliary, torch.zeros_like(per_parent_auxiliary)
    )
    error_bce = per_parent_bce.mean()
    auxiliary_ce = per_parent_auxiliary.mean()
    total = error_bce + float(auxiliary_weight) * auxiliary_ce
    return ViewRiskLoss(
        total=total,
        error_bce=error_bce,
        auxiliary_ce=auxiliary_ce,
        per_parent_error_bce=per_parent_bce,
        per_parent_auxiliary_ce=per_parent_auxiliary,
    )


def count_trainable_parameters(module: nn.Module) -> int:
    """Return the exact scalar count for an instantiated head or ablation."""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


__all__ = [
    "AuxiliaryTask",
    "MAGNITUDE_EFFECT_CLASS_ORDER",
    "RelationAttentionLayer",
    "RelationAwareConfidenceHead",
    "RelationAwareHeadConfig",
    "RelationAwareHeadOutput",
    "SIGNED_EFFECT_CLASS_ORDER",
    "ViewRiskLoss",
    "compute_view_risk_loss",
    "count_trainable_parameters",
]
