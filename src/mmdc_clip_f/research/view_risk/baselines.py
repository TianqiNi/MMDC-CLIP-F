"""Fair confidence controls for the intervention-supervised view-risk study.

This module contains model and fitting primitives only.  It deliberately does
not provide a training loop, metric implementation, checkpoint selector, or
CLI.  Every output ranks the prediction made by the frozen classifier supplied
with the input; no control is allowed to replace that prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from math import sqrt
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mmdc_clip_f.backbones import get_backbone
from mmdc_clip_f.confidence.model import MVACNConfig, MVACNHead

from .cache import CachedTargets
from .fusion import FusionPairs, RSNA_FUSION_PAIRS, fuse_view_logits, validate_fusion_pairs
from .head import RelationAwareConfidenceHead, RelationAwareHeadConfig, count_trainable_parameters
from .head_inputs import PreparedHeadInputs, RawHeadInputs, ViewRiskInputPreparation
from .inputs import CANONICAL_VIEWS, NUM_CLASSES, canonicalize_observed_views
from .roles import Operation, Role, RoleManifest, run_with_role_access


VILU_PRIMARY_SOURCE_NOTE: Mapping[str, str] = MappingProxyType(
    {
        "paper": "https://arxiv.org/html/2507.07620v1#S4.SS2",
        "authors_repository": "https://github.com/ykrmm/ViLU",
        "architecture": "learned visual-query and class-text-key/value cross-attention",
        "source_loss": "batch-weighted BCE with log(1 + correct_count/error_count)",
        "study_adaptation": "explicit unweighted BCE; not an exact reproduction of source weighting",
    }
)


SAME_INPUT_FEATURE_ORDER = (
    "canonical_view_features",
    "canonical_omission_features",
    "global_features_scores_probabilities_evidence_vacuity",
    "observed_mask",
    "removal_valid_mask",
    "no_removal",
)


class ScoreOrientation(str, Enum):
    """Which end of a scalar ranking denotes safer or riskier predictions."""

    HIGHER_CONFIDENCE = "higher_is_more_confident"
    HIGHER_ERROR = "higher_is_more_likely_error"


class ProbabilityStatus(str, Enum):
    """Whether a value has a defensible probability interpretation."""

    NOT_A_PROBABILITY = "not_a_probability"
    CLASSIFIER_CLASS_PROBABILITY = "classifier_class_probability_not_correctness_calibration"
    TUNE_CALIBRATED_ERROR_PROBABILITY = "tune_fitted_monotone_error_probability"
    LEARNED_ERROR_PROBABILITY = "learned_error_probability_not_demonstrated_calibrated"


@dataclass(frozen=True)
class ScalarScore:
    values: Tensor
    orientation: ScoreOrientation
    probability_status: ProbabilityStatus
    classifier_prediction: Tensor
    no_removal: Tensor | None = None


def _validate_scores(scores: Tensor) -> None:
    if not isinstance(scores, Tensor) or scores.ndim != 2 or scores.shape[1] != NUM_CLASSES:
        raise ValueError("scores must have shape [batch, 4]")
    if not scores.is_floating_point():
        raise TypeError("scores must be floating point")
    if scores.requires_grad:
        raise ValueError("scores must be detached from the frozen classifier")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")


def _validate_binary_targets(targets: Tensor, batch: int, device: torch.device) -> Tensor:
    if not isinstance(targets, Tensor) or targets.ndim != 1 or targets.shape[0] != batch:
        raise ValueError("binary targets must have shape [batch]")
    if targets.device != device:
        raise ValueError("targets and features must share a device")
    if targets.is_floating_point() or ((targets != 0) & (targets != 1)).any():
        raise ValueError("binary targets must contain integer 0 or 1")
    return targets


def _authorize_rows(
    manifest: RoleManifest,
    *,
    operation: Operation,
    role: Role,
    row_count: int,
    exam_keys: Sequence[str],
    expected_exam_keys: Sequence[str] | None = None,
) -> tuple[int, ...]:
    """Bind fitting rows to selected records and return authoritative densities."""

    if not isinstance(manifest, RoleManifest):
        raise TypeError("manifest must be a RoleManifest")
    selected_records = run_with_role_access(
        manifest,
        operation=operation,
        roles=(role,),
        loader=lambda records: tuple(records),
    )
    if not selected_records:
        raise PermissionError(f"manifest has no records authorized for {operation.value}")
    selected_keys = tuple(record.exam_key for record in selected_records)
    supplied_keys = tuple(exam_keys)
    expected_keys = (
        selected_keys if expected_exam_keys is None else tuple(expected_exam_keys)
    )
    if (
        len(supplied_keys) != row_count
        or supplied_keys != expected_keys
        or any(key not in set(selected_keys) for key in supplied_keys)
        or (expected_exam_keys is None and len(set(supplied_keys)) != len(supplied_keys))
    ):
        raise ValueError("tensor rows are not identity-bound to the authorized schedule")
    densities = {record.exam_key: record.density for record in selected_records}
    return tuple(densities[key] for key in supplied_keys)


class TemperatureScaler:
    """One positive temperature, fitted only on tune-set multiclass NLL."""

    def __init__(self) -> None:
        self._temperature: Tensor | None = None

    @property
    def temperature(self) -> Tensor:
        if self._temperature is None:
            raise RuntimeError("temperature has not been tune-fitted")
        return self._temperature

    def fit(
        self,
        scores: Tensor,
        labels: Tensor,
        *,
        manifest: RoleManifest,
        exam_keys: Sequence[str],
        max_iter: int = 64,
    ) -> "TemperatureScaler":
        _validate_scores(scores)
        if not isinstance(labels, Tensor) or labels.shape != (scores.shape[0],):
            raise ValueError("temperature labels must have shape [batch]")
        if labels.device != scores.device or labels.is_floating_point():
            raise ValueError("temperature labels must be integer values on the scores device")
        if ((labels < 0) | (labels >= NUM_CLASSES)).any():
            raise ValueError("temperature labels must be in range 0..3")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be positive")
        manifest_labels = _authorize_rows(
            manifest,
            operation=Operation.TUNE_SELECTION,
            role=Role.TUNE,
            row_count=scores.shape[0],
            exam_keys=exam_keys,
        )
        authoritative = torch.tensor(manifest_labels, dtype=torch.long, device=labels.device)
        if not torch.equal(labels.long(), authoritative):
            raise ValueError("temperature labels disagree with manifest densities")
        log_temperature = torch.zeros(
            (), dtype=scores.dtype, device=scores.device, requires_grad=True
        )
        optimizer = torch.optim.LBFGS(
            (log_temperature,), max_iter=max_iter, line_search_fn="strong_wolfe"
        )

        def closure() -> Tensor:
            optimizer.zero_grad()
            temperature = log_temperature.clamp(-12, 12).exp()
            loss = F.cross_entropy(scores / temperature, labels.long())
            loss.backward()
            return loss

        optimizer.step(closure)
        self._temperature = log_temperature.detach().clamp(-12, 12).exp()
        return self

    def transform(self, scores: Tensor) -> Tensor:
        _validate_scores(scores)
        temperature = self.temperature.to(device=scores.device, dtype=scores.dtype)
        scaled = scores / temperature
        if not torch.equal(scaled.argmax(dim=1), scores.argmax(dim=1)):
            raise RuntimeError("positive temperature unexpectedly changed frozen predictions")
        return scaled


def scalar_baseline_scores(
    scores: Tensor,
    *,
    temperature: TemperatureScaler | None = None,
) -> Mapping[str, ScalarScore]:
    """Return protocol scalar controls with explicit orientation and status."""

    _validate_scores(scores)
    prediction = scores.argmax(dim=1)
    probabilities = scores.softmax(dim=1)
    top_two = probabilities.topk(2, dim=1).values
    result: dict[str, ScalarScore] = {
        "msp": ScalarScore(
            top_two[:, 0],
            ScoreOrientation.HIGHER_CONFIDENCE,
            ProbabilityStatus.CLASSIFIER_CLASS_PROBABILITY,
            prediction,
        ),
        "margin": ScalarScore(
            top_two[:, 0] - top_two[:, 1],
            ScoreOrientation.HIGHER_CONFIDENCE,
            ProbabilityStatus.NOT_A_PROBABILITY,
            prediction,
        ),
        "negative_entropy": ScalarScore(
            (probabilities * probabilities.clamp_min(torch.finfo(scores.dtype).tiny).log()).sum(1),
            ScoreOrientation.HIGHER_CONFIDENCE,
            ProbabilityStatus.NOT_A_PROBABILITY,
            prediction,
        ),
        "energy": ScalarScore(
            -torch.logsumexp(scores, dim=1),
            ScoreOrientation.HIGHER_ERROR,
            ProbabilityStatus.NOT_A_PROBABILITY,
            prediction,
        ),
    }
    if temperature is not None:
        scaled = temperature.transform(scores)
        result["temperature_scaled_msp"] = ScalarScore(
            scaled.softmax(1).max(1).values,
            ScoreOrientation.HIGHER_CONFIDENCE,
            ProbabilityStatus.CLASSIFIER_CLASS_PROBABILITY,
            prediction,
        )
    return MappingProxyType(result)


def absolute_omission_sensitivity(
    omission_score_differences: Tensor,
    removal_valid_mask: Tensor,
    *,
    classifier_prediction: Tensor | None = None,
) -> ScalarScore:
    """Maximum mean absolute parent/child score change over valid removals."""

    if (
        not isinstance(omission_score_differences, Tensor)
        or omission_score_differences.ndim != 3
        or omission_score_differences.shape[1:] != (4, NUM_CLASSES)
        or not omission_score_differences.is_floating_point()
    ):
        raise ValueError("omission_score_differences must have shape [batch, 4, 4]")
    batch = omission_score_differences.shape[0]
    if removal_valid_mask.ndim == 1:
        removal_valid_mask = removal_valid_mask.unsqueeze(0).expand(batch, -1)
    if removal_valid_mask.shape != (batch, 4) or removal_valid_mask.dtype != torch.bool:
        raise ValueError("removal_valid_mask must be boolean with shape [batch, 4]")
    if not torch.isfinite(omission_score_differences).all():
        raise ValueError("omission score differences must be finite")
    per_removal = omission_score_differences.abs().mean(dim=2)
    per_removal = per_removal.masked_fill(~removal_valid_mask, -torch.inf)
    no_removal = ~removal_valid_mask.any(dim=1)
    values = per_removal.max(dim=1).values
    values = torch.where(no_removal, torch.zeros_like(values), values)
    if classifier_prediction is None:
        classifier_prediction = torch.full(
            (batch,), -1, dtype=torch.long, device=omission_score_differences.device
        )
    elif classifier_prediction.shape != (batch,) or classifier_prediction.dtype != torch.long:
        raise ValueError("classifier_prediction must have shape [batch] and dtype long")
    return ScalarScore(
        values,
        ScoreOrientation.HIGHER_ERROR,
        ProbabilityStatus.NOT_A_PROBABILITY,
        classifier_prediction,
        no_removal,
    )


class MonotoneLogisticProbabilityAdapter:
    """Tune-fitted monotone map from a scalar ranking to error probability."""

    def __init__(self, orientation: ScoreOrientation) -> None:
        self.orientation = ScoreOrientation(orientation)
        self._slope: Tensor | None = None
        self._intercept: Tensor | None = None

    @property
    def probability_status(self) -> ProbabilityStatus:
        if self._slope is None:
            return ProbabilityStatus.NOT_A_PROBABILITY
        return ProbabilityStatus.TUNE_CALIBRATED_ERROR_PROBABILITY

    def _error_oriented(self, score: Tensor) -> Tensor:
        return score if self.orientation is ScoreOrientation.HIGHER_ERROR else -score

    def fit(
        self,
        score: Tensor,
        errors: Tensor,
        *,
        manifest: RoleManifest,
        exam_keys: Sequence[str],
        max_iter: int = 64,
    ) -> "MonotoneLogisticProbabilityAdapter":
        if not isinstance(score, Tensor) or score.ndim != 1 or not score.is_floating_point():
            raise ValueError("scalar score must be a floating tensor with shape [batch]")
        if score.requires_grad or not torch.isfinite(score).all():
            raise ValueError("scalar score must be finite and detached")
        errors = _validate_binary_targets(errors, score.shape[0], score.device)
        _authorize_rows(
            manifest,
            operation=Operation.TUNE_SELECTION,
            role=Role.TUNE,
            row_count=score.shape[0],
            exam_keys=exam_keys,
        )
        raw_slope = torch.zeros((), dtype=score.dtype, device=score.device, requires_grad=True)
        intercept = torch.zeros((), dtype=score.dtype, device=score.device, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            (raw_slope, intercept), max_iter=max_iter, line_search_fn="strong_wolfe"
        )
        x = self._error_oriented(score)

        def closure() -> Tensor:
            optimizer.zero_grad()
            slope = F.softplus(raw_slope) + torch.finfo(score.dtype).eps
            loss = F.binary_cross_entropy_with_logits(slope * x + intercept, errors.to(score.dtype))
            loss.backward()
            return loss

        optimizer.step(closure)
        self._slope = F.softplus(raw_slope.detach()) + torch.finfo(score.dtype).eps
        self._intercept = intercept.detach()
        return self

    def predict_error_probability(self, score: Tensor) -> Tensor:
        if self._slope is None or self._intercept is None:
            raise RuntimeError(
                "scalar probability requires a tune-fitted monotone logistic adapter"
            )
        if not isinstance(score, Tensor) or score.ndim != 1 or not torch.isfinite(score).all():
            raise ValueError("scalar score must be finite with shape [batch]")
        x = self._error_oriented(score)
        return torch.sigmoid(
            self._slope.to(x.device, x.dtype) * x + self._intercept.to(x.device, x.dtype)
        )


DS_FEATURE_ORDER = (
    "fused_vacuity",
    "view_vacuity_L_CC",
    "view_vacuity_L_MLO",
    "view_vacuity_R_CC",
    "view_vacuity_R_MLO",
    "conflict_branch_1",
    "conflict_branch_2",
    "conflict_root",
)


@dataclass(frozen=True)
class DSBaselineFeatures:
    values: Tensor
    valid_mask: Tensor
    observed_mask: Tensor
    conflict_valid_mask: Tensor
    fusion_pairs: FusionPairs
    feature_order: tuple[str, ...] = field(default=DS_FEATURE_ORDER, init=False)

    def as_logistic_input(self) -> Tensor:
        """Values plus explicit validity/missingness bits, in stable order."""

        return torch.cat(
            (self.values.masked_fill(~self.valid_mask, 0), self.valid_mask.to(self.values.dtype)),
            dim=1,
        )


def build_ds_features(
    logits_by_view: Mapping[str, Tensor],
    observed: Iterable[str] | Sequence[bool] | Tensor,
    *,
    fusion_pairs: Sequence[Sequence[str]] = RSNA_FUSION_PAIRS,
) -> DSBaselineFeatures:
    """Extract vacuity and semantic tree conflicts from the accepted DS path."""

    pairs = validate_fusion_pairs(fusion_pairs)
    observed_views = canonicalize_observed_views(observed)
    result = fuse_view_logits(logits_by_view, observed_views, fusion_pairs=pairs)
    batch = result.scores.shape[0]
    values = result.scores.new_zeros(batch, len(DS_FEATURE_ORDER))
    valid = torch.zeros(batch, len(DS_FEATURE_ORDER), dtype=torch.bool, device=result.scores.device)
    observed_mask = torch.zeros(batch, 4, dtype=torch.bool, device=result.scores.device)
    values[:, 0] = NUM_CLASSES / result.fused_alpha.sum(dim=1)
    valid[:, 0] = True
    observed_set = set(observed_views)
    for column, view in enumerate(CANONICAL_VIEWS):
        if view not in observed_set:
            continue
        alpha = F.softplus(logits_by_view[view]) + 1
        values[:, 1 + column] = NUM_CLASSES / alpha.sum(dim=1)
        valid[:, 1 + column] = True
        observed_mask[:, column] = True

    branch_valid = tuple(all(view in observed_set for view in pair) for pair in pairs)
    root_valid = any(view in observed_set for view in pairs[0]) and any(
        view in observed_set for view in pairs[1]
    )
    conflict_valid = branch_valid + (root_valid,)
    active_index = 0
    for slot, is_valid in enumerate(conflict_valid):
        if not is_valid:
            continue
        if active_index >= len(result.stats.conflicts):
            raise RuntimeError(
                "accepted fusion conflict trace is shorter than its configured operations"
            )
        values[:, 5 + slot] = result.stats.conflicts[active_index]
        valid[:, 5 + slot] = True
        active_index += 1
    if active_index != len(result.stats.conflicts):
        raise RuntimeError(
            "accepted fusion conflict trace is longer than its configured operations"
        )
    conflict_mask = (
        torch.tensor(conflict_valid, dtype=torch.bool, device=result.scores.device)
        .unsqueeze(0)
        .expand(batch, -1)
    )
    return DSBaselineFeatures(values, valid, observed_mask, conflict_mask, pairs)


def _fit_logistic(
    features: Tensor,
    errors: Tensor,
    regularization: float,
    *,
    max_iter: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    standardized = (features - mean) / scale
    weight = torch.zeros(
        features.shape[1], dtype=features.dtype, device=features.device, requires_grad=True
    )
    bias = torch.zeros((), dtype=features.dtype, device=features.device, requires_grad=True)
    optimizer = torch.optim.LBFGS((weight, bias), max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> Tensor:
        optimizer.zero_grad()
        logits = standardized @ weight + bias
        loss = F.binary_cross_entropy_with_logits(logits, errors.to(features.dtype))
        loss = loss + 0.5 * regularization * weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return mean.detach(), scale.detach(), weight.detach(), bias.detach()


@dataclass(frozen=True)
class DSLogisticErrorControl:
    """Confidence-fit scaler/weights with tune-only regularization selection."""

    mean: Tensor
    scale: Tensor
    weight: Tensor
    bias: Tensor
    selected_regularization: float
    selection_trials: int
    probability_status: ProbabilityStatus = ProbabilityStatus.LEARNED_ERROR_PROBABILITY

    @classmethod
    def fit_with_tune_selection(
        cls,
        confidence_features: DSBaselineFeatures,
        confidence_errors: Tensor,
        tune_features: DSBaselineFeatures,
        tune_errors: Tensor,
        *,
        confidence_manifest: RoleManifest,
        confidence_exam_keys: Sequence[str],
        tune_manifest: RoleManifest,
        tune_exam_keys: Sequence[str],
        confidence_expected_exam_keys: Sequence[str] | None = None,
        tune_expected_exam_keys: Sequence[str] | None = None,
        regularizations: Sequence[float] = (0.0, 1e-4, 1e-3, 1e-2),
        max_iter: int = 64,
    ) -> "DSLogisticErrorControl":
        if not isinstance(confidence_features, DSBaselineFeatures) or not isinstance(
            tune_features, DSBaselineFeatures
        ):
            raise TypeError("DS logistic fitting requires DSBaselineFeatures with validity masks")
        confidence_matrix = confidence_features.as_logistic_input()
        tune_matrix = tune_features.as_logistic_input()
        if confidence_matrix.shape[1] != tune_matrix.shape[1]:
            raise ValueError("confidence-fit and tune feature widths must match")
        confidence_errors = _validate_binary_targets(
            confidence_errors, confidence_matrix.shape[0], confidence_matrix.device
        )
        tune_errors = _validate_binary_targets(
            tune_errors, tune_matrix.shape[0], tune_matrix.device
        )
        if confidence_matrix.device != tune_matrix.device:
            raise ValueError("confidence-fit and tune tensors must share a device")
        candidates = tuple(float(value) for value in regularizations)
        if not candidates or len(set(candidates)) != len(candidates):
            raise ValueError("regularizations must be a nonempty finite set without duplicates")
        if any(not torch.isfinite(torch.tensor(value)) or value < 0 for value in candidates):
            raise ValueError("regularizations must be finite and nonnegative")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be positive")
        _authorize_rows(
            confidence_manifest,
            operation=Operation.CONFIDENCE_FITTING,
            role=Role.CONFIDENCE_FIT,
            row_count=confidence_matrix.shape[0],
            exam_keys=confidence_exam_keys,
            expected_exam_keys=confidence_expected_exam_keys,
        )
        _authorize_rows(
            tune_manifest,
            operation=Operation.TUNE_SELECTION,
            role=Role.TUNE,
            row_count=tune_matrix.shape[0],
            exam_keys=tune_exam_keys,
            expected_exam_keys=tune_expected_exam_keys,
        )
        fitted: list[tuple[float, Tensor, Tensor, Tensor, Tensor, float]] = []
        for regularization in candidates:
            mean, scale, weight, bias = _fit_logistic(
                confidence_matrix, confidence_errors, regularization, max_iter=max_iter
            )
            tune_logits = ((tune_matrix - mean) / scale) @ weight + bias
            tune_nll = F.binary_cross_entropy_with_logits(
                tune_logits, tune_errors.to(tune_logits.dtype)
            ).item()
            fitted.append((regularization, mean, scale, weight, bias, tune_nll))
        selected = min(fitted, key=lambda item: (item[-1], item[0]))
        return cls(
            mean=selected[1],
            scale=selected[2],
            weight=selected[3],
            bias=selected[4],
            selected_regularization=selected[0],
            selection_trials=len(candidates),
        )

    def predict_error_probability(self, features: DSBaselineFeatures) -> Tensor:
        if not isinstance(features, DSBaselineFeatures):
            raise TypeError("DS logistic prediction requires DSBaselineFeatures")
        matrix = features.as_logistic_input()
        if matrix.shape[1] != self.weight.numel():
            raise ValueError("DS logistic features have an invalid shape")
        logits = (
            (matrix - self.mean.to(matrix.device, matrix.dtype))
            / self.scale.to(matrix.device, matrix.dtype)
        ) @ self.weight.to(matrix.device, matrix.dtype) + self.bias.to(matrix.device, matrix.dtype)
        return torch.sigmoid(logits)


class MVACNObjective(str, Enum):
    TCP_MSE = "tcp_mse"
    CORRECTNESS_BCE = "correctness_bce"


@dataclass(frozen=True)
class MVACNOutput:
    confidence_logit: Tensor
    confidence: Tensor
    classifier_prediction: Tensor
    current_scores: Tensor
    observed_mask: Tensor
    removal_valid_mask: Tensor


def _validate_legacy_view_order(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("legacy_view_order must list each canonical view exactly once")
    order = tuple(value)
    if len(order) != len(CANONICAL_VIEWS) or set(order) != set(CANONICAL_VIEWS):
        raise ValueError("legacy_view_order must list each canonical view exactly once")
    return order


class MaskedMVACNAdapter(nn.Module):
    """Owned legacy MV-ACN head with explicit configured token order."""

    def __init__(
        self,
        config: MVACNConfig,
        *,
        legacy_view_order: Sequence[str],
    ) -> None:
        super().__init__()
        self.legacy_view_order = _validate_legacy_view_order(legacy_view_order)
        self.head = MVACNHead(config)

    @property
    def config(self) -> MVACNConfig:
        return self.head.config

    @classmethod
    def for_backbone(
        cls,
        backbone: str,
        *,
        legacy_view_order: Sequence[str],
    ) -> "MaskedMVACNAdapter":
        spec = get_backbone(backbone)
        if spec.name == "vit_b_32":
            config = MVACNConfig(hidden_dim=spec.hidden_size, dropout=0.2, drop_cls_tokens=True)
        else:
            config = MVACNConfig(hidden_dim=spec.hidden_size, dropout=0.3, drop_cls_tokens=False)
        return cls(config, legacy_view_order=legacy_view_order)

    @classmethod
    def from_legacy(
        cls,
        legacy: MVACNHead,
        *,
        legacy_view_order: Sequence[str],
    ) -> "MaskedMVACNAdapter":
        if not isinstance(legacy, MVACNHead):
            raise TypeError("legacy must be an MVACNHead")
        result = cls(legacy.config, legacy_view_order=legacy_view_order)
        result.head.load_state_dict(legacy.state_dict(), strict=True)
        return result

    @staticmethod
    def output_from_logit(
        logit: Tensor,
        classifier_prediction: Tensor,
        *,
        current_scores: Tensor,
        observed_mask: Tensor,
    ) -> MVACNOutput:
        if logit.ndim != 1 or classifier_prediction.shape != logit.shape:
            raise ValueError("logit and classifier prediction must have shape [batch]")
        removal_valid_mask = observed_mask & (observed_mask.sum(1, keepdim=True) > 1)
        _validate_output_input_binding(
            classifier_prediction,
            current_scores,
            observed_mask,
            removal_valid_mask,
        )
        return MVACNOutput(
            confidence_logit=logit,
            confidence=torch.sigmoid(logit),
            classifier_prediction=classifier_prediction,
            current_scores=current_scores,
            observed_mask=observed_mask,
            removal_valid_mask=removal_valid_mask,
        )

    def forward(
        self,
        hidden_by_view: Mapping[str, Tensor],
        observed: Iterable[str] | Sequence[bool] | Tensor,
        *,
        classifier_prediction: Tensor,
        current_scores: Tensor,
    ) -> MVACNOutput:
        observed_views = canonicalize_observed_views(observed)
        missing = set(observed_views).difference(hidden_by_view)
        if missing:
            raise ValueError(f"hidden_by_view is missing observed views: {sorted(missing)}")
        ordered_observed_views = tuple(
            view for view in self.legacy_view_order if view in observed_views
        )
        views = tuple(hidden_by_view[view] for view in ordered_observed_views)
        reference = views[0]
        if reference.ndim != 3:
            raise ValueError("each hidden view must have shape [batch, tokens, hidden_dim]")
        batch = reference.shape[0]
        for view in views:
            if view.ndim != 3 or view.shape[0] != batch or view.shape[2] != self.config.hidden_dim:
                raise ValueError("observed hidden views have inconsistent shapes")
            if view.requires_grad or not torch.isfinite(view).all():
                raise ValueError("MV-ACN hidden states must be finite and frozen/detached")
            if self.config.drop_cls_tokens and view.shape[1] < 2:
                raise ValueError("cannot exclude CLS from a singleton token sequence")
        if classifier_prediction.shape != (batch,) or classifier_prediction.dtype != torch.long:
            raise ValueError("classifier_prediction must be long with shape [batch]")
        if set(observed_views) == set(CANONICAL_VIEWS):
            # This direct legacy branch is the exact compatibility contract.
            confidence_logit = self.head(*(hidden_by_view[view] for view in self.legacy_view_order))
        else:
            prepared = (
                tuple(view[:, 1:] for view in views) if self.config.drop_cls_tokens else views
            )
            hidden = torch.cat(prepared, dim=1)
            token = self.head.confidence_token.expand(batch, -1, -1)
            hidden = torch.cat((token, hidden), dim=1)
            attended, _ = self.head.attention(hidden, hidden, hidden, need_weights=False)
            pooled = self.head.normalization(attended[:, 0])
            confidence_logit = self.head.mlp(pooled).squeeze(-1)
        mask = (
            torch.tensor(
                [view in observed_views for view in CANONICAL_VIEWS],
                dtype=torch.bool,
                device=reference.device,
            )
            .unsqueeze(0)
            .expand(batch, -1)
        )
        removal_valid_mask = mask & (mask.sum(1, keepdim=True) > 1)
        _validate_output_input_binding(
            classifier_prediction,
            current_scores,
            mask,
            removal_valid_mask,
        )
        return MVACNOutput(
            confidence_logit=confidence_logit,
            confidence=torch.sigmoid(confidence_logit),
            classifier_prediction=classifier_prediction,
            current_scores=current_scores,
            observed_mask=mask,
            removal_valid_mask=removal_valid_mask,
        )


def _validate_output_input_binding(
    prediction: Tensor,
    current_scores: Tensor,
    observed_mask: Tensor,
    removal_valid_mask: Tensor,
) -> None:
    """Validate the label-free current parent identity carried by an output."""

    _validate_scores(current_scores)
    batch = current_scores.shape[0]
    if prediction.shape != (batch,) or prediction.dtype != torch.long:
        raise ValueError("classifier_prediction must be long with shape [batch]")
    if observed_mask.shape != (batch, 4) or observed_mask.dtype != torch.bool:
        raise ValueError("current observed mask must be boolean with shape [batch, 4]")
    if removal_valid_mask.shape != (batch, 4) or removal_valid_mask.dtype != torch.bool:
        raise ValueError("current removal mask must be boolean with shape [batch, 4]")
    if any(
        value.device != current_scores.device
        for value in (prediction, observed_mask, removal_valid_mask)
    ):
        raise ValueError("current score, prediction, and masks must share one device")
    if not observed_mask.any(1).all():
        raise ValueError("current observed mask must retain at least one view")
    if not torch.equal(observed_mask, observed_mask[:1].expand_as(observed_mask)):
        raise ValueError("current observed mask must be batch-shared")
    expected_removal = observed_mask & (observed_mask.sum(1, keepdim=True) > 1)
    if not torch.equal(removal_valid_mask, expected_removal):
        raise ValueError("current removal mask disagrees with observed views")
    if not torch.equal(prediction, current_scores.argmax(1)):
        raise ValueError("classifier prediction disagrees with current input scores")


def _validate_current_cached_targets(
    targets: CachedTargets,
    prediction: Tensor,
    current_scores: Tensor,
    observed_mask: Tensor,
    removal_valid_mask: Tensor,
) -> None:
    if not isinstance(targets, CachedTargets):
        raise TypeError("objectives require current realized CachedTargets")
    _validate_output_input_binding(
        prediction,
        current_scores,
        observed_mask,
        removal_valid_mask,
    )
    batch = prediction.shape[0]
    if (
        targets.scores.shape != (batch, NUM_CLASSES)
        or targets.probabilities.shape != targets.scores.shape
    ):
        raise ValueError("current realized CachedTargets have invalid score shapes")
    if targets.labels.shape != (batch,) or targets.tcp.shape != (batch,):
        raise ValueError("current realized CachedTargets have invalid target shapes")
    if (
        targets.valid_removal_mask.shape != (4,)
        or targets.valid_removal_mask.dtype != torch.bool
        or targets.valid_removal_mask.device != current_scores.device
    ):
        raise ValueError("current realized CachedTargets have an invalid removal mask")
    if not torch.equal(current_scores, targets.scores):
        raise ValueError("current input scores disagree with CachedTargets scores")
    target_removal = targets.valid_removal_mask.unsqueeze(0).expand(batch, -1)
    if not torch.equal(removal_valid_mask, target_removal):
        raise ValueError("current input mask disagrees with CachedTargets removal mask")
    expected_prediction = targets.scores.argmax(1)
    expected_probabilities = targets.scores.softmax(1)
    expected_error = (expected_prediction != targets.labels).long()
    expected_tcp = expected_probabilities.gather(1, targets.labels[:, None]).squeeze(1)
    if not torch.equal(expected_prediction, targets.observed_prediction):
        raise ValueError("current realized targets disagree with frozen classifier scores")
    if not torch.equal(prediction, targets.observed_prediction):
        raise ValueError(
            "confidence output replaced or mismatched the frozen classifier prediction"
        )
    if not torch.equal(expected_error, targets.observed_error):
        raise ValueError("current realized correctness targets are stale")
    if not torch.allclose(expected_probabilities, targets.probabilities, atol=1e-6, rtol=1e-6):
        raise ValueError("current realized class probabilities are stale")
    if not torch.allclose(expected_tcp, targets.tcp, atol=1e-6, rtol=1e-6):
        raise ValueError("current realized TCP is stale")


def compute_mvacn_objective(
    output: MVACNOutput,
    targets: CachedTargets,
    objective: MVACNObjective,
) -> Tensor:
    """TCP/MSE or correctness/BCE against the exact realized cache target."""

    if not isinstance(output, MVACNOutput):
        raise TypeError("output must be MVACNOutput")
    objective = MVACNObjective(objective)
    _validate_current_cached_targets(
        targets,
        output.classifier_prediction,
        output.current_scores,
        output.observed_mask,
        output.removal_valid_mask,
    )
    if objective is MVACNObjective.TCP_MSE:
        return F.mse_loss(output.confidence, targets.tcp.to(output.confidence.dtype))
    correctness = 1 - targets.observed_error.to(output.confidence_logit.dtype)
    return F.binary_cross_entropy_with_logits(output.confidence_logit, correctness)


@dataclass(frozen=True)
class ViLUOutput:
    error_logit: Tensor
    error_probability: Tensor
    confidence: Tensor
    attention_weights: Tensor
    classifier_prediction: Tensor
    predicted_text_embedding: Tensor
    current_scores: Tensor
    observed_mask: Tensor
    removal_valid_mask: Tensor


class ViLUFailureAdapter(nn.Module):
    """Multi-view ViLU adaptation using learned visual-query/text-key/value XA."""

    def __init__(
        self,
        *,
        visual_width: int,
        text_embeddings: Tensor,
        hidden_dim: int = 128,
        mlp_hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if visual_width <= 0 or hidden_dim <= 0 or mlp_hidden_dim <= 0:
            raise ValueError("ViLU dimensions must be positive")
        if (
            not isinstance(text_embeddings, Tensor)
            or text_embeddings.ndim != 2
            or text_embeddings.shape[0] != NUM_CLASSES
            or not text_embeddings.is_floating_point()
            or not torch.isfinite(text_embeddings).all()
        ):
            raise ValueError("text_embeddings must be finite with shape [4, text_width]")
        if text_embeddings.requires_grad:
            raise ValueError("ViLU text embeddings must be frozen/detached")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.visual_width = visual_width
        self.text_width = int(text_embeddings.shape[1])
        self.hidden_dim = hidden_dim
        self.register_buffer("text_embeddings", text_embeddings.detach().clone(), persistent=True)
        self.query = nn.Linear(visual_width, hidden_dim)
        self.key = nn.Linear(self.text_width, hidden_dim)
        self.value = nn.Linear(self.text_width, hidden_dim)
        self.failure_mlp = nn.Sequential(
            nn.Linear(visual_width + self.text_width + hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 1),
        )

    def forward(
        self,
        projected_visual_embeddings: Tensor,
        observed_mask: Tensor,
        *,
        classifier_prediction: Tensor,
        current_scores: Tensor,
    ) -> ViLUOutput:
        if (
            projected_visual_embeddings.ndim != 3
            or projected_visual_embeddings.shape[1:] != (4, self.visual_width)
            or not projected_visual_embeddings.is_floating_point()
        ):
            raise ValueError("projected visual embeddings must have shape [batch, 4, visual_width]")
        batch = projected_visual_embeddings.shape[0]
        if observed_mask.shape != (batch, 4) or observed_mask.dtype != torch.bool:
            raise ValueError("observed_mask must be boolean with shape [batch, 4]")
        if not observed_mask.any(1).all():
            raise ValueError("ViLU requires at least one observed visual embedding")
        if classifier_prediction.shape != (batch,) or classifier_prediction.dtype != torch.long:
            raise ValueError("classifier_prediction must be long with shape [batch]")
        if ((classifier_prediction < 0) | (classifier_prediction >= NUM_CLASSES)).any():
            raise ValueError("classifier_prediction must be in range 0..3")
        if (
            projected_visual_embeddings.requires_grad
            or not torch.isfinite(projected_visual_embeddings).all()
        ):
            raise ValueError("ViLU visual embeddings must be finite and frozen/detached")
        weights = observed_mask.unsqueeze(-1).to(projected_visual_embeddings.dtype)
        pooled_visual = (projected_visual_embeddings * weights).sum(1) / weights.sum(1)
        text = self.text_embeddings.to(
            projected_visual_embeddings.device, projected_visual_embeddings.dtype
        )
        query = self.query(pooled_visual)
        keys = self.key(text)
        values = self.value(text)
        attention_weights = torch.softmax(
            query @ keys.transpose(0, 1) / sqrt(self.hidden_dim), dim=1
        )
        attended_text = attention_weights @ values
        predicted_text = text[classifier_prediction]
        representation = torch.cat((pooled_visual, predicted_text, attended_text), dim=1)
        error_logit = self.failure_mlp(representation).squeeze(1)
        error_probability = torch.sigmoid(error_logit)
        removal_valid_mask = observed_mask & (observed_mask.sum(1, keepdim=True) > 1)
        _validate_output_input_binding(
            classifier_prediction,
            current_scores,
            observed_mask,
            removal_valid_mask,
        )
        return ViLUOutput(
            error_logit=error_logit,
            error_probability=error_probability,
            confidence=1 - error_probability,
            attention_weights=attention_weights,
            classifier_prediction=classifier_prediction,
            predicted_text_embedding=predicted_text,
            current_scores=current_scores,
            observed_mask=observed_mask,
            removal_valid_mask=removal_valid_mask,
        )


def compute_vilu_failure_loss(output: ViLUOutput, targets: CachedTargets) -> Tensor:
    """Unweighted matched BCE adaptation (not ViLU's original weighted BCE)."""

    if not isinstance(output, ViLUOutput):
        raise TypeError("output must be ViLUOutput")
    _validate_current_cached_targets(
        targets,
        output.classifier_prediction,
        output.current_scores,
        output.observed_mask,
        output.removal_valid_mask,
    )
    return F.binary_cross_entropy_with_logits(
        output.error_logit, targets.observed_error.to(output.error_logit.dtype)
    )


def flatten_same_input_features(prepared: PreparedHeadInputs) -> Tensor:
    """Canonical candidate feature slots followed only by their validity masks."""

    if not isinstance(prepared, PreparedHeadInputs):
        raise TypeError("prepared must be PreparedHeadInputs")
    batch = prepared.view_features.shape[0]
    return torch.cat(
        (
            prepared.view_features.reshape(batch, -1),
            prepared.omission_features.reshape(batch, -1),
            prepared.global_features,
            prepared.observed_mask.to(prepared.view_features.dtype),
            prepared.removal_valid_mask.to(prepared.view_features.dtype),
            prepared.no_removal.to(prepared.view_features.dtype),
        ),
        dim=1,
    )


class LearnedAffineScaler(nn.Module):
    """Control-owned affine scaling, learned only with that control's fit data."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(feature_dim))
        self.shift = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, value: Tensor) -> Tensor:
        return value * self.log_scale.clamp(-12, 12).exp() + self.shift


@dataclass(frozen=True)
class SameInputOutput:
    error_logit: Tensor
    error_probability: Tensor
    confidence: Tensor
    classifier_prediction: Tensor
    current_scores: Tensor
    observed_mask: Tensor
    removal_valid_mask: Tensor


@dataclass(frozen=True)
class DensityControlOutput:
    class_logits: Tensor
    frozen_prediction_probability: Tensor
    confidence: Tensor
    classifier_prediction: Tensor
    current_scores: Tensor
    observed_mask: Tensor
    removal_valid_mask: Tensor


def candidate_parameter_count(backbone: str) -> int:
    """Compute the accepted default candidate budget at runtime."""

    return count_trainable_parameters(
        RelationAwareConfidenceHead(RelationAwareHeadConfig(backbone=backbone))
    )


def _same_input_dimensions(preparation: ViewRiskInputPreparation) -> tuple[int, int]:
    continuous = (
        4 * preparation.hidden_dim
        + 4 * preparation.omission_feature_dim
        + preparation.global_feature_dim
    )
    return continuous, continuous + 9  # observed(4), removable(4), singleton/no-removal(1)


class _CapacityMatchedBase(nn.Module):
    def __init__(self, backbone: str, output_dim: int) -> None:
        super().__init__()
        spec = get_backbone(backbone)
        self.backbone = spec.name
        self.preparation = ViewRiskInputPreparation(spec.hidden_size)
        continuous_dim, input_dim = _same_input_dimensions(self.preparation)
        self.continuous_dim = continuous_dim
        self.input_dim = input_dim
        self.scaler = LearnedAffineScaler(continuous_dim)
        target = candidate_parameter_count(spec.name)
        fixed = count_trainable_parameters(self.preparation) + count_trainable_parameters(
            self.scaler
        )
        coefficient = input_dim + output_dim + 1
        hidden_dim = max(1, round((target - fixed - output_dim) / coefficient))
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )
        actual = count_trainable_parameters(self)
        if abs(actual - target) / target > 0.10:
            raise RuntimeError("capacity-matched control is outside the 10% candidate budget")

    def _features(self, raw: RawHeadInputs) -> tuple[Tensor, PreparedHeadInputs]:
        if raw.backbone != self.backbone:
            raise ValueError("raw input backbone does not match same-input control")
        prepared = self.preparation(raw)
        flat = flatten_same_input_features(prepared)
        continuous = self.scaler(flat[:, : self.continuous_dim])
        return torch.cat((continuous, flat[:, self.continuous_dim :]), dim=1), prepared


class SameInputMLP(_CapacityMatchedBase):
    """Capacity-matched binary MLP over exactly the candidate's prepared input."""

    def __init__(self, backbone: str) -> None:
        super().__init__(backbone, 1)

    def forward(self, raw: RawHeadInputs) -> SameInputOutput:
        features, prepared = self._features(raw)
        error_logit = self.network(features).squeeze(1)
        error_probability = torch.sigmoid(error_logit)
        _validate_output_input_binding(
            prepared.current_prediction,
            raw.current_scores,
            prepared.observed_mask,
            prepared.removal_valid_mask,
        )
        return SameInputOutput(
            error_logit=error_logit,
            error_probability=error_probability,
            confidence=1 - error_probability,
            classifier_prediction=prepared.current_prediction,
            current_scores=raw.current_scores,
            observed_mask=prepared.observed_mask,
            removal_valid_mask=prepared.removal_valid_mask,
        )


def gather_frozen_prediction_probability(
    class_logits: Tensor, classifier_prediction: Tensor
) -> Tensor:
    if class_logits.ndim != 2 or class_logits.shape[1] != NUM_CLASSES:
        raise ValueError("class_logits must have shape [batch, 4]")
    if (
        classifier_prediction.shape != (class_logits.shape[0],)
        or classifier_prediction.dtype != torch.long
    ):
        raise ValueError("classifier_prediction must be long with shape [batch]")
    if ((classifier_prediction < 0) | (classifier_prediction >= NUM_CLASSES)).any():
        raise ValueError("classifier_prediction must be in range 0..3")
    return class_logits.softmax(1).gather(1, classifier_prediction[:, None]).squeeze(1)


class SameInputDensityControl(_CapacityMatchedBase):
    """Four-class control scored only at the frozen classifier prediction."""

    def __init__(self, backbone: str) -> None:
        super().__init__(backbone, NUM_CLASSES)

    def forward(self, raw: RawHeadInputs) -> DensityControlOutput:
        features, prepared = self._features(raw)
        class_logits = self.network(features)
        probability = gather_frozen_prediction_probability(
            class_logits, prepared.current_prediction
        )
        _validate_output_input_binding(
            prepared.current_prediction,
            raw.current_scores,
            prepared.observed_mask,
            prepared.removal_valid_mask,
        )
        return DensityControlOutput(
            class_logits=class_logits,
            frozen_prediction_probability=probability,
            confidence=probability,
            classifier_prediction=prepared.current_prediction,
            current_scores=raw.current_scores,
            observed_mask=prepared.observed_mask,
            removal_valid_mask=prepared.removal_valid_mask,
        )


def compute_same_input_error_loss(output: SameInputOutput, targets: CachedTargets) -> Tensor:
    """Unweighted current-error BCE without accepting labels as input features."""

    if not isinstance(output, SameInputOutput):
        raise TypeError("output must be SameInputOutput")
    _validate_current_cached_targets(
        targets,
        output.classifier_prediction,
        output.current_scores,
        output.observed_mask,
        output.removal_valid_mask,
    )
    return F.binary_cross_entropy_with_logits(
        output.error_logit, targets.observed_error.to(output.error_logit.dtype)
    )


def compute_density_control_loss(output: DensityControlOutput, targets: CachedTargets) -> Tensor:
    """Density CE; confidence remains gathered at the frozen classifier prediction."""

    if not isinstance(output, DensityControlOutput):
        raise TypeError("output must be DensityControlOutput")
    _validate_current_cached_targets(
        targets,
        output.classifier_prediction,
        output.current_scores,
        output.observed_mask,
        output.removal_valid_mask,
    )
    return F.cross_entropy(output.class_logits, targets.labels.long())


@dataclass(frozen=True)
class BaselineDefinition:
    objective: str
    input_exposure: str
    inference_features: str
    requires_current_realized_targets: bool
    loss_weighting: str = "unweighted"
    claims_exact_source_loss: bool = True
    actual_exposure_verified: bool = False


BASELINE_DEFINITIONS: Mapping[str, BaselineDefinition] = MappingProxyType(
    {
        "msp": BaselineDefinition(
            "raw_ranking", "same_realized_input", "frozen_classifier_scores", False
        ),
        "temperature_scaled_msp": BaselineDefinition(
            "tune_nll_then_raw_ranking", "same_realized_input", "frozen_classifier_scores", False
        ),
        "margin": BaselineDefinition(
            "raw_ranking", "same_realized_input", "frozen_classifier_scores", False
        ),
        "negative_entropy": BaselineDefinition(
            "raw_ranking", "same_realized_input", "frozen_classifier_scores", False
        ),
        "energy": BaselineDefinition(
            "raw_error_ranking", "same_realized_input", "frozen_classifier_scores", False
        ),
        "ds_logistic": BaselineDefinition(
            "error_logistic_bce",
            "same_realized_input",
            "fused_and_per_view_vacuity+stable_tree_conflicts+validity_masks",
            True,
        ),
        "absolute_omission_sensitivity": BaselineDefinition(
            "raw_error_ranking+tune_logistic_for_brier",
            "same_realized_input",
            "valid_parent_child_score_differences+no_removal",
            False,
        ),
        "original_mvacn_tcp": BaselineDefinition(
            "tcp_mse", "clean_four_view", "legacy_hidden_tokens", True
        ),
        "correctness_mvacn": BaselineDefinition(
            "correctness_bce", "clean_with_labeled_masked_adaptation", "legacy_hidden_tokens", True
        ),
        "matched_mvacn_tcp": BaselineDefinition(
            "tcp_mse", "matched_augmentation_and_masks", "legacy_hidden_tokens", True
        ),
        "matched_mvacn_correctness": BaselineDefinition(
            "correctness_bce", "matched_augmentation_and_masks", "legacy_hidden_tokens", True
        ),
        "vilu": BaselineDefinition(
            "failure_bce",
            "matched_augmentation_and_masks",
            "pooled_observed_projected_visual+frozen_predicted_text+attended_class_text",
            True,
            loss_weighting="unweighted_bce_adaptation",
            claims_exact_source_loss=False,
        ),
        "same_input_mlp": BaselineDefinition(
            "error_bce",
            "matched_augmentation_and_masks",
            "exact_candidate_prepared_features_and_masks",
            True,
        ),
        "same_input_density": BaselineDefinition(
            "density_ce",
            "matched_augmentation_and_masks",
            "exact_candidate_prepared_features_and_masks",
            True,
        ),
    }
)


@dataclass(frozen=True)
class P3AAblationDefinition:
    head_config: RelationAwareHeadConfig
    auxiliary_weight: float
    input_exposure: str


def p3a_ablation_definitions(
    base: RelationAwareHeadConfig,
) -> Mapping[str, P3AAblationDefinition]:
    """Configure accepted P3A head/loss interfaces without copying the candidate."""

    if not isinstance(base, RelationAwareHeadConfig):
        raise TypeError("base must be RelationAwareHeadConfig")
    matched = "matched_augmentation_and_masks"
    return MappingProxyType(
        {
            "no_effect_supervision": P3AAblationDefinition(base, 0.0, matched),
            "no_intervention_features": P3AAblationDefinition(
                replace(base, use_omission_features=False), 1.0, matched
            ),
            "no_view_relations": P3AAblationDefinition(
                replace(base, use_relation_biases=False), 1.0, matched
            ),
            "magnitude_effects": P3AAblationDefinition(
                replace(base, auxiliary_task="magnitude"), 1.0, matched
            ),
            "corruption_auxiliary": P3AAblationDefinition(
                replace(base, auxiliary_task="corruption"), 1.0, matched
            ),
            "hidden_only": P3AAblationDefinition(
                replace(base, input_mode="hidden_only"), 1.0, matched
            ),
            "evidence_only": P3AAblationDefinition(
                replace(base, input_mode="evidence_only"), 1.0, matched
            ),
            "clean_fitting": P3AAblationDefinition(base, 1.0, "clean_parents_and_masks"),
        }
    )


# Fixed architecture/calibration candidate counts.  These are not claims that
# checkpoint exposure has already been matched; P4 must freeze and audit its
# common finite checkpoint/epoch table before any pilot access.
CONTROL_SELECTION_BUDGET: Mapping[str, int] = MappingProxyType(
    {
        "common_architecture_trials_per_method": 1,
        "training_seeds": 3,
        "rsna_checkpoint_candidates_per_seed": 20,
        "ddsm_checkpoint_candidates_per_seed": 50,
        "temperature": 1,
        "scalar_logistic": 1,
        "ds_regularization": 4,
        "mvacn_architecture": 1,
        "vilu_architecture": 1,
        "same_input_architecture": 1,
        "p3a_ablation_architecture": 1,
    }
)


__all__ = [
    "BASELINE_DEFINITIONS",
    "CONTROL_SELECTION_BUDGET",
    "DSBaselineFeatures",
    "DSLogisticErrorControl",
    "DensityControlOutput",
    "LearnedAffineScaler",
    "MVACNObjective",
    "MVACNOutput",
    "MaskedMVACNAdapter",
    "MonotoneLogisticProbabilityAdapter",
    "P3AAblationDefinition",
    "ProbabilityStatus",
    "SameInputDensityControl",
    "SameInputMLP",
    "SameInputOutput",
    "SAME_INPUT_FEATURE_ORDER",
    "ScalarScore",
    "ScoreOrientation",
    "TemperatureScaler",
    "ViLUFailureAdapter",
    "ViLUOutput",
    "VILU_PRIMARY_SOURCE_NOTE",
    "absolute_omission_sensitivity",
    "build_ds_features",
    "candidate_parameter_count",
    "compute_density_control_loss",
    "compute_mvacn_objective",
    "compute_same_input_error_loss",
    "compute_vilu_failure_loss",
    "flatten_same_input_features",
    "gather_frozen_prediction_probability",
    "p3a_ablation_definitions",
    "scalar_baseline_scores",
]
