"""Research metrics and patient-paired inference for view-risk evaluation.

The records in this module may contain private patient and exam keys while they
are in memory.  They are evaluation inputs, not serializable result artifacts;
aggregate result objects deliberately retain no identifiers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .inputs import CANONICAL_VIEWS
from .perturbations import EvaluationCell, enumerate_evaluation_panels


ConfidenceKind = Literal["probability", "ranking"]
PanelName = Literal[
    "clean_four_view",
    "clean_masks",
    "primary",
    "strong_seen",
    "common_mode",
]

_PANEL_NAMES = {
    "clean_four_view",
    "clean_masks",
    "primary",
    "strong_seen",
    "common_mode",
}


@dataclass(frozen=True)
class AveragePrecisionResult:
    """Grouped noninterpolated AP and the class counts behind availability."""

    value: float | None
    positive_count: int
    negative_count: int


@dataclass(frozen=True)
class CoverageRisk:
    """Expected selective risk at a requested and realized rank coverage."""

    requested_coverage: float
    selected_count: int
    total_count: int
    realized_coverage: float
    risk: float


@dataclass(frozen=True)
class ConfidencePanelMetrics:
    """Exam-level confidence metrics for one already-defined panel slice."""

    n_exams: int
    correct_count: int
    error_count: int
    accuracy: float
    error_prevalence: float
    aurc: float
    risk_at_80: CoverageRisk
    risk_at_90: CoverageRisk
    error_positive_ap: AveragePrecisionResult
    correct_positive_ap: AveragePrecisionResult
    brier: float | None
    confidence_kind: ConfidenceKind


def _binary_vector(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    try:
        numeric = array.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only 0 and 1") from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError(f"{name} must contain only 0 and 1")
    return numeric.astype(np.int64)


def _score_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    try:
        numeric = array.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} must contain only finite values")
    return numeric


def _integer_weights(weights: np.ndarray | None, size: int) -> np.ndarray:
    if weights is None:
        return np.ones(size, dtype=np.int64)
    array = np.asarray(weights)
    if array.shape != (size,):
        raise ValueError("weights must have one value per exam")
    try:
        numeric = array.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("weights must be nonnegative integers") from exc
    if (
        not np.isfinite(numeric).all()
        or (numeric < 0).any()
        or not np.equal(numeric, np.floor(numeric)).all()
    ):
        raise ValueError("weights must be nonnegative integers")
    result = numeric.astype(np.int64)
    if int(result.sum()) == 0:
        raise ValueError("weights must retain at least one exam")
    return result


def _group_boundaries(sorted_scores: np.ndarray) -> tuple[tuple[int, int], ...]:
    starts = np.flatnonzero(
        np.concatenate((np.array([True]), sorted_scores[1:] != sorted_scores[:-1]))
    )
    ends = np.concatenate((starts[1:], np.array([len(sorted_scores)])))
    return tuple((int(start), int(end)) for start, end in zip(starts, ends))


def _expected_aurc_and_risks(
    errors: np.ndarray,
    confidence: np.ndarray,
    weights: np.ndarray | None = None,
    coverages: tuple[float, ...] = (0.8, 0.9),
) -> tuple[float, tuple[CoverageRisk, ...]]:
    integer_weights = _integer_weights(weights, len(errors))
    order = np.argsort(-confidence, kind="stable")
    sorted_scores = confidence[order]
    sorted_errors = errors[order]
    sorted_weights = integer_weights[order]
    total = int(sorted_weights.sum())
    coverage_ranks = tuple(int(math.ceil(coverage * total)) for coverage in coverages)
    coverage_values: list[float | None] = [None] * len(coverages)

    cumulative_count = 0
    cumulative_errors = 0.0
    risk_sum = 0.0
    harmonic = np.concatenate((np.array([0.0]), np.cumsum(1 / np.arange(1, total + 1))))
    for start, end in _group_boundaries(sorted_scores):
        group_weights = sorted_weights[start:end]
        group_size = int(group_weights.sum())
        if group_size == 0:
            continue
        group_errors = float(np.dot(group_weights, sorted_errors[start:end]))
        delta_harmonic = harmonic[cumulative_count + group_size] - harmonic[cumulative_count]
        # Sum j=1..m of (S + j E/m)/(a+j), without expanding bootstrap copies.
        risk_sum += cumulative_errors * delta_harmonic
        risk_sum += (group_errors / group_size) * (
            group_size - cumulative_count * delta_harmonic
        )
        for index, rank in enumerate(coverage_ranks):
            if coverage_values[index] is None and cumulative_count < rank <= (
                cumulative_count + group_size
            ):
                within_group = rank - cumulative_count
                coverage_values[index] = (
                    cumulative_errors + within_group * group_errors / group_size
                ) / rank
        cumulative_count += group_size
        cumulative_errors += group_errors

    risks = tuple(
        CoverageRisk(
            requested_coverage=coverage,
            selected_count=rank,
            total_count=total,
            realized_coverage=rank / total,
            risk=float(value),
        )
        for coverage, rank, value in zip(coverages, coverage_ranks, coverage_values)
        if value is not None
    )
    if len(risks) != len(coverages):
        raise RuntimeError("internal coverage calculation did not reach every requested rank")
    return float(risk_sum / total), risks


def _grouped_average_precision(
    positive: np.ndarray,
    score: np.ndarray,
    weights: np.ndarray | None = None,
) -> AveragePrecisionResult:
    integer_weights = _integer_weights(weights, len(positive))
    positive_count = int(np.dot(integer_weights, positive))
    total = int(integer_weights.sum())
    result = AveragePrecisionResult(None, positive_count, total - positive_count)
    if positive_count == 0 or positive_count == total:
        return result

    order = np.argsort(-score, kind="stable")
    sorted_scores = score[order]
    sorted_positive = positive[order]
    sorted_weights = integer_weights[order]
    cumulative_positive = 0
    cumulative_count = 0
    average_precision = 0.0
    for start, end in _group_boundaries(sorted_scores):
        group_weights = sorted_weights[start:end]
        group_count = int(group_weights.sum())
        if group_count == 0:
            continue
        group_positive = int(np.dot(group_weights, sorted_positive[start:end]))
        cumulative_positive += group_positive
        cumulative_count += group_count
        average_precision += (
            group_positive / positive_count
        ) * cumulative_positive / cumulative_count
    return AveragePrecisionResult(float(average_precision), positive_count, total - positive_count)


def confidence_panel_metrics(
    correct: Sequence[int] | np.ndarray,
    confidence: Sequence[float] | np.ndarray,
    *,
    confidence_kind: ConfidenceKind,
) -> ConfidencePanelMetrics:
    """Compute the protocol's confidence metrics for one panel.

    ``confidence_kind`` is mandatory.  A raw ranking score is valid for AURC
    and both AP orientations, but only a declared correctness probability is
    used for Brier score.
    """

    correct_array = _binary_vector(correct, "correct")
    confidence_array = _score_vector(confidence, "confidence")
    if len(correct_array) != len(confidence_array):
        raise ValueError("correct and confidence must have the same length")
    if confidence_kind not in ("probability", "ranking"):
        raise ValueError("confidence_kind must be 'probability' or 'ranking'")
    if confidence_kind == "probability" and (
        (confidence_array < 0).any() or (confidence_array > 1).any()
    ):
        raise ValueError("probability confidence must be in [0, 1]")

    errors = 1 - correct_array
    aurc, coverage_risks = _expected_aurc_and_risks(errors, confidence_array)
    correct_ap = _grouped_average_precision(correct_array, confidence_array)
    error_ap = _grouped_average_precision(errors, -confidence_array)
    n_exams = len(correct_array)
    correct_count = int(correct_array.sum())
    brier = (
        float(np.mean((confidence_array - correct_array) ** 2))
        if confidence_kind == "probability"
        else None
    )
    return ConfidencePanelMetrics(
        n_exams=n_exams,
        correct_count=correct_count,
        error_count=n_exams - correct_count,
        accuracy=correct_count / n_exams,
        error_prevalence=(n_exams - correct_count) / n_exams,
        aurc=aurc,
        risk_at_80=coverage_risks[0],
        risk_at_90=coverage_risks[1],
        error_positive_ap=error_ap,
        correct_positive_ap=correct_ap,
        brier=brier,
        confidence_kind=confidence_kind,
    )


@dataclass(frozen=True)
class EffectClassificationMetrics:
    """Three-way effect metrics in row-truth/column-prediction class order."""

    confusion_matrix: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]
    support: tuple[int, int, int]
    recall: tuple[float | None, float | None, float | None]
    balanced_accuracy: float | None
    f1: tuple[float | None, float | None, float | None]
    macro_f1: float | None


@dataclass(frozen=True)
class EffectStratumMetrics:
    """Observed effect performance and the majority-unchanged reference."""

    n_parents: int
    n_valid: int
    singleton_parent_count: int
    actual: EffectClassificationMetrics
    majority_zero_reference: EffectClassificationMetrics


@dataclass(frozen=True)
class EffectMetrics:
    """Overall and non-overlapping clean/stressed constrained-effect panels."""

    class_order: tuple[int, int, int]
    overall: EffectStratumMetrics
    clean: EffectStratumMetrics
    stressed: EffectStratumMetrics


def _classification_metrics(target: np.ndarray, prediction: np.ndarray) -> EffectClassificationMetrics:
    confusion = np.zeros((3, 3), dtype=np.int64)
    for truth, predicted in zip(target, prediction):
        confusion[int(truth), int(predicted)] += 1
    support_array = confusion.sum(axis=1)
    recall: list[float | None] = []
    f1: list[float | None] = []
    for index in range(3):
        support = int(support_array[index])
        if support == 0:
            recall.append(None)
            f1.append(None)
            continue
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum()) - true_positive
        false_negative = support - true_positive
        recall.append(true_positive / support)
        denominator = 2 * true_positive + false_positive + false_negative
        f1.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    available_recall = [value for value in recall if value is not None]
    available_f1 = [value for value in f1 if value is not None]
    return EffectClassificationMetrics(
        confusion_matrix=tuple(tuple(int(value) for value in row) for row in confusion),  # type: ignore[arg-type]
        support=tuple(int(value) for value in support_array),  # type: ignore[arg-type]
        recall=tuple(recall),  # type: ignore[arg-type]
        balanced_accuracy=(
            float(np.mean(available_recall)) if available_recall else None
        ),
        f1=tuple(f1),  # type: ignore[arg-type]
        macro_f1=float(np.mean(available_f1)) if available_f1 else None,
    )


def _effect_stratum(
    target_index: np.ndarray,
    prediction_index: np.ndarray,
    valid: np.ndarray,
    parent_selector: np.ndarray,
) -> EffectStratumMetrics:
    selected_valid = valid[parent_selector]
    flattened_selector = selected_valid.reshape(-1)
    selected_target = target_index[parent_selector].reshape(-1)[flattened_selector]
    selected_prediction = prediction_index[parent_selector].reshape(-1)[flattened_selector]
    n_valid = len(selected_target)
    majority_zero = np.full(n_valid, 1, dtype=np.int64)
    return EffectStratumMetrics(
        n_parents=int(parent_selector.sum()),
        n_valid=n_valid,
        singleton_parent_count=int(np.sum(selected_valid.sum(axis=1) == 0)),
        actual=_classification_metrics(selected_target, selected_prediction),
        majority_zero_reference=_classification_metrics(selected_target, majority_zero),
    )


def effect_metrics(
    effect_targets: Sequence[Sequence[int]] | np.ndarray,
    reported_probabilities: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    valid_slots: Sequence[Sequence[bool]] | np.ndarray,
    *,
    stressed: Sequence[bool] | np.ndarray,
) -> EffectMetrics:
    """Score constrained probabilities only at valid removable-view slots.

    Probability columns are fixed to effect classes ``[-1, 0, +1]``.  Invalid
    missing-view and singleton slots are not interpreted as unchanged targets.
    """

    targets = np.asarray(effect_targets)
    probabilities = np.asarray(reported_probabilities, dtype=np.float64)
    valid = np.asarray(valid_slots)
    stressed_array = np.asarray(stressed)
    if targets.ndim != 2 or targets.size == 0:
        raise ValueError("effect_targets must be a non-empty two-dimensional array")
    if probabilities.shape != (*targets.shape, 3):
        raise ValueError("reported_probabilities must have shape [parents, slots, 3]")
    if valid.shape != targets.shape or valid.dtype != np.bool_:
        raise ValueError("valid_slots must be a boolean array matching effect_targets")
    if stressed_array.shape != (targets.shape[0],) or stressed_array.dtype != np.bool_:
        raise ValueError("stressed must be a boolean value for every parent")

    valid_probabilities = probabilities[valid]
    if (
        not np.isfinite(valid_probabilities).all()
        or (valid_probabilities < 0).any()
        or (valid_probabilities > 1).any()
    ):
        raise ValueError("valid reported effect probabilities must be finite and in [0, 1]")
    if not np.allclose(valid_probabilities.sum(axis=1), 1.0, rtol=1e-7, atol=1e-7):
        raise ValueError("valid reported effect probabilities must sum to one")
    try:
        valid_targets = targets[valid].astype(np.int64)
        exact_targets = targets[valid].astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("valid effect targets must use class values -1, 0, or +1") from exc
    if (
        not np.isfinite(exact_targets).all()
        or not np.equal(exact_targets, valid_targets).all()
        or not np.isin(valid_targets, (-1, 0, 1)).all()
    ):
        raise ValueError("valid effect targets must use class values -1, 0, or +1")

    target_index = np.zeros(targets.shape, dtype=np.int64)
    target_index[valid] = valid_targets + 1
    prediction_index = np.zeros(targets.shape, dtype=np.int64)
    prediction_index[valid] = np.argmax(valid_probabilities, axis=1)
    all_parents = np.ones(targets.shape[0], dtype=np.bool_)
    return EffectMetrics(
        class_order=(-1, 0, 1),
        overall=_effect_stratum(target_index, prediction_index, valid, all_parents),
        clean=_effect_stratum(target_index, prediction_index, valid, ~stressed_array),
        stressed=_effect_stratum(target_index, prediction_index, valid, stressed_array),
    )


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class EvaluationPrediction:
    """One frozen-classifier prediction and one method's confidence score.

    Patient and exam identifiers remain private runtime inputs.  ``cohort`` is
    an immutable manifest/provenance identity used to reject mixed evaluations.
    """

    dataset: str
    role: str
    cohort: str
    method: str
    training_seed: int
    patient_id: str
    exam_id: str
    panel: PanelName
    target: int
    prediction: int
    confidence: float
    confidence_kind: ConfidenceKind
    family: str | None = None
    severity: str | None = None
    target_view: str | None = None
    variant: str | None = None
    mask: tuple[str, ...] = tuple(CANONICAL_VIEWS)
    realization_id: str = "frozen"

    def __post_init__(self) -> None:
        for name in ("dataset", "role", "cohort", "method", "patient_id", "exam_id"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        object.__setattr__(self, "realization_id", _required_text(self.realization_id, "realization_id"))
        if self.panel not in _PANEL_NAMES:
            raise ValueError(f"panel must be one of {sorted(_PANEL_NAMES)}")
        if isinstance(self.training_seed, bool) or not isinstance(self.training_seed, int):
            raise ValueError("training_seed must be an integer")
        for name in ("target", "prediction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value < 4:
                raise ValueError(f"{name} must be a four-class index")
        if self.confidence_kind not in ("probability", "ranking"):
            raise ValueError("confidence_kind must be 'probability' or 'ranking'")
        try:
            confidence = float(self.confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError("confidence must be finite") from exc
        if not math.isfinite(confidence):
            raise ValueError("confidence must be finite")
        if self.confidence_kind == "probability" and not 0 <= confidence <= 1:
            raise ValueError("probability confidence must be in [0, 1]")
        object.__setattr__(self, "confidence", confidence)
        mask = tuple(self.mask)
        if (
            not mask
            or len(set(mask)) != len(mask)
            or any(view not in CANONICAL_VIEWS for view in mask)
            or mask != tuple(view for view in CANONICAL_VIEWS if view in mask)
        ):
            raise ValueError("mask must be a nonempty canonical named-view subset")
        object.__setattr__(self, "mask", mask)
        if self.target_view is not None and self.target_view not in CANONICAL_VIEWS:
            raise ValueError("target_view must be canonical when present")


@dataclass(frozen=True)
class PanelCellAURC:
    """A cell metric after averaging direction/orientation variants."""

    family: str | None
    severity: str | None
    target_view: str | None
    mask: tuple[str, ...]
    variant_aurc: tuple[tuple[str | None, float], ...]
    aurc: float


@dataclass(frozen=True)
class PanelAURC:
    """A complete, separately evaluated protocol panel."""

    dataset: str
    role: str
    cohort: str
    method: str
    training_seed: int
    panel: PanelName
    n_patients: int
    n_exams: int
    n_cells: int
    mean_aurc: float
    cells: tuple[PanelCellAURC, ...]


@dataclass(frozen=True)
class _ExpectedCell:
    family: str | None
    severity: str | None
    target_view: str | None
    mask: tuple[str, ...]
    variants: tuple[str | None, ...]

    @property
    def key(self) -> tuple[object, ...]:
        return (self.family, self.severity, self.target_view, self.mask)


def _from_evaluation_cell(cell: EvaluationCell) -> _ExpectedCell:
    return _ExpectedCell(
        family=cell.family,
        severity=cell.severity,
        target_view=cell.target_view,
        mask=tuple(CANONICAL_VIEWS),
        variants=cell.variants,
    )


def _expected_cells(panel: PanelName) -> tuple[_ExpectedCell, ...]:
    panels = enumerate_evaluation_panels()
    if panel == "primary":
        return tuple(_from_evaluation_cell(cell) for cell in panels.primary_cells)
    if panel == "strong_seen":
        return tuple(_from_evaluation_cell(cell) for cell in panels.strong_seen_family_cells)
    if panel == "common_mode":
        return tuple(_from_evaluation_cell(cell) for cell in panels.common_mode_cells)
    if panel == "clean_four_view":
        return (_ExpectedCell(None, None, None, tuple(CANONICAL_VIEWS), (None,)),)
    if panel == "clean_masks":
        return tuple(_ExpectedCell(None, None, None, mask, (None,)) for mask in panels.clean_masks)
    raise ValueError(f"unknown panel: {panel!r}")


def _record_cell_key(record: EvaluationPrediction) -> tuple[object, ...]:
    return (record.family, record.severity, record.target_view, record.mask)


@dataclass(frozen=True)
class _OrganizedPanel:
    records: tuple[EvaluationPrediction, ...]
    expected_cells: tuple[_ExpectedCell, ...]
    rows_by_cell: tuple[tuple[tuple[EvaluationPrediction, ...], ...], ...]
    exam_keys: tuple[tuple[str, str], ...]


def _organize_panel(records: Sequence[EvaluationPrediction]) -> _OrganizedPanel:
    materialized = tuple(records)
    if not materialized:
        raise ValueError("panel predictions cannot be empty")
    if any(not isinstance(record, EvaluationPrediction) for record in materialized):
        raise TypeError("all panel predictions must be EvaluationPrediction records")
    panels = {record.panel for record in materialized}
    if len(panels) != 1:
        raise ValueError("predictions must contain exactly one panel")
    panel = next(iter(panels))
    metadata_fields = ("dataset", "role", "cohort", "method", "training_seed", "confidence_kind")
    for field in metadata_fields:
        if len({getattr(record, field) for record in materialized}) != 1:
            label = "one method" if field == "method" else f"one {field.replace('_', ' ')}"
            raise ValueError(f"panel predictions must belong to {label}")

    expected_cells = _expected_cells(panel)
    expected_lookup = {cell.key: cell for cell in expected_cells}
    grouped: dict[
        tuple[tuple[object, ...], str | None],
        dict[tuple[str, str], EvaluationPrediction],
    ] = {}
    exam_patient: dict[str, str] = {}
    exam_target: dict[tuple[str, str], int] = {}
    for record in materialized:
        cell_key = _record_cell_key(record)
        expected = expected_lookup.get(cell_key)
        if expected is None or record.variant not in expected.variants:
            raise ValueError(f"unexpected cell or variant in {panel} panel")
        prior_patient = exam_patient.setdefault(record.exam_id, record.patient_id)
        if prior_patient != record.patient_id:
            raise ValueError("one exam cannot belong to multiple patients")
        exam_key = (record.patient_id, record.exam_id)
        prior_target = exam_target.setdefault(exam_key, record.target)
        if prior_target != record.target:
            raise ValueError("an exam has unmatched targets across panel variants")
        group = grouped.setdefault((cell_key, record.variant), {})
        if exam_key in group:
            raise ValueError("panel contains a duplicate exam prediction")
        group[exam_key] = record

    first_group: dict[tuple[str, str], EvaluationPrediction] | None = None
    rows_by_cell: list[tuple[tuple[EvaluationPrediction, ...], ...]] = []
    for cell in expected_cells:
        variant_rows: list[tuple[EvaluationPrediction, ...]] = []
        for variant in cell.variants:
            group = grouped.get((cell.key, variant))
            if group is None:
                raise ValueError("panel is missing a required cell or variant")
            if first_group is None:
                first_group = group
            if set(group) != set(first_group):
                raise ValueError("panel has missing exams or variant alignment failure")
            variant_rows.append(tuple(group[key] for key in sorted(group)))
        rows_by_cell.append(tuple(variant_rows))
    if len(grouped) != sum(len(cell.variants) for cell in expected_cells):
        raise ValueError("panel contains unexpected duplicate cell variants")
    if first_group is None:
        raise RuntimeError("internal panel organization found no variants")
    return _OrganizedPanel(
        records=materialized,
        expected_cells=expected_cells,
        rows_by_cell=tuple(rows_by_cell),
        exam_keys=tuple(sorted(first_group)),
    )


def _variant_aurc(
    rows: tuple[EvaluationPrediction, ...],
    patient_weights: dict[str, int] | None = None,
) -> float:
    correct = np.fromiter((row.target == row.prediction for row in rows), dtype=np.int64)
    errors = 1 - correct
    confidence = np.fromiter((row.confidence for row in rows), dtype=np.float64)
    weights = (
        None
        if patient_weights is None
        else np.fromiter((patient_weights[row.patient_id] for row in rows), dtype=np.int64)
    )
    return _expected_aurc_and_risks(errors, confidence, weights)[0]


def _panel_result(
    organized: _OrganizedPanel,
    patient_weights: dict[str, int] | None = None,
) -> PanelAURC:
    cell_results: list[PanelCellAURC] = []
    for expected, cell_rows in zip(organized.expected_cells, organized.rows_by_cell):
        variant_values = tuple(
            (variant, _variant_aurc(rows, patient_weights))
            for variant, rows in zip(expected.variants, cell_rows)
        )
        cell_results.append(
            PanelCellAURC(
                family=expected.family,
                severity=expected.severity,
                target_view=expected.target_view,
                mask=expected.mask,
                variant_aurc=variant_values,
                aurc=float(np.mean([value for _, value in variant_values])),
            )
        )
    first = organized.records[0]
    cells = tuple(cell_results)
    return PanelAURC(
        dataset=first.dataset,
        role=first.role,
        cohort=first.cohort,
        method=first.method,
        training_seed=first.training_seed,
        panel=first.panel,
        n_patients=len({patient for patient, _ in organized.exam_keys}),
        n_exams=len(organized.exam_keys),
        n_cells=len(cells),
        mean_aurc=float(np.mean([cell.aurc for cell in cells])),
        cells=cells,
    )


def evaluate_aurc_panel(records: Sequence[EvaluationPrediction]) -> PanelAURC:
    """Evaluate one complete protocol panel without mixing it into another."""

    return _panel_result(_organize_panel(records))


def evaluate_primary_endpoint(records: Sequence[EvaluationPrediction]) -> PanelAURC:
    """Compute the equal-weight 48-cell primary mean AURC endpoint."""

    materialized = tuple(records)
    if materialized and any(record.panel != "primary" for record in materialized):
        raise ValueError("evaluate_primary_endpoint accepts only primary panel predictions")
    result = evaluate_aurc_panel(materialized)
    if result.n_cells != 48:
        raise RuntimeError("the authoritative primary enumeration no longer contains 48 cells")
    return result


@dataclass(frozen=True)
class BootstrapDifference:
    """A percentile interval with explicit undefined-replicate accounting."""

    training_seed: int | None
    point_difference: float
    lower: float | None
    upper: float | None
    confidence_level: float
    defined_replicates: int
    undefined_replicates: int


@dataclass(frozen=True)
class TrainingSeedSummary:
    """Fixed-seed spread; training seeds are not resampled as patients."""

    n_seeds: int
    mean: float
    sample_sd: float | None
    minimum: float
    maximum: float


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Patient-paired candidate-minus-baseline AURC inference."""

    dataset: str
    role: str
    cohort: str
    panel: PanelName
    candidate_method: str
    baseline_method: str
    n_patients: int
    n_exams: int
    patient_exam_counts: tuple[int, ...]
    n_resamples: int
    random_seed: int
    per_seed: tuple[BootstrapDifference, ...]
    seed_mean: BootstrapDifference
    seed_summary: TrainingSeedSummary


@dataclass(frozen=True)
class _PreparedVariant:
    errors: np.ndarray
    scores: np.ndarray
    patient_index: np.ndarray
    groups: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _PreparedEndpoint:
    cells: tuple[tuple[_PreparedVariant, ...], ...]


def _prepare_endpoint(organized: _OrganizedPanel, patient_lookup: dict[str, int]) -> _PreparedEndpoint:
    cells: list[tuple[_PreparedVariant, ...]] = []
    for cell_rows in organized.rows_by_cell:
        variants: list[_PreparedVariant] = []
        for rows in cell_rows:
            errors = np.fromiter((row.target != row.prediction for row in rows), dtype=np.int64)
            scores = np.fromiter((row.confidence for row in rows), dtype=np.float64)
            patient_index = np.fromiter(
                (patient_lookup[row.patient_id] for row in rows), dtype=np.int64
            )
            order = np.argsort(-scores, kind="stable")
            variants.append(
                _PreparedVariant(
                    errors=errors[order],
                    scores=scores[order],
                    patient_index=patient_index[order],
                    groups=_group_boundaries(scores[order]),
                )
            )
        cells.append(tuple(variants))
    return _PreparedEndpoint(tuple(cells))


def _prepared_aurc_draws(prepared: _PreparedEndpoint, patient_draw_counts: np.ndarray) -> np.ndarray:
    draw_count = patient_draw_counts.shape[0]
    maximum_weighted_exams = 0
    for variant in prepared.cells[0]:
        row_weights = patient_draw_counts[:, variant.patient_index]
        maximum_weighted_exams = max(maximum_weighted_exams, int(row_weights.sum(axis=1).max()))
    harmonic = np.concatenate(
        (np.array([0.0]), np.cumsum(1 / np.arange(1, maximum_weighted_exams + 1)))
    )
    endpoint_total = np.zeros(draw_count, dtype=np.float64)
    for cell in prepared.cells:
        cell_total = np.zeros(draw_count, dtype=np.float64)
        for variant in cell:
            weights = patient_draw_counts[:, variant.patient_index]
            cumulative_count = np.zeros(draw_count, dtype=np.int64)
            cumulative_errors = np.zeros(draw_count, dtype=np.float64)
            risk_sum = np.zeros(draw_count, dtype=np.float64)
            for start, end in variant.groups:
                group_weights = weights[:, start:end]
                group_size = group_weights.sum(axis=1).astype(np.int64)
                group_errors = group_weights @ variant.errors[start:end]
                active = group_size > 0
                if not active.any():
                    continue
                delta_harmonic = np.zeros(draw_count, dtype=np.float64)
                delta_harmonic[active] = (
                    harmonic[cumulative_count[active] + group_size[active]]
                    - harmonic[cumulative_count[active]]
                )
                risk_sum[active] += cumulative_errors[active] * delta_harmonic[active]
                risk_sum[active] += (group_errors[active] / group_size[active]) * (
                    group_size[active] - cumulative_count[active] * delta_harmonic[active]
                )
                cumulative_count += group_size
                cumulative_errors += group_errors
            variant_values = np.full(draw_count, np.nan, dtype=np.float64)
            retained = cumulative_count > 0
            variant_values[retained] = risk_sum[retained] / cumulative_count[retained]
            cell_total += variant_values / len(cell)
        endpoint_total += cell_total / len(prepared.cells)
    return endpoint_total


def _sample_patient_counts(
    n_patients: int, n_resamples: int, random_seed: int
) -> np.ndarray:
    generator = np.random.default_rng(random_seed)
    draws = generator.integers(0, n_patients, size=(n_resamples, n_patients))
    counts = np.zeros((n_resamples, n_patients), dtype=np.int64)
    rows = np.repeat(np.arange(n_resamples), n_patients)
    np.add.at(counts, (rows, draws.reshape(-1)), 1)
    return counts


def _sample_key(record: EvaluationPrediction) -> tuple[object, ...]:
    return (
        record.panel,
        record.family,
        record.severity,
        record.target_view,
        record.variant,
        record.mask,
        record.patient_id,
        record.exam_id,
        record.realization_id,
    )


def _bootstrap_difference(
    training_seed: int | None,
    point_difference: float,
    replicates: np.ndarray,
    confidence_level: float,
) -> BootstrapDifference:
    finite = replicates[np.isfinite(replicates)]
    undefined = len(replicates) - len(finite)
    if len(finite):
        tail = 100 * (1 - confidence_level) / 2
        lower, upper = np.percentile(finite, (tail, 100 - tail))
        bounds: tuple[float | None, float | None] = (float(lower), float(upper))
    else:
        bounds = (None, None)
    return BootstrapDifference(
        training_seed=training_seed,
        point_difference=float(point_difference),
        lower=bounds[0],
        upper=bounds[1],
        confidence_level=confidence_level,
        defined_replicates=len(finite),
        undefined_replicates=undefined,
    )


def patient_paired_bootstrap_aurc_difference(
    records: Sequence[EvaluationPrediction],
    *,
    candidate_method: str,
    baseline_method: str,
    panel: PanelName = "primary",
    n_resamples: int = 2000,
    random_seed: int = 2026,
    confidence_level: float = 0.95,
) -> PairedBootstrapResult:
    """Bootstrap a complete panel's candidate-minus-baseline mean AURC.

    One replacement draw supplies patient multiplicities to every exam, cell,
    variant, method, and training seed.  Differences are formed within each
    training seed before the same-draw seed mean is calculated.
    """

    materialized = tuple(records)
    if not materialized:
        raise ValueError("paired bootstrap predictions cannot be empty")
    if any(not isinstance(record, EvaluationPrediction) for record in materialized):
        raise TypeError("all bootstrap predictions must be EvaluationPrediction records")
    if candidate_method == baseline_method:
        raise ValueError("candidate and baseline methods must differ")
    if panel not in _PANEL_NAMES:
        raise ValueError(f"panel must be one of {sorted(_PANEL_NAMES)}")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError("random_seed must be a nonnegative integer")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    if len({record.dataset for record in materialized}) != 1:
        raise ValueError("paired bootstrap requires exactly one dataset")
    for field in ("role", "cohort", "panel"):
        if len({getattr(record, field) for record in materialized}) != 1:
            raise ValueError(f"paired bootstrap requires exactly one {field}")
    if materialized[0].panel != panel:
        raise ValueError("records do not match the requested bootstrap panel")
    if {record.method for record in materialized} != {candidate_method, baseline_method}:
        raise ValueError("records must contain exactly the candidate and baseline methods")

    seeds_by_method = {
        method: {record.training_seed for record in materialized if record.method == method}
        for method in (candidate_method, baseline_method)
    }
    if seeds_by_method[candidate_method] != seeds_by_method[baseline_method]:
        raise ValueError("candidate and baseline training seeds must align")
    training_seeds = tuple(sorted(seeds_by_method[candidate_method]))

    organized: dict[tuple[str, int], _OrganizedPanel] = {}
    keyed: dict[tuple[str, int], dict[tuple[object, ...], EvaluationPrediction]] = {}
    for method in (candidate_method, baseline_method):
        for training_seed in training_seeds:
            subset = tuple(
                record
                for record in materialized
                if record.method == method and record.training_seed == training_seed
            )
            organized[(method, training_seed)] = _organize_panel(subset)
            keyed[(method, training_seed)] = {_sample_key(record): record for record in subset}

    reference_key = (candidate_method, training_seeds[0])
    reference = keyed[reference_key]
    for method_seed, sample_records in keyed.items():
        if set(sample_records) != set(reference):
            raise ValueError("methods/training seeds have missing samples or variant alignment failure")
        for key, reference_record in reference.items():
            record = sample_records[key]
            if (record.target, record.prediction) != (
                reference_record.target,
                reference_record.prediction,
            ):
                raise ValueError("methods/training seeds do not share frozen predictions and targets")

    patient_ids = tuple(sorted({record.patient_id for record in reference.values()}))
    patient_lookup = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    exam_keys = tuple(sorted({(record.patient_id, record.exam_id) for record in reference.values()}))
    exam_counts = {patient_id: 0 for patient_id in patient_ids}
    for patient_id, _ in exam_keys:
        exam_counts[patient_id] += 1
    patient_exam_counts = tuple(sorted(exam_counts.values()))
    draw_counts = _sample_patient_counts(len(patient_ids), n_resamples, random_seed)
    point_counts = np.ones((1, len(patient_ids)), dtype=np.int64)

    prepared = {
        key: _prepare_endpoint(value, patient_lookup) for key, value in organized.items()
    }
    per_seed_results: list[BootstrapDifference] = []
    replicate_differences: list[np.ndarray] = []
    point_differences: list[float] = []
    for training_seed in training_seeds:
        candidate = prepared[(candidate_method, training_seed)]
        baseline = prepared[(baseline_method, training_seed)]
        candidate_replicates = _prepared_aurc_draws(candidate, draw_counts)
        baseline_replicates = _prepared_aurc_draws(baseline, draw_counts)
        differences = candidate_replicates - baseline_replicates
        point_difference = float(
            _prepared_aurc_draws(candidate, point_counts)[0]
            - _prepared_aurc_draws(baseline, point_counts)[0]
        )
        point_differences.append(point_difference)
        replicate_differences.append(differences)
        per_seed_results.append(
            _bootstrap_difference(training_seed, point_difference, differences, confidence_level)
        )

    stacked = np.stack(replicate_differences)
    seed_mean_replicates = np.where(
        np.isfinite(stacked).all(axis=0),
        np.mean(stacked, axis=0),
        np.nan,
    )
    seed_mean_point = float(np.mean(point_differences))
    seed_array = np.asarray(point_differences, dtype=np.float64)
    first = materialized[0]
    return PairedBootstrapResult(
        dataset=first.dataset,
        role=first.role,
        cohort=first.cohort,
        panel=panel,
        candidate_method=candidate_method,
        baseline_method=baseline_method,
        n_patients=len(patient_ids),
        n_exams=len(exam_keys),
        patient_exam_counts=patient_exam_counts,
        n_resamples=n_resamples,
        random_seed=random_seed,
        per_seed=tuple(per_seed_results),
        seed_mean=_bootstrap_difference(
            None,
            seed_mean_point,
            seed_mean_replicates,
            confidence_level,
        ),
        seed_summary=TrainingSeedSummary(
            n_seeds=len(seed_array),
            mean=seed_mean_point,
            sample_sd=float(np.std(seed_array, ddof=1)) if len(seed_array) > 1 else None,
            minimum=float(seed_array.min()),
            maximum=float(seed_array.max()),
        ),
    )


__all__ = [
    "AveragePrecisionResult",
    "BootstrapDifference",
    "ConfidencePanelMetrics",
    "CoverageRisk",
    "EffectClassificationMetrics",
    "EffectMetrics",
    "EffectStratumMetrics",
    "EvaluationPrediction",
    "PairedBootstrapResult",
    "PanelAURC",
    "PanelCellAURC",
    "TrainingSeedSummary",
    "confidence_panel_metrics",
    "effect_metrics",
    "evaluate_aurc_panel",
    "evaluate_primary_endpoint",
    "patient_paired_bootstrap_aurc_difference",
]
