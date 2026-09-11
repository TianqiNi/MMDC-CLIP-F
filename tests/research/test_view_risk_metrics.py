from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mmdc_clip_f.research.view_risk.inputs import CANONICAL_VIEWS
from mmdc_clip_f.research.view_risk.metrics import (
    EvaluationPrediction,
    confidence_panel_metrics,
    effect_metrics,
    evaluate_aurc_panel,
    evaluate_primary_endpoint,
    patient_paired_bootstrap_aurc_difference,
)
from mmdc_clip_f.research.view_risk.perturbations import enumerate_evaluation_panels


def test_aurc_uses_expected_order_inside_exact_ties_and_is_input_order_invariant() -> None:
    correct = np.array([1, 0, 1])
    confidence = np.array([0.9, 0.9, 0.1])

    forward = confidence_panel_metrics(correct, confidence, confidence_kind="probability")
    reverse = confidence_panel_metrics(
        correct[::-1], confidence[::-1], confidence_kind="probability"
    )

    # The first tie has one error in two items, so both expected prefix risks are 1/2.
    assert forward.aurc == pytest.approx((0.5 + 0.5 + 1 / 3) / 3)
    assert reverse.aurc == forward.aurc
    assert forward.risk_at_80.risk == pytest.approx(1 / 3)
    assert forward.risk_at_80.realized_coverage == 1.0
    assert forward.risk_at_90.selected_count == forward.risk_at_80.selected_count
    assert forward.risk_at_90.realized_coverage == forward.risk_at_80.realized_coverage
    assert forward.risk_at_90.risk == forward.risk_at_80.risk


def test_partial_coverage_inside_tie_uses_expected_prefix_risk() -> None:
    result = confidence_panel_metrics(
        [1, 1, 0, 1, 0],
        [1.0, 0.5, 0.5, 0.5, 0.5],
        confidence_kind="probability",
    )

    # At 80% coverage, j=3 items have been drawn from a four-item tie with E=2.
    assert result.risk_at_80.selected_count == 4
    assert result.risk_at_80.realized_coverage == 0.8
    assert result.risk_at_80.risk == pytest.approx((3 * 2 / 4) / 4)
    assert result.risk_at_90.selected_count == 5
    assert result.risk_at_90.realized_coverage == 1.0
    assert result.risk_at_90.risk == pytest.approx(2 / 5)


def test_grouped_noninterpolated_ap_reports_both_orientations_and_brier() -> None:
    result = confidence_panel_metrics(
        [1, 0, 1, 0],
        [0.9, 0.9, 0.2, 0.1],
        confidence_kind="probability",
    )

    assert result.correct_positive_ap.value == pytest.approx(7 / 12)
    assert result.error_positive_ap.value == pytest.approx(3 / 4)
    assert result.correct_positive_ap.positive_count == 2
    assert result.error_positive_ap.positive_count == 2
    assert result.brier == pytest.approx(np.mean((np.array([0.9, 0.9, 0.2, 0.1]) - [1, 0, 1, 0]) ** 2))
    assert result.accuracy == 0.5
    assert result.error_prevalence == 0.5


def test_one_class_ap_is_unavailable_with_counts() -> None:
    result = confidence_panel_metrics([1, 1], [0.2, 0.8], confidence_kind="probability")

    assert result.correct_positive_ap.value is None
    assert result.correct_positive_ap.positive_count == 2
    assert result.correct_positive_ap.negative_count == 0
    assert result.error_positive_ap.value is None
    assert result.error_positive_ap.positive_count == 0
    assert result.error_positive_ap.negative_count == 2


def test_raw_ranking_score_never_silently_receives_probability_brier() -> None:
    raw = confidence_panel_metrics([1, 0], [0.8, 0.2], confidence_kind="ranking")
    assert raw.brier is None

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        confidence_panel_metrics([1, 0], [1.2, -0.1], confidence_kind="probability")
    with pytest.raises(ValueError, match="finite"):
        confidence_panel_metrics([1, 0], [0.5, np.nan], confidence_kind="ranking")
    with pytest.raises(ValueError, match="same length"):
        confidence_panel_metrics([1], [0.5, 0.4], confidence_kind="probability")


def test_effect_metrics_use_only_valid_constrained_slots_and_report_strata() -> None:
    targets = np.array(
        [
            [99, 99, 99, 99],
            [-1, 0, 99, 99],
            [1, 99, 99, 99],
        ]
    )
    valid = np.array(
        [
            [False, False, False, False],
            [True, True, False, False],
            [True, False, False, False],
        ]
    )
    probabilities = np.full((3, 4, 3), np.nan)
    probabilities[1, 0] = [0.8, 0.1, 0.1]  # correct damage
    probabilities[1, 1] = [0.1, 0.2, 0.7]  # unchanged predicted as repair
    probabilities[2, 0] = [0.1, 0.2, 0.7]  # correct repair

    result = effect_metrics(targets, probabilities, valid, stressed=[False, False, True])

    assert result.class_order == (-1, 0, 1)
    assert result.overall.actual.confusion_matrix == ((1, 0, 0), (0, 0, 1), (0, 0, 1))
    assert result.overall.actual.support == (1, 1, 1)
    assert result.overall.actual.recall == pytest.approx((1.0, 0.0, 1.0))
    assert result.overall.actual.balanced_accuracy == pytest.approx(2 / 3)
    assert result.overall.actual.f1 == pytest.approx((1.0, 0.0, 2 / 3))
    assert result.overall.actual.macro_f1 == pytest.approx(5 / 9)
    assert result.overall.majority_zero_reference.balanced_accuracy == pytest.approx(1 / 3)
    assert result.clean.actual.support == (1, 1, 0)
    assert result.clean.actual.recall[2] is None
    assert result.clean.actual.f1[2] is None
    assert result.stressed.actual.support == (0, 0, 1)
    assert result.stressed.actual.recall[:2] == (None, None)
    assert result.overall.singleton_parent_count == 1
    assert result.clean.singleton_parent_count == 1
    assert result.stressed.singleton_parent_count == 0


def test_effect_metrics_validate_only_reportable_slots_but_require_probabilities_there() -> None:
    targets = np.array([[0, 99]])
    valid = np.array([[True, False]])
    probabilities = np.array([[[0.1, 0.8, 0.1], [np.nan, np.nan, np.nan]]])
    assert effect_metrics(targets, probabilities, valid, stressed=[False]).overall.n_valid == 1

    probabilities[0, 0] = [0.2, 0.2, 0.2]
    with pytest.raises(ValueError, match="sum to one"):
        effect_metrics(targets, probabilities, valid, stressed=[False])
    with pytest.raises(ValueError, match="valid effect targets"):
        effect_metrics([[2]], [[[0.0, 1.0, 0.0]]], [[True]], stressed=[False])


def _panel_rows(
    *,
    method: str = "candidate",
    training_seed: int = 42,
    exams: tuple[tuple[str, str], ...] = (("p0", "e0"), ("p1", "e1")),
) -> list[EvaluationPrediction]:
    rows: list[EvaluationPrediction] = []
    for cell in enumerate_evaluation_panels().primary_cells:
        for variant in cell.variants:
            for patient_id, exam_id in exams:
                prediction = 0
                confidence = 0.8
                if (
                    cell.family == "brightness"
                    and cell.severity == "mild"
                    and cell.target_view == "L_CC"
                ):
                    prediction = 1 if exam_id == exams[0][1] else 0
                    if variant == "lower":
                        confidence = 0.9 if prediction else 0.8
                    else:
                        confidence = 0.1 if prediction else 0.2
                rows.append(
                    EvaluationPrediction(
                        dataset="synthetic",
                        role="pilot",
                        cohort="manifest-v1",
                        method=method,
                        training_seed=training_seed,
                        patient_id=patient_id,
                        exam_id=exam_id,
                        panel="primary",
                        target=0,
                        prediction=prediction,
                        confidence=confidence,
                        confidence_kind="ranking",
                        family=cell.family,
                        severity=cell.severity,
                        target_view=cell.target_view,
                        variant=variant,
                        mask=CANONICAL_VIEWS,
                        realization_id=f"{exam_id}/{cell.family}/{cell.severity}/{cell.target_view}/{variant}",
                    )
                )
    return rows


def test_primary_endpoint_uses_equal_cell_then_variant_weighting() -> None:
    result = evaluate_primary_endpoint(_panel_rows())

    special = next(
        cell
        for cell in result.cells
        if (cell.family, cell.severity, cell.target_view)
        == ("brightness", "mild", "L_CC")
    )
    assert special.variant_aurc == (("lower", pytest.approx(0.75)), ("upper", pytest.approx(0.25)))
    assert special.aurc == pytest.approx(0.5)
    assert result.mean_aurc == pytest.approx(0.5 / 48)
    assert result.n_cells == 48
    assert result.n_exams == 2
    assert result.n_patients == 2


def test_primary_endpoint_refuses_missing_unmatched_or_mixed_panel_rows() -> None:
    rows = _panel_rows()
    with pytest.raises(ValueError, match="missing|alignment"):
        evaluate_primary_endpoint(rows[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_primary_endpoint([*rows, rows[0]])
    with pytest.raises(ValueError, match="one method"):
        evaluate_primary_endpoint([*rows[:-1], replace(rows[-1], method="baseline")])
    with pytest.raises(ValueError, match="only primary"):
        evaluate_primary_endpoint([replace(row, panel="common_mode") for row in rows])


def test_panels_are_evaluated_separately_and_clean_masks_must_be_complete() -> None:
    rows: list[EvaluationPrediction] = []
    for mask in enumerate_evaluation_panels().clean_masks:
        rows.append(
            EvaluationPrediction(
                dataset="synthetic",
                role="pilot",
                cohort="manifest-v1",
                method="candidate",
                training_seed=42,
                patient_id="p0",
                exam_id="e0",
                panel="clean_masks",
                target=0,
                prediction=0,
                confidence=0.5,
                confidence_kind="probability",
                mask=mask,
            )
        )

    result = evaluate_aurc_panel(rows)
    assert result.panel == "clean_masks"
    assert result.n_cells == 14
    assert result.mean_aurc == 0.0
    with pytest.raises(ValueError, match="missing"):
        evaluate_aurc_panel(rows[:-1])
    with pytest.raises(ValueError, match="one panel"):
        evaluate_aurc_panel([*rows, replace(rows[0], panel="clean_four_view", mask=CANONICAL_VIEWS)])


def _paired_rows() -> list[EvaluationPrediction]:
    exams = (("p0", "e0"), ("p0", "e1"), ("p1", "e2"))
    rows: list[EvaluationPrediction] = []
    for method in ("candidate", "baseline"):
        for training_seed in (7, 8):
            for row in _panel_rows(method=method, training_seed=training_seed, exams=exams):
                is_correct = row.prediction == row.target
                if method == "candidate":
                    confidence = 0.9 if is_correct else 0.1
                elif training_seed == 7:
                    confidence = 0.1 if is_correct else 0.9
                else:
                    confidence = 0.5
                rows.append(replace(row, confidence=confidence))
    return rows


def test_patient_paired_bootstrap_retains_unequal_exam_clusters_and_is_deterministic() -> None:
    rows = _paired_rows()
    first = patient_paired_bootstrap_aurc_difference(
        rows,
        candidate_method="candidate",
        baseline_method="baseline",
        n_resamples=40,
        random_seed=2026,
    )
    replay = patient_paired_bootstrap_aurc_difference(
        list(reversed(rows)),
        candidate_method="candidate",
        baseline_method="baseline",
        n_resamples=40,
        random_seed=2026,
    )

    assert first == replay
    assert first.patient_exam_counts == (1, 2)
    assert first.n_patients == 2
    assert first.n_exams == 3
    assert {item.training_seed for item in first.per_seed} == {7, 8}
    assert all(item.point_difference < 0 for item in first.per_seed)
    assert first.seed_mean.point_difference < 0
    assert first.seed_mean.upper <= 0
    assert first.seed_mean.undefined_replicates == 0
    assert first.seed_summary.sample_sd is not None
    assert first.seed_summary.sample_sd > 0


def test_paired_bootstrap_refuses_dataset_sample_seed_and_frozen_prediction_mismatch() -> None:
    rows = _paired_rows()
    with pytest.raises(ValueError, match="one dataset"):
        patient_paired_bootstrap_aurc_difference(
            [*rows, replace(rows[0], dataset="other")],
            candidate_method="candidate",
            baseline_method="baseline",
            n_resamples=2,
        )

    missing_variant = [
        row
        for index, row in enumerate(rows)
        if not (index == 0 and row.method == "candidate" and row.training_seed == 7)
    ]
    with pytest.raises(ValueError, match="missing|alignment"):
        patient_paired_bootstrap_aurc_difference(
            missing_variant,
            candidate_method="candidate",
            baseline_method="baseline",
            n_resamples=2,
        )

    changed_prediction = list(rows)
    index = next(i for i, row in enumerate(changed_prediction) if row.method == "baseline")
    changed_prediction[index] = replace(
        changed_prediction[index], prediction=1 - changed_prediction[index].prediction
    )
    with pytest.raises(ValueError, match="frozen predictions"):
        patient_paired_bootstrap_aurc_difference(
            changed_prediction,
            candidate_method="candidate",
            baseline_method="baseline",
            n_resamples=2,
        )

    one_missing_seed = [
        row for row in rows if not (row.method == "baseline" and row.training_seed == 8)
    ]
    with pytest.raises(ValueError, match="training seeds"):
        patient_paired_bootstrap_aurc_difference(
            one_missing_seed,
            candidate_method="candidate",
            baseline_method="baseline",
            n_resamples=2,
        )
