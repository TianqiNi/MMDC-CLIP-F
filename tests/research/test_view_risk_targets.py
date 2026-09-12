from __future__ import annotations

import pytest
import torch
from torch import Tensor

from mmdc_clip_f.research.view_risk import CANONICAL_VIEWS, build_intervention_targets


def _raw_logits_for_evidence(evidence: Tensor) -> Tensor:
    return torch.log(torch.expm1(evidence))


def _effect_fixture() -> tuple[dict[str, Tensor], Tensor]:
    baseline = 0.2
    left_cc = torch.tensor(
        [
            [3.0, 10.0, baseline, baseline],
            [10.0, 3.0, baseline, baseline],
            [baseline, 8.0, 2.0, baseline],
        ],
        dtype=torch.float64,
    )
    left_mlo = torch.tensor(
        [
            [3.0, 1.0, baseline, baseline],
            [1.0, 3.0, baseline, baseline],
            [baseline, 7.0, 1.0, baseline],
        ],
        dtype=torch.float64,
    )
    logits = {
        "L_CC": _raw_logits_for_evidence(left_cc),
        "L_MLO": _raw_logits_for_evidence(left_mlo),
    }
    return logits, torch.tensor([0, 0, 2], dtype=torch.long)


def test_targets_encode_repair_damage_and_unchanged_from_actual_omissions() -> None:
    logits, labels = _effect_fixture()

    targets = build_intervention_targets(logits, labels, ("L_MLO", "L_CC"))

    assert targets.view_order == CANONICAL_VIEWS
    assert targets.observed_views == ("L_CC", "L_MLO")
    torch.testing.assert_close(targets.observed_prediction, torch.tensor([1, 0, 1]))
    torch.testing.assert_close(targets.observed_error, torch.tensor([1, 0, 1]))
    torch.testing.assert_close(
        targets.valid_removal_mask,
        torch.tensor([True, True, False, False]),
    )
    torch.testing.assert_close(
        targets.omission_predictions,
        torch.tensor([[0, 1, -1, -1], [1, 0, -1, -1], [1, 1, -1, -1]]),
    )
    # delta = observed error - omission error: +1 repairs, -1 damages, 0 is unchanged.
    torch.testing.assert_close(
        targets.omission_effects,
        torch.tensor([[1, 0, -1, -1], [-1, 0, -1, -1], [0, 0, -1, -1]]),
    )
    torch.testing.assert_close(
        targets.omission_labels,
        torch.tensor([[2, 1, -1, -1], [0, 1, -1, -1], [1, 1, -1, -1]]),
    )
    # The third sample remains wrong with the same class prediction after either omission.
    assert targets.observed_prediction[2].item() == 1
    assert targets.omission_predictions[2, :2].tolist() == [1, 1]
    assert targets.omission_effects[2, :2].tolist() == [0, 0]


def test_singleton_has_global_error_target_but_no_valid_removal() -> None:
    logits, labels = _effect_fixture()

    targets = build_intervention_targets({"L_CC": logits["L_CC"]}, labels, ("L_CC",))

    torch.testing.assert_close(targets.observed_prediction, torch.tensor([1, 0, 1]))
    torch.testing.assert_close(targets.observed_error, torch.tensor([1, 0, 1]))
    assert not targets.valid_removal_mask.any()
    assert torch.equal(targets.omission_predictions, torch.full((3, 4), -1, dtype=torch.long))
    assert torch.equal(targets.omission_effects, torch.full((3, 4), -1, dtype=torch.long))
    assert torch.equal(targets.omission_labels, torch.full((3, 4), -1, dtype=torch.long))


@pytest.mark.parametrize(
    ("labels", "match"),
    [
        (torch.tensor([[0], [0], [2]]), "one-dimensional"),
        (torch.tensor([0, 0]), "batch"),
        (torch.tensor([0.0, 0.0, 2.0]), "integer dtype"),
        (torch.tensor([0, -1, 2]), "range"),
        (torch.tensor([0, 4, 2]), "range"),
    ],
)
def test_malformed_labels_fail_clearly(labels: Tensor, match: str) -> None:
    logits, _ = _effect_fixture()

    with pytest.raises((TypeError, ValueError), match=match):
        build_intervention_targets(logits, labels, ("L_CC", "L_MLO"))


def test_target_generation_does_not_mutate_or_backpropagate_into_classifier_logits() -> None:
    logits, labels = _effect_fixture()
    trainable_logits = {
        view: value.detach().clone().requires_grad_(True) for view, value in logits.items()
    }
    before = {view: value.detach().clone() for view, value in trainable_logits.items()}

    targets = build_intervention_targets(trainable_logits, labels, ("L_CC", "L_MLO"))

    assert not targets.observed_error.requires_grad
    assert not targets.omission_effects.requires_grad
    for view, value in trainable_logits.items():
        assert value.grad is None
        torch.testing.assert_close(value.detach(), before[view])
