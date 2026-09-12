from __future__ import annotations

from itertools import combinations

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mmdc_clip_f.model import MultiViewCLIPClassifier
from mmdc_clip_f.research.view_risk import (
    CANONICAL_VIEWS,
    NoObservedViewsError,
    fuse_view_logits,
)


def _raw_logits_for_evidence(evidence: Tensor) -> Tensor:
    """Independent inverse of softplus for positive test evidence."""

    return torch.log(torch.expm1(evidence))


def _legacy_classifier() -> MultiViewCLIPClassifier:
    return MultiViewCLIPClassifier(
        nn.Identity(),
        input_order=CANONICAL_VIEWS,
        fusion_pairs=(("L_CC", "L_MLO"), ("R_CC", "R_MLO")),
    )


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_full_fusion_matches_legacy_classifier_without_mutating_inputs(dtype: torch.dtype) -> None:
    logits = {
        view: (
            torch.arange(8, dtype=dtype).reshape(2, 4) / 7
            + index * torch.tensor(0.17, dtype=dtype)
            - torch.tensor(0.8, dtype=dtype)
        )
        for index, view in enumerate(CANONICAL_VIEWS)
    }
    before = {view: value.clone() for view, value in logits.items()}

    actual = fuse_view_logits(logits, CANONICAL_VIEWS)
    legacy = _legacy_classifier()
    expected_scores = legacy.fuse_logits(logits)
    legacy_alphas = {view: F.softplus(value) + 1 for view, value in logits.items()}
    left_alpha, _ = legacy._combine_two(legacy_alphas["L_CC"], legacy_alphas["L_MLO"])
    right_alpha, _ = legacy._combine_two(legacy_alphas["R_CC"], legacy_alphas["R_MLO"])
    expected_alpha, _ = legacy._combine_two(left_alpha, right_alpha)

    tolerance = {torch.float64: (1e-12, 1e-12), torch.float32: (2e-6, 2e-6)}[dtype]
    torch.testing.assert_close(actual.scores, expected_scores, rtol=tolerance[0], atol=tolerance[1])
    torch.testing.assert_close(
        actual.fused_alpha, expected_alpha, rtol=tolerance[0], atol=tolerance[1]
    )
    assert actual.scores.dtype == dtype
    assert actual.scores.device == logits["L_CC"].device
    assert actual.scores.shape == (2, 4)
    torch.testing.assert_close(actual.scores, F.softplus(actual.fused_alpha))
    assert not torch.allclose(
        actual.scores,
        actual.fused_alpha / actual.fused_alpha.sum(dim=1, keepdim=True),
    )
    for view in CANONICAL_VIEWS:
        torch.testing.assert_close(logits[view], before[view])


def test_two_source_fusion_matches_hand_computed_evidence_identity() -> None:
    evidence_left = torch.tensor([[0.5, 1.0, 2.0, 0.25]], dtype=torch.float64)
    evidence_right = torch.tensor([[3.0, 0.75, 0.5, 1.5]], dtype=torch.float64)
    logits = {
        "L_CC": _raw_logits_for_evidence(evidence_left),
        "R_MLO": _raw_logits_for_evidence(evidence_right),
    }

    result = fuse_view_logits(logits, ("R_MLO", "L_CC"))

    # For four classes, Dempster's rule independently reduces to this evidence identity.
    expected_evidence = evidence_left + evidence_right + evidence_left * evidence_right / 4
    torch.testing.assert_close(result.fused_alpha, expected_evidence + 1, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(result.stats.evidence, expected_evidence, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        result.stats.strength, (expected_evidence + 1).sum(dim=1, keepdim=True)
    )
    torch.testing.assert_close(result.stats.uncertainty, 4 / result.stats.strength)
    assert result.observed_views == ("L_CC", "R_MLO")


def test_omission_matches_independent_vacuous_evidence_identity() -> None:
    evidence = {
        "L_CC": torch.tensor([[1.0, 0.3, 0.2, 0.1]], dtype=torch.float64),
        "L_MLO": torch.tensor([[0.4, 2.0, 0.5, 0.2]], dtype=torch.float64),
        "R_CC": torch.tensor([[8.0, 0.1, 0.1, 0.1]], dtype=torch.float64),
        "R_MLO": torch.tensor([[0.2, 0.2, 9.0, 0.2]], dtype=torch.float64),
    }
    logits = {view: _raw_logits_for_evidence(value) for view, value in evidence.items()}

    omitted = fuse_view_logits(logits, ("L_CC", "L_MLO"))

    # Missing sources are vacuous: their zero evidence is an identity element.
    zero = torch.zeros_like(evidence["L_CC"])
    left_pair = evidence["L_CC"] + evidence["L_MLO"] + (evidence["L_CC"] * evidence["L_MLO"] / 4)
    right_vacuous_pair = zero + zero + zero * zero / 4
    expected = left_pair + right_vacuous_pair + left_pair * right_vacuous_pair / 4
    torch.testing.assert_close(omitted.fused_alpha, expected + 1, rtol=1e-12, atol=1e-12)


def test_all_15_nonempty_named_subsets_equal_their_boolean_masks() -> None:
    logits = {
        view: torch.full((3, 4), -0.4 + index * 0.2, dtype=torch.float64)
        + torch.arange(4, dtype=torch.float64) / 10
        for index, view in enumerate(reversed(CANONICAL_VIEWS))
    }
    seen: set[tuple[str, ...]] = set()

    for size in range(1, 5):
        for subset in combinations(CANONICAL_VIEWS, size):
            mask = torch.tensor([view in subset for view in CANONICAL_VIEWS], dtype=torch.bool)
            by_names = fuse_view_logits(logits, reversed(subset))
            by_mask = fuse_view_logits(logits, mask)
            assert by_names.observed_views == subset
            assert by_mask.observed_views == subset
            assert by_names.scores.shape == (3, 4)
            assert torch.isfinite(by_names.scores).all()
            torch.testing.assert_close(by_names.scores, by_mask.scores)
            seen.add(by_names.observed_views)

    assert len(seen) == 15


def test_unobserved_logits_receive_no_gradient() -> None:
    logits = {
        view: torch.randn(2, 4, dtype=torch.float64, requires_grad=True) for view in CANONICAL_VIEWS
    }

    fuse_view_logits(logits, ("L_CC", "R_CC")).scores.sum().backward()

    assert logits["L_CC"].grad is not None
    assert logits["R_CC"].grad is not None
    assert logits["L_MLO"].grad is None
    assert logits["R_MLO"].grad is None


@pytest.mark.parametrize(
    ("observed", "match"),
    [
        ((), "at least one"),
        (("L_CC", "L_CC"), "duplicate"),
        (("L_CC", "not_a_view"), "unknown"),
        ((True, False), "length 4"),
    ],
)
def test_invalid_observed_view_selections_fail_clearly(observed: object, match: str) -> None:
    logits = {"L_CC": torch.zeros(2, 4)}
    error_type = NoObservedViewsError if not observed else ValueError

    with pytest.raises(error_type, match=match):
        fuse_view_logits(logits, observed)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("logits", "observed", "match"),
    [
        ({"L_CC": torch.zeros(2, 3)}, ("L_CC",), "four classes"),
        ({"L_CC": torch.zeros(2, 4), "L_MLO": torch.zeros(3, 4)}, ("L_CC", "L_MLO"), "batch"),
        (
            {"L_CC": torch.zeros(2, 4), "R_CC": torch.zeros(2, 4, dtype=torch.float64)},
            ("L_CC", "R_CC"),
            "dtype",
        ),
        ({"L_CC": torch.tensor([[0.0, float("nan"), 0.0, 0.0]])}, ("L_CC",), "finite"),
        ({"unknown": torch.zeros(2, 4)}, ("L_CC",), "unknown"),
        ({"L_MLO": torch.zeros(2, 4)}, ("L_CC",), "missing"),
    ],
)
def test_malformed_logit_mappings_fail_clearly(
    logits: dict[str, Tensor], observed: tuple[str, ...], match: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        fuse_view_logits(logits, observed)
