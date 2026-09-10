from __future__ import annotations

from itertools import combinations

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mmdc_clip_f.model import MultiViewCLIPClassifier
import mmdc_clip_f.research.view_risk as view_risk


FusionPairs = tuple[tuple[str, str], tuple[str, str]]
RSNA_PAIRS: FusionPairs = (("L_CC", "L_MLO"), ("R_CC", "R_MLO"))
DDSM_PAIRS: FusionPairs = (("L_CC", "R_CC"), ("L_MLO", "R_MLO"))


def _legacy_classifier(fusion_pairs: FusionPairs) -> MultiViewCLIPClassifier:
    return MultiViewCLIPClassifier(
        nn.Identity(),
        input_order=view_risk.CANONICAL_VIEWS,
        fusion_pairs=fusion_pairs,
    )


def _independent_conflict(alpha1: Tensor, alpha2: Tensor) -> Tensor:
    belief1 = (alpha1 - 1) / alpha1.sum(dim=1, keepdim=True)
    belief2 = (alpha2 - 1) / alpha2.sum(dim=1, keepdim=True)
    products = torch.bmm(belief1.unsqueeze(2), belief2.unsqueeze(1))
    agreement = torch.diagonal(products, dim1=-2, dim2=-1).sum(-1)
    return products.sum(dim=(1, 2)) - agreement


def _ordinary_logits(dtype: torch.dtype) -> dict[str, Tensor]:
    return {
        view: torch.tensor([[-1.7, 0.2, 2.3, -0.4], [1.1, -2.2, 0.7, 3.0]], dtype=dtype)
        + index * torch.tensor(0.31, dtype=dtype)
        for index, view in enumerate(view_risk.CANONICAL_VIEWS)
    }


def _tree_sensitive_logits() -> dict[str, Tensor]:
    values = torch.tensor([170365.78125, 1.1348471641540527, 43.50401306152344, 18.158279418945312])
    stacked = torch.full((1, 4, 4), -100.0)
    stacked[0, :, 0] = values
    stacked[0, :, 1] = values[[0, 1, 3, 2]]
    return {view: stacked[:, index, :] for index, view in enumerate(view_risk.CANONICAL_VIEWS)}


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("fusion_pairs", [RSNA_PAIRS, DDSM_PAIRS])
def test_each_configured_tree_matches_legacy_scores_predictions_and_pair_conflicts(
    dtype: torch.dtype,
    fusion_pairs: FusionPairs,
) -> None:
    logits = _ordinary_logits(dtype)
    legacy = _legacy_classifier(fusion_pairs)
    alpha = {view: F.softplus(value) + 1 for view, value in logits.items()}
    first, _ = legacy._combine_two(alpha[fusion_pairs[0][0]], alpha[fusion_pairs[0][1]])
    second, _ = legacy._combine_two(alpha[fusion_pairs[1][0]], alpha[fusion_pairs[1][1]])
    expected_alpha, _ = legacy._combine_two(first, second)
    expected_scores = legacy.fuse_logits(logits)
    expected_conflicts = (
        _independent_conflict(alpha[fusion_pairs[0][0]], alpha[fusion_pairs[0][1]]),
        _independent_conflict(alpha[fusion_pairs[1][0]], alpha[fusion_pairs[1][1]]),
        _independent_conflict(first, second),
    )

    result = view_risk.fuse_view_logits(
        logits,
        view_risk.CANONICAL_VIEWS,
        fusion_pairs=fusion_pairs,
    )

    tolerance = {
        torch.float64: (1e-12, 1e-12),
        torch.float32: (2e-6, 2e-6),
    }[dtype]
    torch.testing.assert_close(
        result.fused_alpha, expected_alpha, rtol=tolerance[0], atol=tolerance[1]
    )
    torch.testing.assert_close(result.scores, expected_scores, rtol=tolerance[0], atol=tolerance[1])
    assert torch.equal(result.scores.argmax(dim=1), expected_scores.argmax(dim=1))
    assert result.fusion_pairs == fusion_pairs
    assert len(result.stats.conflicts) == 3
    for actual, expected in zip(result.stats.conflicts, expected_conflicts):
        torch.testing.assert_close(actual, expected, rtol=tolerance[0], atol=tolerance[1])


@pytest.mark.parametrize("fusion_pairs", [RSNA_PAIRS, DDSM_PAIRS])
def test_both_trees_support_all_15_named_subsets_and_masks(
    fusion_pairs: FusionPairs,
) -> None:
    logits = _ordinary_logits(torch.float64)
    seen: set[tuple[str, ...]] = set()

    for size in range(1, 5):
        for subset in combinations(view_risk.CANONICAL_VIEWS, size):
            mask = torch.tensor(
                [view in subset for view in view_risk.CANONICAL_VIEWS],
                dtype=torch.bool,
            )
            by_names = view_risk.fuse_view_logits(
                logits, reversed(subset), fusion_pairs=fusion_pairs
            )
            by_mask = view_risk.fuse_view_logits(logits, mask, fusion_pairs=fusion_pairs)
            assert by_names.observed_views == subset
            assert by_names.fusion_pairs == fusion_pairs
            torch.testing.assert_close(by_names.scores, by_mask.scores)
            seen.add(by_names.observed_views)

    assert len(seen) == 15


def test_default_tree_is_rsna_and_tree_choice_controls_known_float32_tie() -> None:
    logits = _tree_sensitive_logits()

    default = view_risk.fuse_view_logits(logits, view_risk.CANONICAL_VIEWS)
    rsna = view_risk.fuse_view_logits(logits, view_risk.CANONICAL_VIEWS, fusion_pairs=RSNA_PAIRS)
    ddsm = view_risk.fuse_view_logits(logits, view_risk.CANONICAL_VIEWS, fusion_pairs=DDSM_PAIRS)

    assert view_risk.RSNA_FUSION_PAIRS == RSNA_PAIRS
    assert view_risk.DDSM_FUSION_PAIRS == DDSM_PAIRS
    assert default.fusion_pairs == RSNA_PAIRS
    torch.testing.assert_close(default.scores, rsna.scores)
    assert rsna.scores.argmax(dim=1).tolist() == [0]
    assert ddsm.scores.argmax(dim=1).tolist() == [1]


def test_target_builder_propagates_tree_to_parent_and_every_omission() -> None:
    logits = _tree_sensitive_logits()
    labels = torch.tensor([0])

    rsna = view_risk.build_intervention_targets(
        logits,
        labels,
        view_risk.CANONICAL_VIEWS,
        fusion_pairs=RSNA_PAIRS,
    )
    ddsm = view_risk.build_intervention_targets(
        logits,
        labels,
        view_risk.CANONICAL_VIEWS,
        fusion_pairs=DDSM_PAIRS,
    )

    assert rsna.fusion_pairs == RSNA_PAIRS
    assert ddsm.fusion_pairs == DDSM_PAIRS
    assert rsna.observed_prediction.tolist() == [0]
    assert rsna.omission_predictions.tolist() == [[0, 0, 1, 0]]
    assert rsna.omission_effects.tolist() == [[0, 0, -1, 0]]
    assert ddsm.observed_prediction.tolist() == [1]
    assert ddsm.omission_predictions.tolist() == [[1, 1, 1, 0]]
    assert ddsm.omission_effects.tolist() == [[0, 0, 0, 1]]
    assert ddsm.omission_labels.tolist() == [[1, 1, 1, 2]]


@pytest.mark.parametrize(
    "fusion_pairs",
    [
        (("L_CC", "L_MLO"),),
        (("L_CC", "L_MLO"), ("R_CC", "L_CC")),
        (("L_CC", "L_MLO"), ("R_CC", "unknown")),
    ],
)
def test_fusion_and_targets_reject_invalid_tree(fusion_pairs: object) -> None:
    logits = _ordinary_logits(torch.float32)

    with pytest.raises((TypeError, ValueError), match="fusion_pairs.*partition"):
        view_risk.fuse_view_logits(
            logits,
            view_risk.CANONICAL_VIEWS,
            fusion_pairs=fusion_pairs,
        )
    with pytest.raises((TypeError, ValueError), match="fusion_pairs.*partition"):
        view_risk.build_intervention_targets(
            logits,
            torch.tensor([0, 1]),
            view_risk.CANONICAL_VIEWS,
            fusion_pairs=fusion_pairs,
        )


@pytest.mark.parametrize("fusion_pairs", [RSNA_PAIRS, DDSM_PAIRS])
def test_extreme_finite_logits_fail_before_nonfinite_results_or_argmax(
    fusion_pairs: FusionPairs,
) -> None:
    logits = {
        view: torch.full((1, 4), 1e38, dtype=torch.float32) for view in view_risk.CANONICAL_VIEWS
    }
    assert all(torch.isfinite(value).all() for value in logits.values())
    assert issubclass(view_risk.InvalidDSFusionError, ValueError)

    with pytest.raises(view_risk.InvalidDSFusionError, match="nonfinite"):
        view_risk.fuse_view_logits(
            logits,
            view_risk.CANONICAL_VIEWS,
            fusion_pairs=fusion_pairs,
        )
    with pytest.raises(view_risk.InvalidDSFusionError, match="nonfinite"):
        view_risk.build_intervention_targets(
            logits,
            torch.tensor([3]),
            view_risk.CANONICAL_VIEWS,
            fusion_pairs=fusion_pairs,
        )
