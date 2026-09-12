from __future__ import annotations

from dataclasses import replace
from itertools import combinations

import pytest
import torch

from mmdc_clip_f.research.view_risk.features import FrozenViewFeatures
from mmdc_clip_f.research.view_risk.fusion import DDSM_FUSION_PAIRS, RSNA_FUSION_PAIRS
from mmdc_clip_f.research.view_risk.head_inputs import (
    IdentityFeatureScaler,
    ViewRiskInputPreparation,
    prepare_raw_head_inputs,
)
from mmdc_clip_f.research.view_risk.inputs import CANONICAL_VIEWS


def _features(
    observed: tuple[str, ...],
    *,
    hidden_width: int,
    fusion_pairs=RSNA_FUSION_PAIRS,
    batch_size: int = 2,
) -> FrozenViewFeatures:
    logits = {
        view: (
            torch.arange(batch_size * 4, dtype=torch.float32).reshape(batch_size, 4) / 9
            + 0.17 * CANONICAL_VIEWS.index(view)
        )
        for view in reversed(observed)
    }
    from mmdc_clip_f.research.view_risk.fusion import fuse_view_logits

    fused = fuse_view_logits(logits, observed, fusion_pairs=fusion_pairs)
    hidden = {
        view: (
            torch.arange(batch_size * 3 * hidden_width, dtype=torch.float32)
            .reshape(batch_size, 3, hidden_width)
            .remainder(29)
            / 29
            + CANONICAL_VIEWS.index(view)
        )
        for view in reversed(observed)
    }
    projected = {
        view: torch.full((batch_size, 6), float(CANONICAL_VIEWS.index(view) + 1))
        for view in reversed(observed)
    }
    text = torch.eye(6)[:4]
    return FrozenViewFeatures(
        logits_by_view=logits,
        hidden_by_view=hidden,
        projected_by_view=projected,
        normalized_text_embeddings=text,
        scores=fused.scores,
        probabilities=torch.softmax(fused.scores, dim=1),
        observed_views=observed,
        fusion_pairs=fusion_pairs,
    )


@pytest.mark.parametrize(
    ("backbone", "width", "fusion_pairs"),
    [
        ("vit_b_32", 768, RSNA_FUSION_PAIRS),
        ("vit_b_32", 768, DDSM_FUSION_PAIRS),
        ("vit_l_14_336", 1024, RSNA_FUSION_PAIRS),
        ("vit_l_14_336", 1024, DDSM_FUSION_PAIRS),
    ],
)
def test_all_15_masks_prepare_canonical_label_free_parent_and_children(
    backbone: str, width: int, fusion_pairs
) -> None:
    seen: set[tuple[bool, ...]] = set()
    for size in range(1, 5):
        for observed in combinations(CANONICAL_VIEWS, size):
            source = _features(observed, hidden_width=width, fusion_pairs=fusion_pairs)
            before = tuple(tensor.clone() for tensor in source.all_tensors())

            raw = prepare_raw_head_inputs(source, backbone=backbone)

            expected_mask = torch.tensor(
                [[view in observed for view in CANONICAL_VIEWS]] * source.batch_size
            )
            assert torch.equal(raw.observed_mask, expected_mask)
            assert raw.pooled_tokens.shape == (source.batch_size, 4, width)
            assert raw.view_evidence.shape == (source.batch_size, 4, 4)
            assert raw.view_vacuity.shape == (source.batch_size, 4, 1)
            assert raw.omission_score_differences.shape == (source.batch_size, 4, 4)
            assert raw.removal_valid_mask.shape == (source.batch_size, 4)
            assert raw.prediction_agreement.shape == (source.batch_size, 4)
            assert raw.current_scores.shape == (source.batch_size, 4)
            assert raw.current_probabilities.shape == (source.batch_size, 4)
            assert torch.equal(raw.current_prediction, source.prediction)
            assert raw.no_removal.shape == (source.batch_size, 1)
            assert torch.equal(raw.no_removal[:, 0], ~raw.removal_valid_mask.any(dim=1))
            assert all(not tensor.requires_grad for tensor in raw.all_tensors())
            assert all(torch.equal(actual, expected) for actual, expected in zip(source.all_tensors(), before))

            for column, view in enumerate(CANONICAL_VIEWS):
                if size > 1 and view in observed:
                    child = tuple(item for item in observed if item != view)
                    from mmdc_clip_f.research.view_risk.fusion import fuse_view_logits

                    child_result = fuse_view_logits(
                        source.logits_by_view, child, fusion_pairs=fusion_pairs
                    )
                    torch.testing.assert_close(
                        raw.omission_score_differences[:, column],
                        source.scores - child_result.scores,
                    )
                    assert torch.equal(
                        raw.omission_prediction[:, column], child_result.scores.argmax(dim=1)
                    )
                    assert torch.equal(
                        raw.prediction_agreement[:, column],
                        source.prediction == child_result.scores.argmax(dim=1),
                    )
                else:
                    assert not raw.removal_valid_mask[:, column].any()
            seen.add(tuple(raw.observed_mask[0].tolist()))
    assert len(seen) == 15


def test_backbone_token_policy_excludes_cls_only_for_vit_b() -> None:
    base = _features(("L_CC",), hidden_width=768, batch_size=1)
    large = _features(("L_CC",), hidden_width=1024, batch_size=1)

    base_raw = prepare_raw_head_inputs(base, backbone="vit_b_32")
    large_raw = prepare_raw_head_inputs(large, backbone="vit_l_14_336")

    torch.testing.assert_close(
        base_raw.pooled_tokens[:, 0], base.hidden_by_view["L_CC"][:, 1:].mean(dim=1)
    )
    torch.testing.assert_close(
        large_raw.pooled_tokens[:, 0], large.hidden_by_view["L_CC"].mean(dim=1)
    )
    assert base_raw.token_policy == "exclude_cls"
    assert large_raw.token_policy == "include_cls"


def test_named_mapping_order_never_changes_canonical_slots() -> None:
    observed = ("L_CC", "R_CC", "R_MLO")
    source = _features(observed, hidden_width=768, batch_size=1)
    raw = prepare_raw_head_inputs(source, backbone="vit_b_32")

    assert raw.view_order == CANONICAL_VIEWS
    for column, view in enumerate(CANONICAL_VIEWS):
        if view in observed:
            expected = source.hidden_by_view[view][:, 1:].mean(dim=1)
            torch.testing.assert_close(raw.pooled_tokens[:, column], expected)


def test_preparation_refuses_stale_classifier_scores_and_wrong_backbone_width() -> None:
    source = _features(("L_CC", "R_MLO"), hidden_width=768)
    stale = FrozenViewFeatures(
        logits_by_view=source.logits_by_view,
        hidden_by_view=source.hidden_by_view,
        projected_by_view=source.projected_by_view,
        normalized_text_embeddings=source.normalized_text_embeddings,
        scores=source.scores.roll(1, dims=1),
        probabilities=torch.softmax(source.scores.roll(1, dims=1), dim=1),
        observed_views=source.observed_views,
        fusion_pairs=source.fusion_pairs,
    )
    with pytest.raises(ValueError, match="accepted fusion"):
        prepare_raw_head_inputs(stale, backbone="vit_b_32")
    with pytest.raises(ValueError, match="hidden width"):
        prepare_raw_head_inputs(source, backbone="vit_l_14_336")


def test_raw_preparation_refuses_target_and_corruption_inputs() -> None:
    source = _features(("L_CC", "L_MLO"), hidden_width=768)
    with pytest.raises(TypeError, match="unexpected keyword"):
        prepare_raw_head_inputs(  # type: ignore[call-arg]
            source, backbone="vit_b_32", labels=torch.tensor([0, 1])
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        prepare_raw_head_inputs(  # type: ignore[call-arg]
            source, backbone="vit_b_32", corruption_family="noise"
        )


def test_identity_scaler_has_no_fitting_or_state() -> None:
    scaler = IdentityFeatureScaler()
    value = torch.randn(2, 5)
    assert scaler(value) is value
    assert scaler.state_dict() == {}
    with pytest.raises(RuntimeError, match="does not fit"):
        scaler.fit(value)


@pytest.mark.parametrize("mode", ("combined", "hidden_only", "evidence_only"))
def test_each_preparation_module_owns_its_learned_projection(mode: str) -> None:
    first = ViewRiskInputPreparation(768, input_mode=mode)
    second = ViewRiskInputPreparation(768, input_mode=mode)
    first_parameters = dict(first.named_parameters())
    second_parameters = dict(second.named_parameters())

    assert first_parameters
    assert set(first_parameters) == set(second_parameters)
    assert all(first_parameters[name] is not second_parameters[name] for name in first_parameters)
    if mode != "evidence_only":
        assert first.hidden_projection.in_features == 768
        assert first.hidden_projection.out_features == 128


def test_preparation_masks_slot_values_even_if_absent_raw_storage_is_perturbed() -> None:
    torch.manual_seed(4)
    raw = prepare_raw_head_inputs(
        _features(("L_CC", "R_MLO"), hidden_width=768), backbone="vit_b_32"
    )
    module = ViewRiskInputPreparation(768, input_mode="combined").eval()
    missing = ~raw.observed_mask
    perturbed = replace(
        raw,
        pooled_tokens=raw.pooled_tokens.masked_fill(missing.unsqueeze(-1), 1e6),
        view_evidence=raw.view_evidence.masked_fill(missing.unsqueeze(-1), -1e6),
        view_vacuity=raw.view_vacuity.masked_fill(missing.unsqueeze(-1), 1e6),
        omission_score_differences=raw.omission_score_differences.masked_fill(
            missing.unsqueeze(-1), -1e6
        ),
    )

    prepared = module(raw)
    altered = module(perturbed)

    torch.testing.assert_close(prepared.view_features, altered.view_features)
    torch.testing.assert_close(prepared.omission_features, altered.omission_features)
    torch.testing.assert_close(prepared.global_features, altered.global_features)
