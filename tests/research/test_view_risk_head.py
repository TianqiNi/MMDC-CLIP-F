from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from mmdc_clip_f.research.view_risk.features import FrozenViewFeatures
from mmdc_clip_f.research.view_risk.fusion import RSNA_FUSION_PAIRS, fuse_view_logits
from mmdc_clip_f.research.view_risk.head import (
    MAGNITUDE_EFFECT_CLASS_ORDER,
    RelationAwareConfidenceHead,
    RelationAwareHeadConfig,
    RelationAwareHeadOutput,
    SIGNED_EFFECT_CLASS_ORDER,
    compute_view_risk_loss,
    count_trainable_parameters,
)
from mmdc_clip_f.research.view_risk.head_inputs import prepare_raw_head_inputs
from mmdc_clip_f.research.view_risk.inputs import CANONICAL_VIEWS
from mmdc_clip_f.research.view_risk.targets import InterventionTargets, build_intervention_targets


def _raw(
    observed=("L_CC", "L_MLO", "R_CC"), *, hidden_width: int = 768, batch_size: int = 2
):
    logits = {
        view: torch.tensor(
            [[2.0 + i, -1.0, 0.2 * i, 0.1], [-1.0, 2.0 - i, 0.3, 0.2]],
            dtype=torch.float32,
        )[:batch_size]
        for i, view in enumerate(CANONICAL_VIEWS)
        if view in observed
    }
    fused = fuse_view_logits(logits, observed)
    generator = torch.Generator().manual_seed(91)
    hidden = {
        view: torch.randn(batch_size, 4, hidden_width, generator=generator)
        for view in observed
    }
    projected = {view: torch.randn(batch_size, 6, generator=generator) for view in observed}
    features = FrozenViewFeatures(
        logits_by_view=logits,
        hidden_by_view=hidden,
        projected_by_view=projected,
        normalized_text_embeddings=torch.eye(6)[:4],
        scores=fused.scores,
        probabilities=torch.softmax(fused.scores, dim=1),
        observed_views=tuple(observed),
        fusion_pairs=RSNA_FUSION_PAIRS,
    )
    backbone = "vit_b_32" if hidden_width == 768 else "vit_l_14_336"
    return prepare_raw_head_inputs(features, backbone=backbone), logits


def _targets(logits, observed, labels=(0, 1)) -> InterventionTargets:
    return build_intervention_targets(logits, torch.tensor(labels), observed)


def test_default_architecture_and_public_output_shapes() -> None:
    raw, _ = _raw()
    model = RelationAwareConfidenceHead(RelationAwareHeadConfig(backbone="vit_b_32"))
    model.eval()

    output = model(raw)

    assert SIGNED_EFFECT_CLASS_ORDER == (-1, 0, 1)
    assert MAGNITUDE_EFFECT_CLASS_ORDER == (0, 1)
    assert len(model.relation_layers) == 2
    for layer in model.relation_layers:
        assert layer.num_heads == 4
        assert layer.head_dim == 32
        assert layer.feed_forward[0].in_features == 128
        assert layer.feed_forward[0].out_features == 256
        assert layer.feed_forward[-2].in_features == 256
        assert layer.feed_forward[-2].out_features == 128
        assert isinstance(layer.attention_normalization, nn.LayerNorm)
        assert isinstance(layer.feed_forward_normalization, nn.LayerNorm)
        assert layer.dropout.p == pytest.approx(0.2)
    assert output.error_logit.shape == (2,)
    assert output.confidence.shape == (2,)
    assert output.effect_representations.shape == (2, 4, 128)
    assert output.raw_auxiliary_logits.shape == (2, 4, 3)
    assert output.reported_auxiliary_probabilities.shape == (2, 4, 3)
    assert torch.equal(output.classifier_prediction, raw.current_prediction)
    torch.testing.assert_close(output.confidence, 1 - torch.sigmoid(output.error_logit))


def test_relation_biases_follow_names_and_ablation_retains_name_embeddings() -> None:
    config = RelationAwareHeadConfig(backbone="vit_b_32")
    model = RelationAwareConfidenceHead(config)
    relation = model.relation_layers[0].relation_codes

    assert relation[0, 1].tolist() == [True, False, False]  # same breast
    assert relation[0, 2].tolist() == [False, True, False]  # same projection
    assert relation[0, 0].tolist() == [True, True, True]  # explicit self too
    without = RelationAwareConfidenceHead(replace(config, use_relation_biases=False))
    assert hasattr(without.preparation, "view_name_embeddings")
    assert not any("relation_bias" in name for name, _ in without.named_parameters())
    assert count_trainable_parameters(model) - count_trainable_parameters(without) == 24


def test_absent_slots_cannot_change_valid_outputs() -> None:
    torch.manual_seed(3)
    raw, _ = _raw(("L_CC", "R_MLO"))
    model = RelationAwareConfidenceHead(RelationAwareHeadConfig(backbone="vit_b_32")).eval()
    missing = ~raw.observed_mask
    altered = replace(
        raw,
        pooled_tokens=raw.pooled_tokens.masked_fill(missing.unsqueeze(-1), 9e5),
        view_evidence=raw.view_evidence.masked_fill(missing.unsqueeze(-1), -8e5),
        view_vacuity=raw.view_vacuity.masked_fill(missing.unsqueeze(-1), 7e5),
        omission_score_differences=raw.omission_score_differences.masked_fill(
            missing.unsqueeze(-1), -6e5
        ),
    )

    expected = model(raw)
    actual = model(altered)

    torch.testing.assert_close(actual.error_logit, expected.error_logit)
    valid = raw.removal_valid_mask
    torch.testing.assert_close(
        actual.raw_auxiliary_logits[valid], expected.raw_auxiliary_logits[valid]
    )


def test_identical_predictions_force_exact_signed_reporting_but_not_raw_logits() -> None:
    torch.manual_seed(7)
    raw, _ = _raw(("L_CC", "L_MLO"))
    raw = replace(raw, prediction_agreement=raw.removal_valid_mask.clone())
    model = RelationAwareConfidenceHead(RelationAwareHeadConfig(backbone="vit_b_32")).eval()

    output = model(raw)
    valid = output.auxiliary_valid_mask

    assert torch.equal(
        output.reported_auxiliary_probabilities[valid],
        torch.tensor([0.0, 1.0, 0.0]).expand(valid.sum(), -1),
    )
    assert torch.all(output.reported_auxiliary_probabilities[~valid] == 0)
    assert torch.isfinite(output.raw_auxiliary_logits[valid]).all()
    assert output.raw_auxiliary_logits[valid].requires_grad


def test_magnitude_constraint_and_corruption_reporting_semantics() -> None:
    raw, _ = _raw(("L_CC", "L_MLO"))
    raw = replace(raw, prediction_agreement=raw.removal_valid_mask.clone())
    magnitude = RelationAwareConfidenceHead(
        RelationAwareHeadConfig(backbone="vit_b_32", auxiliary_task="magnitude")
    ).eval()(raw)
    corruption = RelationAwareConfidenceHead(
        RelationAwareHeadConfig(backbone="vit_b_32", auxiliary_task="corruption")
    ).eval()(raw)

    assert torch.equal(
        magnitude.reported_auxiliary_probabilities[magnitude.auxiliary_valid_mask],
        torch.tensor([1.0, 0.0]).expand(magnitude.auxiliary_valid_mask.sum(), -1),
    )
    # Corruption is defined for observed views (including a singleton) and is never
    # overwritten from classifier agreement.
    assert torch.equal(corruption.auxiliary_valid_mask, raw.observed_mask)
    torch.testing.assert_close(
        corruption.reported_auxiliary_probabilities[corruption.auxiliary_valid_mask],
        corruption.raw_auxiliary_logits[corruption.auxiliary_valid_mask].softmax(dim=-1),
    )


def test_singleton_corruption_supervises_its_only_observed_view() -> None:
    torch.manual_seed(17)
    raw, _ = _raw(("R_CC",))
    model = RelationAwareConfidenceHead(
        RelationAwareHeadConfig(backbone="vit_b_32", auxiliary_task="corruption")
    ).eval()
    output = model(raw)
    output.raw_auxiliary_logits.retain_grad()
    output.effect_representations.retain_grad()
    corruption_targets = torch.tensor([[-1, -1, 1, -1], [-1, -1, 0, -1]], dtype=torch.long)

    assert not raw.removal_valid_mask.any()
    assert torch.equal(output.auxiliary_valid_mask, raw.observed_mask)

    loss = compute_view_risk_loss(
        output,
        torch.tensor([0, 1]),
        corruption_targets=corruption_targets,
    )
    expected = F.cross_entropy(output.raw_auxiliary_logits[:, 2], corruption_targets[:, 2])
    torch.testing.assert_close(loss.auxiliary_ce, expected)
    assert loss.auxiliary_ce.item() > 0

    loss.auxiliary_ce.backward()
    assert output.raw_auxiliary_logits.grad is not None
    assert output.raw_auxiliary_logits.grad[:, 2].abs().sum() > 0
    assert output.raw_auxiliary_logits.grad[:, [0, 1, 3]].count_nonzero() == 0
    assert output.effect_representations.grad is not None
    assert output.effect_representations.grad[:, 2].abs().sum() > 0
    assert output.effect_representations.grad[:, [0, 1, 3]].count_nonzero() == 0


def test_parent_normalized_loss_uses_raw_logits_for_identical_predictions() -> None:
    valid = torch.tensor([[True, True, False, False], [True, True, False, False]])
    raw_logits = torch.tensor(
        [
            [[2.0, 0.0, -1.0], [0.0, 3.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[-1.0, 1.0, 2.0], [2.0, 1.0, -2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        requires_grad=True,
    )
    reported = raw_logits.softmax(-1).detach().clone()
    reported[valid] = torch.tensor([0.0, 1.0, 0.0])
    output = RelationAwareHeadOutput(
        error_logit=torch.tensor([0.4, -0.7], requires_grad=True),
        confidence=torch.tensor([0.2, 0.8]),
        effect_representations=torch.zeros(2, 4, 128),
        raw_auxiliary_logits=raw_logits,
        reported_auxiliary_probabilities=reported,
        auxiliary_valid_mask=valid,
        auxiliary_task="signed",
    )
    targets = InterventionTargets(
        observed_views=("L_CC", "L_MLO"),
        observed_prediction=torch.tensor([0, 1]),
        observed_error=torch.tensor([0, 1]),
        omission_predictions=torch.tensor([[0, 2, -1, -1], [2, 1, -1, -1]]),
        omission_effects=torch.tensor([[0, -1, -1, -1], [1, 0, -1, -1]]),
        omission_labels=torch.tensor([[1, 0, -1, -1], [2, 1, -1, -1]]),
        valid_removal_mask=torch.tensor([True, True, False, False]),
        fusion_pairs=RSNA_FUSION_PAIRS,
    )

    result = compute_view_risk_loss(output, targets, auxiliary_weight=0.6)
    expected_bce = F.binary_cross_entropy_with_logits(
        output.error_logit, targets.observed_error.float()
    )
    parent_0 = (F.cross_entropy(raw_logits[0, 0:1], torch.tensor([1])) + F.cross_entropy(raw_logits[0, 1:2], torch.tensor([0]))) / 2
    parent_1 = (F.cross_entropy(raw_logits[1, 0:1], torch.tensor([2])) + F.cross_entropy(raw_logits[1, 1:2], torch.tensor([1]))) / 2
    expected_ce = (parent_0 + parent_1) / 2

    torch.testing.assert_close(result.error_bce, expected_bce)
    torch.testing.assert_close(result.auxiliary_ce, expected_ce)
    torch.testing.assert_close(result.total, expected_bce + 0.6 * expected_ce)
    result.total.backward()
    assert raw_logits.grad is not None and raw_logits.grad.abs().sum() > 0


def test_mixed_masks_average_each_parents_auxiliary_mean_equally() -> None:
    observed = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
            [False, False, True, False],
        ]
    )
    valid = observed & (observed.sum(dim=1, keepdim=True) > 1)
    assert torch.equal(observed.sum(dim=1), torch.tensor([3, 2, 1]))
    assert torch.equal(valid.sum(dim=1), torch.tensor([3, 2, 0]))
    raw_logits = torch.tensor(
        [
            [[3.0, -1.0, 0.0], [0.0, 1.0, 2.0], [-2.0, 0.5, 1.0], [0.0, 0.0, 0.0]],
            [[-1.0, 2.0, 0.0], [1.5, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        requires_grad=True,
    )
    output = RelationAwareHeadOutput(
        error_logit=torch.tensor([0.2, -0.4, 0.7], requires_grad=True),
        confidence=torch.tensor([0.4, 0.6, 0.3]),
        effect_representations=torch.zeros(3, 4, 128),
        raw_auxiliary_logits=raw_logits,
        reported_auxiliary_probabilities=raw_logits.softmax(dim=-1),
        auxiliary_valid_mask=valid,
        auxiliary_task="signed",
    )
    labels = torch.tensor([[0, 2, 1, -1], [1, 0, -1, -1], [-1, -1, -1, -1]])
    targets = InterventionTargets(
        observed_views=CANONICAL_VIEWS,
        observed_prediction=torch.tensor([0, 1, 2]),
        observed_error=torch.tensor([0, 1, 0]),
        omission_predictions=torch.full((3, 4), -1),
        omission_effects=torch.where(valid, labels - 1, labels),
        omission_labels=labels,
        valid_removal_mask=valid,
        fusion_pairs=RSNA_FUSION_PAIRS,
    )

    result = compute_view_risk_loss(output, targets)
    parent_0 = torch.stack(
        [
            F.cross_entropy(raw_logits[0, slot : slot + 1], labels[0, slot : slot + 1])
            for slot in range(3)
        ]
    ).mean()
    parent_1 = torch.stack(
        [
            F.cross_entropy(raw_logits[1, slot : slot + 1], labels[1, slot : slot + 1])
            for slot in range(2)
        ]
    ).mean()
    parent_2 = raw_logits.new_zeros(())
    expected_per_parent = torch.stack((parent_0, parent_1, parent_2))

    torch.testing.assert_close(result.per_parent_auxiliary_ce, expected_per_parent)
    torch.testing.assert_close(result.auxiliary_ce, expected_per_parent.mean())


def test_singleton_auxiliary_loss_is_zero_and_global_loss_finite() -> None:
    raw, logits = _raw(("R_CC",))
    model = RelationAwareConfidenceHead(RelationAwareHeadConfig(backbone="vit_b_32"))
    output = model(raw)
    targets = _targets(logits, ("R_CC",))

    loss = compute_view_risk_loss(output, targets)

    assert loss.auxiliary_ce.item() == 0.0
    assert torch.isfinite(loss.total)
    assert not output.auxiliary_valid_mask.any()


def test_lambda_zero_preserves_effect_to_risk_gradient_path() -> None:
    torch.manual_seed(11)
    raw, logits = _raw(("L_CC", "L_MLO", "R_CC"))
    model = RelationAwareConfidenceHead(RelationAwareHeadConfig(backbone="vit_b_32"))
    model.eval()
    output = model(raw)
    output.effect_representations.retain_grad()
    targets = _targets(logits, ("L_CC", "L_MLO", "R_CC"))

    loss = compute_view_risk_loss(output, targets, auxiliary_weight=0.0)
    loss.total.backward()

    assert output.effect_representations.grad is not None
    assert output.effect_representations.grad[output.auxiliary_valid_mask].abs().sum() > 0
    effect_grad = sum(
        parameter.grad.abs().sum()
        for parameter in model.effect_representation.parameters()
        if parameter.grad is not None
    )
    assert effect_grad > 0


def test_ablation_inputs_are_isolated_one_factor_at_a_time() -> None:
    torch.manual_seed(13)
    raw, _ = _raw(("L_CC", "L_MLO", "R_CC"))
    missing_or_invalid = ~raw.removal_valid_mask
    variants = {
        "hidden_only": (
            RelationAwareHeadConfig(backbone="vit_b_32", input_mode="hidden_only"),
            replace(
                raw,
                current_scores=raw.current_scores + 20,
                current_probabilities=torch.softmax(raw.current_scores + 20, dim=1),
                fused_evidence=raw.fused_evidence + 20,
                fused_vacuity=raw.fused_vacuity + 20,
                view_evidence=raw.view_evidence + 20,
                view_vacuity=raw.view_vacuity + 20,
                omission_score_differences=raw.omission_score_differences + 20,
            ),
        ),
        "evidence_only": (
            RelationAwareHeadConfig(backbone="vit_b_32", input_mode="evidence_only"),
            replace(raw, pooled_tokens=raw.pooled_tokens + 20),
        ),
        "no_omission": (
            RelationAwareHeadConfig(backbone="vit_b_32", use_omission_features=False),
            replace(
                raw,
                omission_score_differences=raw.omission_score_differences.masked_fill(
                    ~missing_or_invalid.unsqueeze(-1), 20
                ),
            ),
        ),
    }
    for config, changed in variants.values():
        model = RelationAwareConfidenceHead(config).eval()
        first, second = model(raw), model(changed)
        torch.testing.assert_close(first.error_logit, second.error_logit)
        torch.testing.assert_close(first.raw_auxiliary_logits, second.raw_auxiliary_logits)


def test_corruption_targets_enter_only_loss_and_are_validated() -> None:
    raw, _ = _raw(("L_CC", "L_MLO"))
    model = RelationAwareConfidenceHead(
        RelationAwareHeadConfig(backbone="vit_b_32", auxiliary_task="corruption")
    ).eval()
    output = model(raw)
    observed_error = torch.tensor([0, 1])
    targets_a = torch.tensor([[0, 1, -1, -1], [1, 0, -1, -1]])
    targets_b = torch.tensor([[1, 0, -1, -1], [0, 1, -1, -1]])

    loss_a = compute_view_risk_loss(
        output, observed_error, corruption_targets=targets_a
    )
    loss_b = compute_view_risk_loss(
        output, observed_error, corruption_targets=targets_b
    )

    assert not torch.equal(loss_a.per_parent_auxiliary_ce, loss_b.per_parent_auxiliary_ce)
    with pytest.raises(TypeError, match="unexpected keyword"):
        model(raw, corruption_targets=targets_a)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="corruption_targets"):
        compute_view_risk_loss(output, observed_error)
    with pytest.raises(ValueError, match="0, 1, or -1"):
        compute_view_risk_loss(
            output, observed_error, corruption_targets=targets_a.masked_fill(raw.observed_mask, 2)
        )


def test_signed_and_magnitude_losses_use_distinct_auxiliary_targets() -> None:
    raw, logits = _raw(("L_CC", "L_MLO"))
    targets = _targets(logits, ("L_CC", "L_MLO"))
    signed = RelationAwareConfidenceHead(
        RelationAwareHeadConfig(backbone="vit_b_32", auxiliary_task="signed")
    ).eval()(raw)
    magnitude = RelationAwareConfidenceHead(
        RelationAwareHeadConfig(backbone="vit_b_32", auxiliary_task="magnitude")
    ).eval()(raw)

    signed_loss = compute_view_risk_loss(signed, targets)
    magnitude_loss = compute_view_risk_loss(magnitude, targets)
    expected_magnitude = targets.omission_effects.abs()
    valid = magnitude.auxiliary_valid_mask
    manual = F.cross_entropy(
        magnitude.raw_auxiliary_logits[valid], expected_magnitude[valid].long()
    )
    torch.testing.assert_close(magnitude_loss.auxiliary_ce, manual)
    assert signed_loss.total.ndim == magnitude_loss.total.ndim == 0


def test_default_and_ablation_parameter_counts_are_auditable() -> None:
    configs = [
        RelationAwareHeadConfig(backbone="vit_b_32"),
        RelationAwareHeadConfig(backbone="vit_l_14_336"),
        RelationAwareHeadConfig(backbone="vit_b_32", use_omission_features=False),
        RelationAwareHeadConfig(backbone="vit_b_32", use_relation_biases=False),
        RelationAwareHeadConfig(backbone="vit_b_32", auxiliary_task="magnitude"),
        RelationAwareHeadConfig(backbone="vit_b_32", auxiliary_task="corruption"),
        RelationAwareHeadConfig(backbone="vit_b_32", input_mode="hidden_only"),
        RelationAwareHeadConfig(backbone="vit_b_32", input_mode="evidence_only"),
    ]
    counts = [count_trainable_parameters(RelationAwareConfidenceHead(config)) for config in configs]
    assert all(count > 0 for count in counts)
    assert counts[1] > counts[0]
    assert counts[2] < counts[0]
    assert counts[3] == counts[0] - 24
    assert counts[4] == counts[5] < counts[0]
    assert counts[6] < counts[0]
    assert counts[7] < counts[0]
