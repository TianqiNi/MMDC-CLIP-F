from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from mmdc_clip_f.confidence.model import MVACNConfig, MVACNHead
from mmdc_clip_f.research.view_risk.baselines import (
    BASELINE_DEFINITIONS,
    CONTROL_SELECTION_BUDGET,
    DSLogisticErrorControl,
    MaskedMVACNAdapter,
    MonotoneLogisticProbabilityAdapter,
    MVACNObjective,
    ProbabilityStatus,
    SameInputDensityControl,
    SameInputMLP,
    ScoreOrientation,
    TemperatureScaler,
    ViLUFailureAdapter,
    absolute_omission_sensitivity,
    build_ds_features,
    candidate_parameter_count,
    compute_density_control_loss,
    compute_mvacn_objective,
    compute_same_input_error_loss,
    compute_vilu_failure_loss,
    flatten_same_input_features,
    gather_frozen_prediction_probability,
    p3a_ablation_definitions,
    scalar_baseline_scores,
)
from mmdc_clip_f.research.view_risk.cache import CachedTargets
from mmdc_clip_f.research.view_risk.fusion import (
    DDSM_FUSION_PAIRS,
    RSNA_FUSION_PAIRS,
    fuse_view_logits,
)
from mmdc_clip_f.research.view_risk.head import (
    RelationAwareConfidenceHead,
    RelationAwareHeadConfig,
    count_trainable_parameters,
)
from mmdc_clip_f.research.view_risk.head_inputs import PreparedHeadInputs, RawHeadInputs
from mmdc_clip_f.research.view_risk.inputs import CANONICAL_VIEWS
from mmdc_clip_f.research.view_risk.roles import (
    PatientMappingDeclaration,
    PrivateExamRecord,
    Role,
    RoleManifest,
    ViewReference,
)


def _manifest(role: Role, count: int) -> RoleManifest:
    records = []
    for row in range(count):
        records.append(
            PrivateExamRecord(
                dataset_namespace="synthetic-baseline",
                exam_key=f"exam-{role.value}-{row}",
                patient_key=f"group-{role.value}-{row}",
                density=row % 4,
                source_manifest="fixture",
                views={
                    view: ViewReference(
                        image_id=f"image-{role.value}-{row}-{column}",
                        path=f"synthetic/{role.value}/{row}/{column}.bin",
                    )
                    for column, view in enumerate(CANONICAL_VIEWS)
                },
                role=role,
            )
        )
    return RoleManifest(
        dataset_namespace="synthetic-baseline",
        source_hashes={"fixture": "a" * 64},
        patient_mapping=PatientMappingDeclaration(
            "synthetic-baseline", "b" * 64, "verified synthetic grouping", "test fixture"
        ),
        records=records,
    )


def _exam_keys(manifest: RoleManifest) -> tuple[str, ...]:
    return tuple(record.exam_key for record in manifest.records)


def _raw(hidden_width: int = 768, batch: int = 2) -> RawHeadInputs:
    generator = torch.Generator().manual_seed(41)
    observed = torch.ones(batch, 4, dtype=torch.bool)
    removal = observed.clone()
    scores = torch.tensor([[3.0, 1.0, 0.0, -1.0], [0.2, 1.7, 0.4, -0.3]])[:batch]
    prediction = scores.argmax(1)
    return RawHeadInputs(
        pooled_tokens=torch.randn(batch, 4, hidden_width, generator=generator),
        current_scores=scores,
        current_probabilities=scores.softmax(1),
        fused_evidence=torch.rand(batch, 4, generator=generator),
        fused_vacuity=torch.rand(batch, 1, generator=generator),
        view_evidence=torch.rand(batch, 4, 4, generator=generator),
        view_vacuity=torch.rand(batch, 4, 1, generator=generator),
        omission_score_differences=torch.rand(batch, 4, 4, generator=generator),
        observed_mask=observed,
        removal_valid_mask=removal,
        current_prediction=prediction,
        omission_prediction=prediction[:, None].expand(-1, 4).clone(),
        prediction_agreement=torch.ones_like(observed),
        no_removal=torch.zeros(batch, 1, dtype=torch.bool),
        backbone="vit_b_32" if hidden_width == 768 else "vit_l_14_336",
        token_policy="exclude_cls" if hidden_width == 768 else "include_cls",
        fusion_pairs=RSNA_FUSION_PAIRS,
    )


def _cached(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    valid_removal_mask: torch.Tensor | None = None,
) -> CachedTargets:
    probabilities = scores.softmax(1)
    prediction = scores.argmax(1)
    error = (prediction != labels).long()
    batch = scores.shape[0]
    return CachedTargets(
        labels=labels,
        observed_prediction=prediction,
        observed_error=error,
        omission_predictions=torch.full((batch, 4), -1, dtype=torch.long),
        omission_effects=torch.full((batch, 4), -1, dtype=torch.long),
        omission_labels=torch.full((batch, 4), -1, dtype=torch.long),
        valid_removal_mask=(
            torch.zeros(4, dtype=torch.bool) if valid_removal_mask is None else valid_removal_mask
        ),
        scores=scores,
        probabilities=probabilities,
        tcp=probabilities.gather(1, labels[:, None]).squeeze(1),
    )


def test_scalar_signs_status_and_singleton_omission_behavior() -> None:
    scores = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    values = scalar_baseline_scores(scores)
    probabilities = scores.softmax(1)

    assert values["msp"].orientation is ScoreOrientation.HIGHER_CONFIDENCE
    assert values["msp"].probability_status is ProbabilityStatus.CLASSIFIER_CLASS_PROBABILITY
    torch.testing.assert_close(values["msp"].values, probabilities.max(1).values)
    torch.testing.assert_close(
        values["margin"].values,
        probabilities.topk(2, dim=1).values[:, 0] - probabilities.topk(2, dim=1).values[:, 1],
    )
    torch.testing.assert_close(
        values["negative_entropy"].values,
        (probabilities * probabilities.log()).sum(1),
    )
    assert values["energy"].orientation is ScoreOrientation.HIGHER_ERROR
    assert values["energy"].probability_status is ProbabilityStatus.NOT_A_PROBABILITY
    torch.testing.assert_close(values["energy"].values, -torch.logsumexp(scores, dim=1))

    singleton = absolute_omission_sensitivity(
        torch.tensor([[[99.0, -99.0, 4.0, 1.0]] * 4]),
        torch.zeros(1, 4, dtype=torch.bool),
    )
    assert singleton.orientation is ScoreOrientation.HIGHER_ERROR
    assert singleton.values.item() == 0.0
    assert singleton.no_removal.item() is True


def test_temperature_is_tune_only_positive_and_preserves_classifier_argmax() -> None:
    scores = torch.tensor([[4.0, 1.0, 0.0, -2.0], [0.0, 3.0, 1.0, -1.0], [2.0, 0.0, 1.0, -3.0]])
    labels = torch.tensor([0, 1, 2])
    scaler = TemperatureScaler().fit(
        scores,
        labels,
        manifest=_manifest(Role.TUNE, 3),
        exam_keys=_exam_keys(_manifest(Role.TUNE, 3)),
    )
    scaled = scaler.transform(scores)

    assert scaler.temperature.item() > 0
    assert torch.equal(scaled.argmax(1), scores.argmax(1))
    assert scalar_baseline_scores(scores, temperature=scaler)[
        "temperature_scaled_msp"
    ].values.shape == (3,)
    with pytest.raises(PermissionError):
        TemperatureScaler().fit(
            scores,
            labels,
            manifest=_manifest(Role.CONFIDENCE_FIT, 3),
            exam_keys=_exam_keys(_manifest(Role.CONFIDENCE_FIT, 3)),
        )


def test_temperature_refuses_labels_that_disagree_with_manifest_densities() -> None:
    manifest = _manifest(Role.TUNE, 2)
    scaler = TemperatureScaler()

    with pytest.raises(ValueError, match="manifest.*densit"):
        scaler.fit(
            torch.tensor([[9.0, 1.0, 1.0, 1.0], [1.0, 9.0, 1.0, 1.0]]),
            torch.tensor([3, 3]),
            manifest=manifest,
            exam_keys=_exam_keys(manifest),
        )
    with pytest.raises(RuntimeError, match="not been tune-fitted"):
        _ = scaler.temperature


def test_arbitrary_sigmoid_is_not_a_calibrated_scalar_probability() -> None:
    raw_error_score = torch.tensor([-2.0, -0.2, 0.5, 2.0])
    errors = torch.tensor([0, 0, 1, 1])
    adapter = MonotoneLogisticProbabilityAdapter(ScoreOrientation.HIGHER_ERROR)
    with pytest.raises(RuntimeError, match="tune-fitted"):
        adapter.predict_error_probability(raw_error_score)
    adapter.fit(
        raw_error_score,
        errors,
        manifest=_manifest(Role.TUNE, 4),
        exam_keys=_exam_keys(_manifest(Role.TUNE, 4)),
    )
    probability = adapter.predict_error_probability(raw_error_score)
    assert adapter.probability_status is ProbabilityStatus.TUNE_CALIBRATED_ERROR_PROBABILITY
    assert torch.all(probability[1:] >= probability[:-1])
    with pytest.raises(PermissionError):
        MonotoneLogisticProbabilityAdapter(ScoreOrientation.HIGHER_ERROR).fit(
            raw_error_score,
            errors,
            manifest=_manifest(Role.PILOT, 4),
            exam_keys=_exam_keys(_manifest(Role.PILOT, 4)),
        )


@pytest.mark.parametrize(
    ("tree", "observed", "valid_slots"),
    [
        (RSNA_FUSION_PAIRS, ("L_CC", "L_MLO", "R_CC"), (True, False, True)),
        (DDSM_FUSION_PAIRS, ("L_CC", "L_MLO", "R_MLO"), (False, True, True)),
    ],
)
def test_ds_conflicts_map_to_stable_tree_slots_after_pruning(tree, observed, valid_slots) -> None:
    logits = {
        view: torch.tensor([[1.2 + column, -0.5, 0.2, 0.7]])
        for column, view in enumerate(CANONICAL_VIEWS)
        if view in observed
    }
    active = fuse_view_logits(logits, observed, fusion_pairs=tree).stats.conflicts
    features = build_ds_features(logits, observed, fusion_pairs=tree)

    assert tuple(features.conflict_valid_mask[0].tolist()) == valid_slots
    assert features.feature_order == (
        "fused_vacuity",
        "view_vacuity_L_CC",
        "view_vacuity_L_MLO",
        "view_vacuity_R_CC",
        "view_vacuity_R_MLO",
        "conflict_branch_1",
        "conflict_branch_2",
        "conflict_root",
    )
    expected_slots = [index for index, valid in enumerate(valid_slots) if valid]
    for value, slot in zip(active, expected_slots):
        torch.testing.assert_close(features.values[:, 5 + slot], value)
    assert torch.all(features.values[~features.valid_mask] == 0)


def test_ds_fit_and_regularization_roles_come_from_manifests() -> None:
    train_logits = {
        view: torch.tensor([[0.2 + row + column, 0.1, -0.2, 0.3] for row in range(4)])
        for column, view in enumerate(CANONICAL_VIEWS)
    }
    tune_logits = {view: value[[0, 3]] for view, value in train_logits.items()}
    train_x = build_ds_features(train_logits, CANONICAL_VIEWS)
    train_y = torch.tensor([0, 0, 1, 1])
    tune_x = build_ds_features(tune_logits, CANONICAL_VIEWS)
    tune_y = torch.tensor([0, 1])
    control = DSLogisticErrorControl.fit_with_tune_selection(
        train_x,
        train_y,
        tune_x,
        tune_y,
        confidence_manifest=_manifest(Role.CONFIDENCE_FIT, 4),
        confidence_exam_keys=_exam_keys(_manifest(Role.CONFIDENCE_FIT, 4)),
        tune_manifest=_manifest(Role.TUNE, 2),
        tune_exam_keys=_exam_keys(_manifest(Role.TUNE, 2)),
        regularizations=(0.0, 0.1),
    )
    assert control.selected_regularization in (0.0, 0.1)
    assert torch.isfinite(control.predict_error_probability(tune_x)).all()
    with pytest.raises(PermissionError):
        DSLogisticErrorControl.fit_with_tune_selection(
            train_x,
            train_y,
            tune_x,
            tune_y,
            confidence_manifest=_manifest(Role.TUNE, 4),
            confidence_exam_keys=_exam_keys(_manifest(Role.TUNE, 4)),
            tune_manifest=_manifest(Role.TUNE, 2),
            tune_exam_keys=_exam_keys(_manifest(Role.TUNE, 2)),
            regularizations=(0.1,),
        )


@pytest.mark.parametrize(
    "config",
    [
        MVACNConfig(hidden_dim=768, dropout=0.2, drop_cls_tokens=True),
        MVACNConfig(hidden_dim=1024, dropout=0.3, drop_cls_tokens=False),
    ],
)
@pytest.mark.parametrize(
    "legacy_view_order",
    [
        ("L_CC", "L_MLO", "R_CC", "R_MLO"),
        ("L_CC", "R_CC", "L_MLO", "R_MLO"),
    ],
    ids=("rsna", "ddsm"),
)
def test_masked_mvacn_is_exact_for_each_legacy_order_and_finite_on_subsets(
    config, legacy_view_order
) -> None:
    torch.manual_seed(12)
    legacy = MVACNHead(config).eval()
    adapter = MaskedMVACNAdapter.from_legacy(legacy, legacy_view_order=legacy_view_order).eval()
    views = {view: torch.randn(2, 3, config.hidden_dim) for view in CANONICAL_VIEWS}
    prediction = torch.tensor([1, 3])
    current_scores = torch.tensor([[0.0, 4.0, 1.0, -1.0], [0.0, 1.0, 2.0, 4.0]])

    expected = legacy(*(views[view] for view in legacy_view_order))
    actual = adapter(
        views,
        CANONICAL_VIEWS,
        classifier_prediction=prediction,
        current_scores=current_scores,
    )
    torch.testing.assert_close(actual.confidence_logit, expected, rtol=0, atol=0)
    assert torch.equal(actual.classifier_prediction, prediction)
    for observed in (("L_CC",), ("L_MLO", "R_CC"), ("L_CC", "R_CC", "R_MLO")):
        subset = adapter(
            views,
            observed,
            classifier_prediction=prediction,
            current_scores=current_scores,
        )
        assert torch.isfinite(subset.confidence_logit).all()


def test_masked_mvacn_filters_configured_order_for_subsets_and_refuses_invalid_order() -> None:
    ddsm_order = ("L_CC", "R_CC", "L_MLO", "R_MLO")
    config = MVACNConfig(hidden_dim=8, dropout=0.3, drop_cls_tokens=False)
    adapter = MaskedMVACNAdapter(config, legacy_view_order=ddsm_order).eval()
    views = {
        view: torch.full((1, 1, 8), float(column + 1))
        for column, view in enumerate(CANONICAL_VIEWS)
    }
    captured = []
    handle = adapter.head.attention.register_forward_pre_hook(
        lambda _module, arguments: captured.append(arguments[0].detach().clone())
    )
    try:
        adapter(
            views,
            ("L_CC", "L_MLO", "R_CC"),
            classifier_prediction=torch.tensor([0]),
            current_scores=torch.tensor([[4.0, 1.0, 0.0, -1.0]]),
        )
    finally:
        handle.remove()

    expected_tokens = torch.cat((views["L_CC"], views["R_CC"], views["L_MLO"]), dim=1)
    torch.testing.assert_close(captured[0][:, 1:], expected_tokens)
    for invalid in (
        ("L_CC", "L_MLO", "R_CC"),
        ("L_CC", "L_MLO", "R_CC", "R_CC"),
        ("L_CC", "L_MLO", "R_CC", "UNKNOWN"),
    ):
        with pytest.raises(ValueError, match="legacy_view_order"):
            MaskedMVACNAdapter(config, legacy_view_order=invalid)


def test_mvacn_objectives_use_current_cached_targets_and_frozen_prediction() -> None:
    scores = torch.tensor([[3.0, 1.0, 0.0, -1.0], [2.0, 1.0, 0.0, -1.0]])
    targets = _cached(
        scores,
        torch.tensor([0, 1]),
        valid_removal_mask=torch.ones(4, dtype=torch.bool),
    )
    logit = torch.tensor([0.4, -0.7], requires_grad=True)
    output = MaskedMVACNAdapter.output_from_logit(
        logit,
        targets.observed_prediction,
        current_scores=scores,
        observed_mask=torch.ones(2, 4, dtype=torch.bool),
    )

    tcp = compute_mvacn_objective(output, targets, MVACNObjective.TCP_MSE)
    correctness = compute_mvacn_objective(output, targets, MVACNObjective.CORRECTNESS_BCE)
    torch.testing.assert_close(tcp, F.mse_loss(logit.sigmoid(), targets.tcp))
    torch.testing.assert_close(
        correctness,
        F.binary_cross_entropy_with_logits(logit, 1 - targets.observed_error.float()),
    )
    stale = replace(targets, tcp=torch.ones_like(targets.tcp))
    with pytest.raises(ValueError, match="current realized"):
        compute_mvacn_objective(output, stale, MVACNObjective.TCP_MSE)


def test_learned_objectives_reject_stale_same_argmax_scores_and_current_mask() -> None:
    clean_scores = torch.tensor([[9.0, 1.0, 1.0, 1.0], [1.0, 9.0, 1.0, 1.0]])
    current_scores = torch.tensor([[2.0, 1.9, 1.8, 1.7], [1.9, 2.0, 1.8, 1.7]])
    labels = torch.tensor([0, 1])
    observed_mask = torch.tensor([[True, True, False, False]]).expand(2, -1)
    valid_removal_mask = observed_mask[0]
    current_targets = _cached(
        current_scores,
        labels,
        valid_removal_mask=valid_removal_mask,
    )
    stale_clean_targets = _cached(
        clean_scores,
        labels,
        valid_removal_mask=valid_removal_mask,
    )
    assert torch.equal(clean_scores.argmax(1), current_scores.argmax(1))
    assert not torch.allclose(stale_clean_targets.tcp, current_targets.tcp)

    logit = torch.tensor([0.4, -0.7], requires_grad=True)
    mvacn = MaskedMVACNAdapter.output_from_logit(
        logit,
        current_targets.observed_prediction,
        current_scores=current_scores,
        observed_mask=observed_mask,
    )
    text = torch.randn(4, 6)
    vilu = ViLUFailureAdapter(visual_width=5, text_embeddings=text, hidden_dim=7)(
        torch.randn(2, 4, 5),
        observed_mask,
        classifier_prediction=current_targets.observed_prediction,
        current_scores=current_scores,
    )
    raw = replace(
        _raw(),
        current_scores=current_scores,
        current_probabilities=current_scores.softmax(1),
        current_prediction=current_targets.observed_prediction,
        observed_mask=observed_mask,
        removal_valid_mask=observed_mask,
    )
    same_input = SameInputMLP("vit_b_32")(raw)
    density = SameInputDensityControl("vit_b_32")(raw)
    objectives = (
        lambda target: compute_mvacn_objective(mvacn, target, MVACNObjective.TCP_MSE),
        lambda target: compute_vilu_failure_loss(vilu, target),
        lambda target: compute_same_input_error_loss(same_input, target),
        lambda target: compute_density_control_loss(density, target),
    )

    for objective in objectives:
        assert torch.isfinite(objective(current_targets))
        with pytest.raises(ValueError, match="current.*scores"):
            objective(stale_clean_targets)
        with pytest.raises(ValueError, match="current.*mask"):
            objective(
                replace(
                    current_targets,
                    valid_removal_mask=torch.tensor([True, False, False, False]),
                )
            )


def test_vilu_attention_is_learned_image_conditioned_and_selects_frozen_prediction_text() -> None:
    torch.manual_seed(23)
    text = torch.randn(4, 6)
    model = ViLUFailureAdapter(visual_width=5, text_embeddings=text, hidden_dim=7).eval()
    visual = torch.randn(2, 4, 5)
    observed = torch.tensor([[True, True, False, False]]).expand(2, -1)
    prediction = torch.tensor([3, 1])
    current_scores = torch.tensor([[0.0, 0.0, 0.0, 4.0], [0.0, 4.0, 0.0, 0.0]])
    output = model(
        visual,
        observed,
        classifier_prediction=prediction,
        current_scores=current_scores,
    )

    torch.testing.assert_close(output.predicted_text_embedding, text[prediction])
    assert not torch.allclose(output.attention_weights[0], output.attention_weights[1])
    output.error_logit.sum().backward()
    assert model.query.weight.grad is not None and model.query.weight.grad.abs().sum() > 0
    assert model.key.weight.grad is not None and model.key.weight.grad.abs().sum() > 0

    changed = visual.clone()
    changed[~observed] = 1e6
    torch.testing.assert_close(
        model(
            changed,
            observed,
            classifier_prediction=prediction,
            current_scores=current_scores,
        ).error_logit,
        output.error_logit,
    )
    targets = _cached(
        current_scores,
        torch.tensor([0, 1]),
        valid_removal_mask=observed[0],
    )
    loss = compute_vilu_failure_loss(output, targets)
    torch.testing.assert_close(
        loss, F.binary_cross_entropy_with_logits(output.error_logit, targets.observed_error.float())
    )


def test_same_input_flattening_is_canonical_and_excludes_prediction_and_labels() -> None:
    prepared = PreparedHeadInputs(
        view_features=torch.arange(16, dtype=torch.float32).reshape(1, 4, 4),
        omission_features=100 + torch.arange(16, dtype=torch.float32).reshape(1, 4, 4),
        global_features=200 + torch.arange(13, dtype=torch.float32).reshape(1, 13),
        observed_mask=torch.tensor([[True, False, True, False]]),
        removal_valid_mask=torch.tensor([[True, False, True, False]]),
        current_prediction=torch.tensor([3]),
        omission_prediction=torch.tensor([[0, -1, 2, -1]]),
        prediction_agreement=torch.tensor([[False, False, True, False]]),
        no_removal=torch.tensor([[False]]),
    )
    flattened = flatten_same_input_features(prepared)
    expected = torch.cat(
        (
            prepared.view_features.flatten(1),
            prepared.omission_features.flatten(1),
            prepared.global_features,
            prepared.observed_mask.float(),
            prepared.removal_valid_mask.float(),
            prepared.no_removal.float(),
        ),
        dim=1,
    )
    torch.testing.assert_close(flattened, expected)
    assert not torch.isin(torch.tensor([3.0]), flattened[:, -9:]).all()


@pytest.mark.parametrize("backbone", ["vit_b_32", "vit_l_14_336"])
def test_same_input_controls_own_independent_projections_and_match_candidate_capacity(
    backbone,
) -> None:
    binary = SameInputMLP(backbone)
    density = SameInputDensityControl(backbone)
    target = candidate_parameter_count(backbone)

    assert binary.preparation is not density.preparation
    assert binary.scaler is not density.scaler
    assert (
        binary.preparation.hidden_projection.weight.data_ptr()
        != density.preparation.hidden_projection.weight.data_ptr()
    )
    assert abs(count_trainable_parameters(binary) - target) / target <= 0.10
    assert abs(count_trainable_parameters(density) - target) / target <= 0.10

    raw = _raw(768 if backbone == "vit_b_32" else 1024)
    targets = _cached(
        raw.current_scores,
        torch.tensor([0, 2]),
        valid_removal_mask=torch.ones(4, dtype=torch.bool),
    )
    output = binary(raw)
    compute_same_input_error_loss(output, targets).backward()
    assert all(parameter.grad is not None for parameter in binary.parameters())

    density_output = density(raw)
    compute_density_control_loss(density_output, targets).backward()
    assert all(parameter.grad is not None for parameter in density.parameters())
    torch.testing.assert_close(
        density_output.confidence,
        density_output.class_logits.softmax(1)
        .gather(1, raw.current_prediction[:, None])
        .squeeze(1),
    )


def test_four_class_confidence_is_gathered_at_frozen_classifier_prediction() -> None:
    logits = torch.tensor([[4.0, 2.0, 1.0, 0.0], [4.0, 2.0, 1.0, 0.0]])
    frozen_prediction = torch.tensor([2, 1])
    confidence = gather_frozen_prediction_probability(logits, frozen_prediction)

    expected = logits.softmax(1)[torch.arange(2), frozen_prediction]
    torch.testing.assert_close(confidence, expected)
    assert confidence[0] != logits.softmax(1).max(1).values[0]


def test_exposure_metadata_and_p3a_ablations_are_explicit_and_finite() -> None:
    assert BASELINE_DEFINITIONS["original_mvacn_tcp"].input_exposure == "clean_four_view"
    assert (
        BASELINE_DEFINITIONS["matched_mvacn_tcp"].input_exposure == "matched_augmentation_and_masks"
    )
    assert BASELINE_DEFINITIONS["matched_mvacn_tcp"].requires_current_realized_targets
    assert BASELINE_DEFINITIONS["vilu"].loss_weighting == "unweighted_bce_adaptation"
    assert not BASELINE_DEFINITIONS["vilu"].claims_exact_source_loss
    assert all(
        isinstance(value, int) and 0 < value < 100 for value in CONTROL_SELECTION_BUDGET.values()
    )

    base = RelationAwareHeadConfig(backbone="vit_b_32")
    ablations = p3a_ablation_definitions(base)
    assert ablations["no_effect_supervision"].head_config is base
    assert ablations["no_effect_supervision"].auxiliary_weight == 0
    assert ablations["no_intervention_features"].head_config.use_omission_features is False
    assert ablations["no_view_relations"].head_config.use_relation_biases is False
    assert ablations["magnitude_effects"].head_config.auxiliary_task == "magnitude"
    assert ablations["corruption_auxiliary"].head_config.auxiliary_task == "corruption"
    assert ablations["hidden_only"].head_config.input_mode == "hidden_only"
    assert ablations["evidence_only"].head_config.input_mode == "evidence_only"
    assert count_trainable_parameters(RelationAwareConfidenceHead(base)) == 433_436
