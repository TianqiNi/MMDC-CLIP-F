from __future__ import annotations

import json

import pytest

from mmdc_clip_f.research.view_risk.evaluation import (
    CheckpointTuneResult,
    ClassifierTuneCheckpoint,
    ModelArtifactSelection,
    PilotEvaluationSummary,
    TuneCellResult,
    evaluate_pilot_with_role_access,
    evaluate_prediction_panel_with_role_access,
    evaluate_tune_cells_with_role_access,
    freeze_pilot_plan,
    load_pilot_plan,
    select_candidate_checkpoint,
    select_clean_reference,
    select_fresh_classifier_on_tune,
)
from mmdc_clip_f.research.view_risk.metrics import EvaluationPrediction
from mmdc_clip_f.cli import build_parser, main
from mmdc_clip_f.research.view_risk.roles import (
    PatientMappingDeclaration,
    PrivateExamRecord,
    Role,
    RoleManifest,
    ViewReference,
)
from mmdc_clip_f.research.view_risk.training import (
    MANDATORY_METHODS,
    ClassifierProvenance,
    OptimizerConfig,
    ResearchRunConfig,
    RoleBoundBatch,
    TrainingBinding,
    audit_software_fixture,
    fit_fresh_classifier_with_role_access,
    initialize_public_classifier,
)
from torch import nn
import torch


def _manifest(role: Role) -> RoleManifest:
    return RoleManifest(
        dataset_namespace="synthetic-evaluation",
        source_hashes={"fixture": "a" * 64},
        patient_mapping=PatientMappingDeclaration(
            "synthetic-evaluation", "b" * 64, "verified synthetic grouping", "test fixture"
        ),
        records=[
            PrivateExamRecord(
                dataset_namespace="synthetic-evaluation",
                exam_key="exam-safe",
                patient_key="patient-safe",
                density=0,
                source_manifest="fixture",
                views={
                    view: ViewReference(image_id=f"image-{view}", path=f"fixture/{view}.png")
                    for view in ("L_CC", "L_MLO", "R_CC", "R_MLO")
                },
                role=role,
            )
        ],
    )


def _cells(value: float) -> tuple[TuneCellResult, ...]:
    return tuple(
        TuneCellResult(family, severity, view, value)
        for family in ("gaussian_noise", "gaussian_blur")
        for severity in ("mild", "moderate")
        for view in ("L_CC", "L_MLO", "R_CC", "R_MLO")
    )


def _checkpoint(
    epoch: int,
    clean: float,
    stress: float,
    artifact: str,
    *,
    method: str = "candidate",
    tune_manifest_sha256: str = "e" * 64,
    role: str = Role.TUNE.value,
) -> CheckpointTuneResult:
    return CheckpointTuneResult(
        method=method,
        seed=42,
        classifier_checkpoint_sha256="f" * 64,
        tune_manifest_sha256=tune_manifest_sha256,
        epoch=epoch,
        clean_aurc=clean,
        stress_cells=_cells(stress),
        artifact_sha256=artifact,
        role=role,
    )


def test_held_out_or_strong_tune_cell_is_rejected_before_reader() -> None:
    called = False

    def reader(_records, _cells):
        nonlocal called
        called = True
        return ()

    for cell in (
        TuneCellResult("contrast", "mild", "L_CC", 0.2),
        TuneCellResult("gaussian_noise", "strong", "L_CC", 0.2),
    ):
        with pytest.raises(ValueError, match="tune"):
            evaluate_tune_cells_with_role_access(_manifest(Role.TUNE), (cell,), reader)
    assert called is False


def test_reference_guardrail_ties_and_no_eligible_selection() -> None:
    reference = select_clean_reference(
        (
            _checkpoint(2, 0.20, 9.0, "2" * 64, method="correctness_mvacn"),
            _checkpoint(1, 0.20, 8.0, "1" * 64, method="correctness_mvacn"),
            _checkpoint(3, 0.21, 0.0, "3" * 64, method="correctness_mvacn"),
        )
    )
    assert reference.epoch == 1

    selected = select_candidate_checkpoint(
        (
            _checkpoint(1, 0.204, 0.30, "4" * 64),
            _checkpoint(2, 0.203, 0.30, "5" * 64),
            _checkpoint(3, 0.202, 0.31, "6" * 64),
        ),
        clean_reference=reference,
    )
    assert selected is not None and selected.epoch == 2

    assert (
        select_candidate_checkpoint(
            (_checkpoint(1, 0.206, 0.01, "7" * 64),),
            clean_reference=reference,
        )
        is None
    )
    with pytest.raises(ValueError, match="correctness-MVACN"):
        select_candidate_checkpoint(
            (_checkpoint(1, 0.2, 0.2, "8" * 64),),
            clean_reference=_checkpoint(1, 0.2, 0.2, "9" * 64),
        )
    with pytest.raises(ValueError, match="tune-role"):
        _checkpoint(1, 0.2, 0.2, "a" * 64, role=Role.PILOT.value)
    with pytest.raises(ValueError, match="tune manifest"):
        select_candidate_checkpoint(
            (
                _checkpoint(
                    1,
                    0.2,
                    0.2,
                    "b" * 64,
                    tune_manifest_sha256="d" * 64,
                ),
            ),
            clean_reference=reference,
        )


def test_fresh_classifier_selection_reads_only_tune_and_remains_software_only() -> None:
    fit_manifest = _manifest(Role.CLASSIFIER_FIT)
    tune_manifest = _manifest(Role.TUNE)
    model, initialization = initialize_public_classifier(
        "vit_b_32", lambda _public: nn.Linear(1, 1)
    )
    binding = TrainingBinding(
        protocol_sha256="1" * 64,
        config_sha256="2" * 64,
        search_table_sha256="3" * 64,
        manifest_sha256=fit_manifest.manifest_sha256,
        classifier_checkpoint_sha256=initialization.checkpoint_sha256,
        method="fresh_classifier",
        seed=42,
    )

    def batches(records, _epoch):
        yield RoleBoundBatch(
            tuple(record.exam_key for record in records),
            (torch.ones(len(records), 1), torch.zeros(len(records), 1)),
        )

    fitted = fit_fresh_classifier_with_role_access(
        manifest=fit_manifest,
        model=model,
        classifier=initialization,
        readiness=audit_software_fixture(fit_manifest),
        optimizer_config=OptimizerConfig(batch_size=6),
        epochs=1,
        binding=binding,
        batch_loader=batches,
        loss_fn=lambda current, payload: (current(payload[0]) - payload[1]).square().mean(),
    )
    selected = select_fresh_classifier_on_tune(
        tune_manifest,
        initialization=initialization,
        classifier_fit_result=fitted,
        readiness=audit_software_fixture(fit_manifest),
        checkpoint_evaluator=lambda _records: (
            ClassifierTuneCheckpoint(2, "b" * 64, 0.4),
            ClassifierTuneCheckpoint(1, "a" * 64, 0.4),
        ),
    )
    assert selected.checkpoint_sha256 == "a" * 64
    assert selected.workflow_complete is True
    assert selected.pilot_eligible is False
    with pytest.raises(PermissionError, match="readiness"):
        selected.require_pilot_eligible()


def _selections(config: ResearchRunConfig) -> tuple[ModelArtifactSelection, ...]:
    return tuple(
        ModelArtifactSelection(method, seed, f"{index + 10:064x}")
        for index, (method, seed) in enumerate(
            (method, seed) for method in config.methods for seed in config.seeds
        )
    )


def _fresh_selected() -> ClassifierProvenance:
    return ClassifierProvenance.fresh_selected(
        ClassifierProvenance.public_pretrained("vit_b_32", "a" * 64),
        checkpoint_sha256="c" * 64,
        classifier_fit_manifest_sha256=_manifest(Role.CLASSIFIER_FIT).manifest_sha256,
        classifier_fit_update_count=20,
        tune_selection_sha256="e" * 64,
        tune_manifest_sha256="f" * 64,
        readiness=audit_software_fixture(_manifest(Role.CLASSIFIER_FIT)),
    )


def _manifest_hashes(pilot_sha256: str | None = None) -> dict[str, str]:
    values = {role.value: f"{index + 1:064x}" for index, role in enumerate(Role)}
    values[Role.CLASSIFIER_FIT.value] = _manifest(Role.CLASSIFIER_FIT).manifest_sha256
    values[Role.TUNE.value] = "f" * 64
    if pilot_sha256 is not None:
        values[Role.PILOT.value] = pilot_sha256
    return values


def test_pilot_plan_is_immutable_complete_and_binds_all_inputs(tmp_path) -> None:
    config = ResearchRunConfig.default("RSNA", "vit_b_32")
    classifier = _fresh_selected()
    path = tmp_path / "pilot-plan.json"
    plan = freeze_pilot_plan(
        path,
        config=config,
        manifest_sha256_by_role=_manifest_hashes(),
        classifier=classifier,
        selections=_selections(config),
    )

    assert plan.stress_seed == 4242
    assert plan.methods == MANDATORY_METHODS
    assert load_pilot_plan(path, expected_config_sha256=config.sha256).sha256 == plan.sha256
    with pytest.raises(FileExistsError):
        freeze_pilot_plan(
            path,
            config=config,
            manifest_sha256_by_role=_manifest_hashes(),
            classifier=classifier,
            selections=_selections(config),
        )

    document = json.loads(path.read_text())
    document["plan"]["stress_seed"] = 7
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        load_pilot_plan(path, expected_config_sha256=config.sha256)


def test_missing_selection_diagnostic_classifier_and_locked_outcomes_are_refused(tmp_path) -> None:
    config = ResearchRunConfig.default("RSNA", "vit_b_32")
    with pytest.raises(ValueError, match="selection"):
        freeze_pilot_plan(
            tmp_path / "missing.json",
            config=config,
            manifest_sha256_by_role=_manifest_hashes(),
            classifier=_fresh_selected(),
            selections=_selections(config)[:-1],
        )
    with pytest.raises(PermissionError, match="diagnostic"):
        freeze_pilot_plan(
            tmp_path / "diagnostic.json",
            config=config,
            manifest_sha256_by_role=_manifest_hashes(),
            classifier=ClassifierProvenance.diagnostic_original(
                "vit_b_32", "c" * 64, reason="known development exposure"
            ),
            selections=_selections(config),
        )

    # Ordinary manifests cannot even be constructed with locked outcomes; the
    # evaluation wrapper also has no release boolean or alternative role path.
    called = False

    def outcome_reader(_records):
        nonlocal called
        called = True
        return PilotEvaluationSummary("candidate", 42, "primary", 0.1, 1, 1)

    pilot_manifest = _manifest(Role.PILOT)
    plan = freeze_pilot_plan(
        tmp_path / "valid.json",
        config=config,
        manifest_sha256_by_role=_manifest_hashes(pilot_manifest.manifest_sha256),
        classifier=_fresh_selected(),
        selections=_selections(config),
    )
    with pytest.raises(PermissionError):
        evaluate_pilot_with_role_access(
            plan,
            manifest=pilot_manifest,
            role=Role.LOCKED_TEST,
            method="candidate",
            seed=42,
            model_sha256=dict(plan.selection_map)["candidate:42"],
            outcome_reader=outcome_reader,
        )
    assert called is False


def test_pilot_evaluation_checks_model_binding_before_outcome_reader(tmp_path) -> None:
    config = ResearchRunConfig.default("RSNA", "vit_b_32")
    manifest = _manifest(Role.PILOT)
    plan = freeze_pilot_plan(
        tmp_path / "pilot.json",
        config=config,
        manifest_sha256_by_role=_manifest_hashes(manifest.manifest_sha256),
        classifier=_fresh_selected(),
        selections=_selections(config),
    )
    called = False

    def outcome_reader(_records):
        nonlocal called
        called = True
        return PilotEvaluationSummary("candidate", 42, "primary", 0.1, 1, 1)

    with pytest.raises(ValueError, match="model binding"):
        evaluate_pilot_with_role_access(
            plan,
            manifest=manifest,
            role=Role.PILOT,
            method="candidate",
            seed=42,
            model_sha256="f" * 64,
            outcome_reader=outcome_reader,
        )
    assert called is False

    result = evaluate_pilot_with_role_access(
        plan,
        manifest=manifest,
        role=Role.PILOT,
        method="candidate",
        seed=42,
        model_sha256=dict(plan.selection_map)["candidate:42"],
        outcome_reader=outcome_reader,
    )
    assert result.mean_aurc == 0.1
    assert called is True

    called = False
    document = json.loads(plan.source_path.read_text())
    document["plan"]["stress_seed"] = 7
    plan.source_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        evaluate_pilot_with_role_access(
            plan,
            manifest=manifest,
            role=Role.PILOT,
            method="candidate",
            seed=42,
            model_sha256=dict(plan.selection_map)["candidate:42"],
            outcome_reader=outcome_reader,
        )
    assert called is False


def test_role_authorized_evaluation_really_uses_p4a_and_cli_has_no_unlock(tmp_path, capsys) -> None:
    config = ResearchRunConfig.default("RSNA", "vit_b_32")
    manifest = _manifest(Role.PILOT)
    plan = freeze_pilot_plan(
        tmp_path / "p4a-plan.json",
        config=config,
        manifest_sha256_by_role=_manifest_hashes(manifest.manifest_sha256),
        classifier=_fresh_selected(),
        selections=_selections(config),
    )
    row = manifest.records[0]
    summary = evaluate_prediction_panel_with_role_access(
        plan,
        manifest=manifest,
        role=Role.PILOT,
        method="candidate",
        seed=42,
        model_sha256=dict(plan.selection_map)["candidate:42"],
        prediction_reader=lambda _records: (
            EvaluationPrediction(
                dataset="RSNA",
                role=Role.PILOT.value,
                cohort=manifest.manifest_sha256,
                method="candidate",
                training_seed=42,
                patient_id=row.patient_key,
                exam_id=row.exam_key,
                panel="clean_four_view",
                target=row.density,
                prediction=row.density,
                confidence=0.8,
                confidence_kind="probability",
            ),
        ),
    )
    assert summary.mean_aurc == 0.0
    assert summary.patient_ready is False

    parser = build_parser()
    help_text = parser.format_help()
    assert "view-risk-evaluate" in help_text
    parsed = parser.parse_args(
        ["view-risk-validate-config", "--config", str(tmp_path / "config.json")]
    )
    assert not hasattr(parsed, "allow_test_selection")
    (tmp_path / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
    assert main(["view-risk-validate-config", "--config", str(tmp_path / "config.json")]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "valid"
    assert output["patient_readiness"] == "not_assessed"
