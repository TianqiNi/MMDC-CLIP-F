from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

from mmdc_clip_f.backbones import PROMPTS
from mmdc_clip_f.cli import build_parser
from mmdc_clip_f.provenance import sha256_file
from mmdc_clip_f.research.view_risk.cli import run_view_risk_command
from mmdc_clip_f.research.view_risk.artifacts import load_control_artifact
from mmdc_clip_f.research.view_risk.cache import (
    build_exam_cache_with_role_access,
    save_cache_bundle,
)
from mmdc_clip_f.research.view_risk.features import load_verified_frozen_encoder
from mmdc_clip_f.research.view_risk.fusion import RSNA_FUSION_PAIRS
from mmdc_clip_f.research.view_risk.evaluation import (
    ClassifierTuneCheckpoint,
    freeze_pilot_plan,
    load_model_artifact_selection,
    select_fresh_classifier_on_tune,
)
from mmdc_clip_f.research.view_risk.inference import (
    load_learned_confidence_model,
    score_confidence_cache_bundle,
)
from mmdc_clip_f.research.view_risk.production import (
    fit_classifier_images_with_role_access,
    load_confidence_fit_artifact,
    load_public_initialization_artifact,
    load_selected_classifier_artifact,
    load_tensor_checkpoint_state,
    reload_verified_public_classifier,
    save_classifier_fit_artifact,
    save_confidence_fit_artifact,
    save_public_initialization_artifact,
    save_selected_classifier_artifact,
    save_tensor_checkpoint,
)
from mmdc_clip_f.research.view_risk.perturbations import (
    PerturbationSpec,
    realize_parent,
    realize_training_parent,
)
from mmdc_clip_f.research.view_risk.roles import (
    Operation,
    PatientMappingDeclaration,
    PrivateExamRecord,
    Role,
    RoleManifest,
    ViewReference,
    save_private_manifest,
)
from mmdc_clip_f.research.view_risk.training import (
    ClassifierProvenance,
    OptimizerConfig,
    ResearchRunConfig,
    RoleBoundBatch,
    TrainingBinding,
    TrainingResult,
    audit_software_fixture,
    build_method_training_schedule,
    fit_fresh_classifier_with_role_access,
    load_verified_readiness_audit,
    load_pinned_public_clip_classifier,
    pinned_public_clip_configuration,
    save_readiness_audit,
    state_dict_sha256,
)


def _software_fixture(tmp_path, *, namespace="RSNA-SMBC"):
    source = tmp_path / "source.csv"
    mapping = tmp_path / "mapping.csv"
    denylist_file = tmp_path / "locked.txt"
    source.write_bytes(b"authorized research source\n")
    mapping.write_bytes(b"verified patient mapping\n")
    denylist_file.write_bytes(b"locked identity-only evidence\n")
    manifests = []
    for index, role in enumerate(
        (Role.CLASSIFIER_FIT, Role.CONFIDENCE_FIT, Role.TUNE, Role.PILOT)
    ):
        views = {}
        for view_index, view in enumerate(("L_CC", "L_MLO", "R_CC", "R_MLO")):
            image = tmp_path / f"role-{index}-{view}.png"
            Image.new(
                "RGB", (8, 8), color=(index * 30, view_index * 40, index + view_index)
            ).save(image)
            views[view] = ViewReference(
                image_id=f"image-{index}-{view}",
                path=image.name,
                content_sha256=sha256_file(image),
            )
        manifests.append(
            RoleManifest(
                dataset_namespace=namespace,
                source_hashes={"source": sha256_file(source)},
                patient_mapping=PatientMappingDeclaration(
                    namespace,
                    sha256_file(mapping),
                    "researcher-verified mapping fixture",
                    "research fixture generator",
                ),
                records=(
                    PrivateExamRecord(
                        dataset_namespace=namespace,
                        exam_key=f"exam-{index}",
                        patient_key=f"patient-{index}",
                        density=index % 4,
                        source_manifest="source",
                        views=views,
                        role=role,
                    ),
                ),
            )
        )
    audit = audit_software_fixture(tuple(manifests))
    return tuple(manifests), audit


def test_self_hashed_status_cannot_forge_real_readiness(tmp_path) -> None:
    payload = {
        "kind": "real_data",
        "patient_ready": True,
        "dataset_count": 1,
        "exam_count": 4,
        "patient_count": 4,
        "role_exam_counts": {
            role.value: 1
            for role in (Role.CLASSIFIER_FIT, Role.CONFIDENCE_FIT, Role.TUNE, Role.PILOT)
        },
        "locked_isolation_verified": True,
        "manifest_sha256s": [f"{index:064x}" for index in range(1, 5)],
        "locked_denylist_sha256": "f" * 64,
        "verified_file_sha256s": [],
        "verified_files": {},
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    path = tmp_path / "forged-readiness.json"
    path.write_text(
        json.dumps(
            {"audit_sha256": hashlib.sha256(encoded).hexdigest(), "audit": payload}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="readiness.*evidence|unavailable|invalid"):
        load_verified_readiness_audit(path)


def test_production_classifier_rejects_mismatched_dataset_before_artifact_or_image_read(
    tmp_path,
) -> None:
    manifests, _readiness = _software_fixture(tmp_path, namespace="unrelated-study")
    image_reader_called = False

    def image_reader(*_args):
        nonlocal image_reader_called
        image_reader_called = True
        raise AssertionError("dataset mismatch must precede image access")

    with pytest.raises(ValueError, match="dataset namespace"):
        fit_classifier_images_with_role_access(
            artifact_path=tmp_path / "must-not-exist.json",
            manifest=manifests[0],
            manifest_path=tmp_path / "must-not-open-manifest.json",
            model=nn.Linear(1, 1),
            input_ids=torch.zeros(4, 1, dtype=torch.long),
            initialization=None,  # type: ignore[arg-type]
            readiness_artifact_path=tmp_path / "must-not-open-readiness.json",
            config=ResearchRunConfig.default("RSNA"),
            image_root=tmp_path,
            checkpoint_directory=tmp_path,
            resume_checkpoint_path=tmp_path / "must-not-exist.pt",
            image_reader=image_reader,
        )
    assert image_reader_called is False


def test_production_classifier_artifacts_rederive_identity_from_current_bytes(
    tmp_path, monkeypatch
) -> None:
    class FakeCLIP(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([1.0]))

    config = ResearchRunConfig.default("RSNA")

    class ModelLoader:
        @staticmethod
        def from_pretrained(model_name, *, revision):
            public = pinned_public_clip_configuration(config.backbone)
            assert (model_name, revision) == (public.hf_model, public.revision)
            return FakeCLIP()

    class FakeTokenizer:
        def __call__(self, prompts, *, padding, return_tensors):
            assert tuple(prompts) == tuple(PROMPTS)
            assert padding is True and return_tensors == "pt"
            return {"input_ids": torch.arange(8).reshape(4, 2)}

    class TokenizerLoader:
        @staticmethod
        def from_pretrained(model_name, *, revision):
            public = pinned_public_clip_configuration(config.backbone)
            assert (model_name, revision) == (public.hf_model, public.revision)
            return FakeTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(CLIPModel=ModelLoader, CLIPTokenizer=TokenizerLoader),
    )
    manifests, readiness = _software_fixture(tmp_path)
    fit_manifest, confidence_manifest, tune_manifest, _pilot_manifest = manifests
    model, input_ids, initialization = load_pinned_public_clip_classifier(
        config.backbone, config.dataset
    )
    initialization_artifact = save_public_initialization_artifact(
        tmp_path / "public-init.json",
        model=model,
        input_ids=input_ids,
        provenance=initialization,
        dataset=config.dataset,
    )
    assert (
        load_public_initialization_artifact(initialization_artifact.source_path).classifier
        == initialization
    )
    readiness_path = save_readiness_audit(readiness, tmp_path / "readiness.json")
    fit_manifest_path = tmp_path / "classifier-fit-manifest.json"
    tune_manifest_path = tmp_path / "tune-manifest.json"
    fit_manifest_path.write_bytes(b"validated classifier-fit manifest fixture\n")
    tune_manifest_path.write_bytes(b"validated tune manifest fixture\n")
    binding = TrainingBinding(
        protocol_sha256=config.protocol_sha256,
        config_sha256=config.sha256,
        search_table_sha256=config.search_table.sha256,
        manifest_sha256=fit_manifest.manifest_sha256,
        classifier_checkpoint_sha256=initialization.checkpoint_sha256,
        method="fresh_classifier",
        seed=42,
    )

    def batches(records, _epoch):
        yield RoleBoundBatch(tuple(record.exam_key for record in records), None)

    checkpoint_paths = {}

    def save_epoch(epoch, current):
        checkpoint_paths[epoch] = save_tensor_checkpoint(
            current, tmp_path / f"classifier-epoch-{epoch}.safetensors"
        )

    fitted = fit_fresh_classifier_with_role_access(
        manifest=fit_manifest,
        model=model,
        classifier=initialization,
        readiness=readiness,
        optimizer_config=OptimizerConfig(),
        epochs=config.epochs,
        binding=binding,
        batch_loader=batches,
        loss_fn=lambda current, _payload: sum(
            parameter.square().sum() for parameter in current.parameters()
        ),
        epoch_callback=save_epoch,
    )
    with pytest.raises(ValueError, match="final checkpoint state"):
        save_classifier_fit_artifact(
            tmp_path / "classifier-fit-forged-final.json",
            config=config,
            initialization=initialization_artifact,
            readiness_artifact_path=readiness_path,
            manifest=fit_manifest,
            manifest_path=fit_manifest_path,
            binding=binding,
            result=replace(fitted, model_state_sha256="f" * 64),
            checkpoint_paths=checkpoint_paths,
        )
    fit_artifact = save_classifier_fit_artifact(
        tmp_path / "classifier-fit.json",
        config=config,
        initialization=initialization_artifact,
        readiness_artifact_path=readiness_path,
        manifest=fit_manifest,
        manifest_path=fit_manifest_path,
        binding=binding,
        result=fitted,
        checkpoint_paths=checkpoint_paths,
    )
    unaudited_tune = RoleManifest(
        dataset_namespace=tune_manifest.dataset_namespace,
        source_hashes={"source": "1" * 64},
        patient_mapping=tune_manifest.patient_mapping,
        records=tune_manifest.records,
    )
    unaudited_tune_path = tmp_path / "unaudited-tune.json"
    unaudited_tune_path.write_bytes(b"unaudited tune manifest fixture\n")
    checkpoints = tuple(
        ClassifierTuneCheckpoint(epoch, sha256_file(checkpoint_paths[epoch]), 0.25)
        for epoch in config.search_table.checkpoint_epochs
    )
    selected = select_fresh_classifier_on_tune(
        tune_manifest,
        initialization=initialization,
        classifier_fit_result=fitted,
        readiness=readiness,
        checkpoint_evaluator=lambda _records: checkpoints,
    )
    unaudited_selected = select_fresh_classifier_on_tune(
        unaudited_tune,
        initialization=initialization,
        classifier_fit_result=fitted,
        readiness=readiness,
        checkpoint_evaluator=lambda _records: checkpoints,
    )
    with pytest.raises(ValueError, match="readiness|audit"):
        save_selected_classifier_artifact(
            tmp_path / "unaudited-selected.json",
            config=config,
            initialization=initialization_artifact,
            readiness_artifact_path=readiness_path,
            fit_artifact_path=fit_artifact.source_path,
            fit_binding=binding,
            fit_result=fitted,
            tune_manifest=unaudited_tune,
            tune_manifest_path=unaudited_tune_path,
            tune_checkpoints=checkpoints,
            checkpoint_paths=checkpoint_paths,
            selected=unaudited_selected,
        )
    artifact = save_selected_classifier_artifact(
        tmp_path / "selected-classifier.json",
        config=config,
        initialization=initialization_artifact,
        readiness_artifact_path=readiness_path,
        fit_artifact_path=fit_artifact.source_path,
        fit_binding=binding,
        fit_result=fitted,
        tune_manifest=tune_manifest,
        tune_manifest_path=tune_manifest_path,
        tune_checkpoints=checkpoints,
        checkpoint_paths=checkpoint_paths,
        selected=selected,
    )
    reloaded = load_selected_classifier_artifact(
        artifact.source_path, expected_config=config
    )
    assert reloaded.classifier == selected
    assert reloaded.classifier.workflow_complete is True
    assert reloaded.classifier.pilot_eligible is False

    forged_document = json.loads(artifact.source_path.read_text(encoding="utf-8"))
    forged_payload = forged_document["artifact"]
    forged_payload["tune_checkpoints"] = forged_payload["tune_checkpoints"][:1]
    selected_record = forged_payload["tune_checkpoints"][0]
    forged_payload["selected_epoch"] = selected_record["epoch"]
    selection_payload = {
        "metric": "multiclass_nll",
        "role": "tune",
        "manifest_sha256": tune_manifest.manifest_sha256,
        "candidates": [
            {
                "epoch": selected_record["epoch"],
                "artifact_sha256": selected_record["artifact_sha256"],
                "tune_nll": selected_record["tune_nll"],
            }
        ],
        "selected_artifact_sha256": selected_record["artifact_sha256"],
    }
    forged_payload["classifier"]["tune_selection_sha256"] = hashlib.sha256(
        json.dumps(
            selection_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    forged_document["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            forged_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    forged_selected_path = tmp_path / "self-rehashed-partial-selection.json"
    forged_selected_path.write_text(json.dumps(forged_document), encoding="utf-8")
    with pytest.raises(ValueError, match="budget|configuration"):
        load_selected_classifier_artifact(forged_selected_path)

    ddsm_model, ddsm_input_ids, ddsm_provenance = load_pinned_public_clip_classifier(
        config.backbone, "DDSM"
    )
    ddsm_initialization = save_public_initialization_artifact(
        tmp_path / "ddsm-public-init.json",
        model=ddsm_model,
        input_ids=ddsm_input_ids,
        provenance=ddsm_provenance,
        dataset="DDSM",
    )
    image_reader_called = False

    def image_reader(*_args):
        nonlocal image_reader_called
        image_reader_called = True
        raise AssertionError("initialization mismatch must precede image access")

    with pytest.raises(ValueError, match="initialization.*dataset|dataset.*initialization"):
        fit_classifier_images_with_role_access(
            artifact_path=tmp_path / "mismatched-fit.json",
            manifest=fit_manifest,
            manifest_path=fit_manifest_path,
            model=ddsm_model,
            input_ids=ddsm_input_ids,
            initialization=ddsm_initialization,
            readiness_artifact_path=readiness_path,
            config=config,
            image_root=tmp_path,
            checkpoint_directory=tmp_path,
            resume_checkpoint_path=tmp_path / "mismatched-resume.pt",
            stop_after_epoch=1,
            image_reader=image_reader,
        )
    assert image_reader_called is False
    initialization_artifact.weights_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="initialization|checkpoint|tensor"):
        load_selected_classifier_artifact(artifact.source_path, expected_config=config)


def test_role_bound_image_fit_and_tune_selection_are_connected(tmp_path, monkeypatch) -> None:
    class FakeCLIP(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual_projection = nn.Linear(768, 4, bias=False)
            with torch.no_grad():
                self.visual_projection.weight.copy_(
                    torch.linspace(-0.2, 0.2, 4 * 768).reshape(4, 768)
                )
            self.logit_scale = nn.Parameter(torch.tensor(0.0))

            class Vision(nn.Module):
                def forward(self, *, pixel_values):
                    value = pixel_values.mean(dim=(1, 2, 3))
                    pooled = value[:, None].repeat(1, 768)
                    return SimpleNamespace(
                        pooler_output=pooled,
                        last_hidden_state=torch.stack((pooled, pooled + 0.1), dim=1),
                    )

            self.vision_model = Vision()

        def get_image_features(self, *, pixel_values):
            vision = self.vision_model(pixel_values=pixel_values)
            return self.visual_projection(vision.pooler_output)

        def get_text_features(self, *, input_ids):
            return F.one_hot(input_ids[:, 0] % 4, num_classes=4).float()

    class ModelLoader:
        @staticmethod
        def from_pretrained(_model, *, revision):
            assert revision
            return FakeCLIP()

    class FakeTokenizer:
        def __call__(self, prompts, *, padding, return_tensors):
            assert padding is True and return_tensors == "pt"
            return {"input_ids": torch.arange(len(prompts)).reshape(-1, 1)}

    class TokenizerLoader:
        @staticmethod
        def from_pretrained(_model, *, revision):
            assert revision
            return FakeTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(CLIPModel=ModelLoader, CLIPTokenizer=TokenizerLoader),
    )
    manifests, readiness = _software_fixture(tmp_path)
    fit_manifest, confidence_manifest, tune_manifest, _pilot_manifest = manifests
    config = ResearchRunConfig.default("RSNA")
    config_path = tmp_path / "workflow-config.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    readiness_path = save_readiness_audit(readiness, tmp_path / "workflow-readiness.json")
    fit_manifest_path = tmp_path / "workflow-fit-manifest.json"
    tune_manifest_path = tmp_path / "workflow-tune-manifest.json"
    confidence_manifest_path = tmp_path / "workflow-confidence-manifest.json"
    fit_binding = save_private_manifest(
        fit_manifest, fit_manifest_path, private_root=tmp_path
    )
    tune_binding = save_private_manifest(
        tune_manifest, tune_manifest_path, private_root=tmp_path
    )
    confidence_binding = save_private_manifest(
        confidence_manifest, confidence_manifest_path, private_root=tmp_path
    )
    fit_binding_path = tmp_path / "workflow-fit-binding.json"
    tune_binding_path = tmp_path / "workflow-tune-binding.json"
    confidence_binding_path = tmp_path / "workflow-confidence-binding.json"
    fit_binding_path.write_text(
        json.dumps(
            {
                "schema_version": fit_binding.schema_version,
                "dataset_namespace": fit_binding.dataset_namespace,
                "source_hashes": dict(fit_binding.source_hashes),
                "manifest_sha256": fit_binding.manifest_sha256,
            }
        ),
        encoding="utf-8",
    )
    tune_binding_path.write_text(
        json.dumps(
            {
                "schema_version": tune_binding.schema_version,
                "dataset_namespace": tune_binding.dataset_namespace,
                "source_hashes": dict(tune_binding.source_hashes),
                "manifest_sha256": tune_binding.manifest_sha256,
            }
        ),
        encoding="utf-8",
    )
    confidence_binding_path.write_text(
        json.dumps(
            {
                "schema_version": confidence_binding.schema_version,
                "dataset_namespace": confidence_binding.dataset_namespace,
                "source_hashes": dict(confidence_binding.source_hashes),
                "manifest_sha256": confidence_binding.manifest_sha256,
            }
        ),
        encoding="utf-8",
    )
    parser = build_parser()
    fit_result = run_view_risk_command(
        parser.parse_args(
            [
                "view-risk-fit-classifier",
                "--config", str(config_path),
                "--manifest", str(fit_manifest_path),
                "--manifest-binding", str(fit_binding_path),
                "--private-root", str(tmp_path),
                "--image-root", str(tmp_path),
                "--readiness", str(readiness_path),
                "--initialization", str(tmp_path / "workflow-init.json"),
                "--checkpoint-directory", str(tmp_path),
                "--resume-checkpoint", str(tmp_path / "workflow-resume.pt"),
                "--output", str(tmp_path / "workflow-fit.json"),
            ]
        )
    )
    assert fit_result["status"] == "classifier_fit_complete"
    assert fit_result["patient_readiness"] == "not_verified"
    selected_result = run_view_risk_command(
        parser.parse_args(
            [
                "view-risk-select-classifier",
                "--config", str(config_path),
                "--fit-artifact", str(tmp_path / "workflow-fit.json"),
                "--manifest", str(tune_manifest_path),
                "--manifest-binding", str(tune_binding_path),
                "--private-root", str(tmp_path),
                "--image-root", str(tmp_path),
                "--output", str(tmp_path / "workflow-selected.json"),
            ]
        )
    )
    assert selected_result["status"] == "classifier_selected"
    selected = load_selected_classifier_artifact(
        tmp_path / "workflow-selected.json", expected_config=config
    )
    assert selected.classifier.workflow_complete is True
    assert selected.classifier.pilot_eligible is False
    assert selected.selected_checkpoint_path.exists()
    with pytest.raises(PermissionError, match="readiness"):
        freeze_pilot_plan(
            tmp_path / "must-not-promote-software-plan.json",
            config=config,
            manifest_sha256_by_role={
                Role.CLASSIFIER_FIT.value: fit_manifest.manifest_sha256,
                Role.CONFIDENCE_FIT.value: confidence_manifest.manifest_sha256,
                Role.TUNE.value: tune_manifest.manifest_sha256,
                Role.PILOT.value: manifests[3].manifest_sha256,
                Role.LOCKED_TEST.value: "f" * 64,
            },
            classifier=selected.classifier,
            selections=(),
            classifier_artifact_path=selected.source_path,
        )

    control_paths = {}
    for method in ("msp", "margin", "negative_entropy", "energy", "absolute_omission_sensitivity"):
        output = tmp_path / f"control-{method}.json"
        result = run_view_risk_command(
            parser.parse_args(
                [
                    "view-risk-fit-control",
                    "--config", str(config_path),
                    "--classifier-artifact", str(selected.source_path),
                    "--method", method,
                    "--seed", "42",
                    "--output", str(output),
                ]
            )
        )
        assert result["status"] == "control_recorded"
        assert load_control_artifact(output).method == method
        control_paths[method] = output

    scalar_input = tmp_path / "temperature-input.json"
    scalar_input.write_text(
        json.dumps(
            {
                "schema_version": "view-risk-scalar-control-input/v1",
                "exam_keys": [tune_manifest.records[0].exam_key],
                "scores": [[0.1, 0.2, 0.3, 0.4]],
            }
        ),
        encoding="utf-8",
    )
    temperature_output = tmp_path / "control-temperature.json"
    result = run_view_risk_command(
        parser.parse_args(
            [
                "view-risk-fit-control",
                "--config", str(config_path),
                "--classifier-artifact", str(selected.source_path),
                "--method", "temperature_scaled_msp",
                "--seed", "42",
                "--output", str(temperature_output),
                "--tune-manifest", str(tune_manifest_path),
                "--tune-manifest-binding", str(tune_binding_path),
                "--tune-private-root", str(tmp_path),
                "--tune-input", str(scalar_input),
            ]
        )
    )
    assert result["status"] == "control_fitted"
    control_paths["temperature_scaled_msp"] = temperature_output

    def ds_document(manifest, prediction):
        return {
            "schema_version": "view-risk-ds-control-input/v1",
            "exam_keys": [manifest.records[0].exam_key],
            "values": [[0.2] * 8],
            "valid_mask": [[True] * 8],
            "observed_mask": [[True] * 4],
            "conflict_valid_mask": [[True] * 3],
            "fusion_pairs": [list(pair) for pair in RSNA_FUSION_PAIRS],
            "classifier_prediction": [prediction],
        }

    confidence_input = tmp_path / "ds-confidence-input.json"
    tune_input = tmp_path / "ds-tune-input.json"
    confidence_input.write_text(
        json.dumps(ds_document(confidence_manifest, 0)), encoding="utf-8"
    )
    tune_input.write_text(json.dumps(ds_document(tune_manifest, 2)), encoding="utf-8")
    ds_output = tmp_path / "control-ds.json"
    result = run_view_risk_command(
        parser.parse_args(
            [
                "view-risk-fit-control",
                "--config", str(config_path),
                "--classifier-artifact", str(selected.source_path),
                "--method", "ds_logistic",
                "--seed", "42",
                "--output", str(ds_output),
                "--tune-manifest", str(tune_manifest_path),
                "--tune-manifest-binding", str(tune_binding_path),
                "--tune-private-root", str(tmp_path),
                "--tune-input", str(tune_input),
                "--confidence-manifest", str(confidence_manifest_path),
                "--confidence-manifest-binding", str(confidence_binding_path),
                "--confidence-private-root", str(tmp_path),
                "--confidence-input", str(confidence_input),
            ]
        )
    )
    assert result["status"] == "control_fitted"
    assert load_control_artifact(ds_output).selection_trials == len(
        config.search_table.ds_regularizations
    )
    control_paths["ds_logistic"] = ds_output

    classifier_model, classifier_tokens = reload_verified_public_classifier(
        selected.initialization
    )
    encoder = load_verified_frozen_encoder(
        classifier_model,
        classifier_tokens,
        selected.selected_checkpoint_path,
        backbone=config.backbone,
        prompts=PROMPTS,
    )
    images = {
        view: torch.full((3, 32, 32), 0.2 + index * 0.15)
        for index, view in enumerate(("L_CC", "L_MLO", "R_CC", "R_MLO"))
    }
    training_entries = []
    for epoch in config.search_table.checkpoint_epochs:
        draw = build_method_training_schedule(
            confidence_manifest.records,
            epoch=epoch,
            seed=42,
            method="correctness_mvacn",
        )[0].draw
        bundle = build_exam_cache_with_role_access(
            confidence_manifest,
            operation=Operation.CONFIDENCE_FITTING,
            role=Role.CONFIDENCE_FIT,
            exam_key=confidence_manifest.records[0].exam_key,
            sample_key=f"confidence-epoch-{epoch}",
            parent_loader=lambda _record, current=draw: realize_training_parent(
                images, current
            ),
            encoder=encoder,
            implementation_revision="p4b-command-integration",
        )
        cache_path = save_cache_bundle(
            bundle, tmp_path / f"confidence-cache-{epoch}.json"
        ).metadata
        training_entries.append(
            {
                "epoch": epoch,
                "metadata_path": str(cache_path),
                "provenance": bundle.provenance.to_dict(),
            }
        )
    training_cache_index = tmp_path / "confidence-training-index.json"
    training_cache_index.write_text(
        json.dumps(
            {
                "schema_version": "view-risk-training-cache-index/v1",
                "classifier": selected.classifier.to_dict(),
                "entries": training_entries,
            }
        ),
        encoding="utf-8",
    )
    confidence_fit_result = run_view_risk_command(
        parser.parse_args(
            [
                "view-risk-train-confidence",
                "--config", str(config_path),
                "--manifest", str(confidence_manifest_path),
                "--manifest-binding", str(confidence_binding_path),
                "--private-root", str(tmp_path),
                "--cache-index", str(training_cache_index),
                "--method", "correctness_mvacn",
                "--seed", "42",
                "--checkpoint", str(tmp_path / "confidence-resume.pt"),
                "--classifier-artifact", str(selected.source_path),
                "--epoch-checkpoint-directory", str(tmp_path),
                "--training-artifact", str(tmp_path / "confidence-fit-artifact.json"),
            ]
        )
    )
    assert confidence_fit_result["status"] == "trained"
    confidence_fit = load_confidence_fit_artifact(
        tmp_path / "confidence-fit-artifact.json", expected_config=config
    )
    assert confidence_fit.result.completed_epoch == config.epochs

    tune_entries = []
    tune_cells = [
        (None, None, None),
        *[
            (family, severity, view)
            for family in ("gaussian_noise", "gaussian_blur")
            for severity in ("mild", "moderate")
            for view in ("L_CC", "L_MLO", "R_CC", "R_MLO")
        ],
    ]
    for cell_index, (family, severity, target_view) in enumerate(tune_cells):
        if family is None:
            parent = realize_parent(images, tuple(images))
        else:
            spec = PerturbationSpec.for_sample(
                family,
                severity,
                private_sample_key="tune-exam",
                cell=f"tune/{family}/{severity}/{target_view}",
                seed=config.stress_seed,
            )
            parent = realize_parent(
                images, tuple(images), perturbations={target_view: spec}
            )
        bundle = build_exam_cache_with_role_access(
            tune_manifest,
            operation=Operation.TUNE_SELECTION,
            role=Role.TUNE,
            exam_key=tune_manifest.records[0].exam_key,
            sample_key=f"tune-cell-{cell_index}",
            parent_loader=lambda _record, current=parent: current,
            encoder=encoder,
            implementation_revision="p4b-command-integration",
        )
        cache_path = save_cache_bundle(
            bundle, tmp_path / f"tune-cache-{cell_index}.json"
        ).metadata
        tune_entries.append(
            {
                "family": family,
                "severity": severity,
                "target_view": target_view,
                "metadata_path": str(cache_path),
                "provenance": bundle.provenance.to_dict(),
            }
        )
    tune_cache_index = tmp_path / "confidence-tune-index.json"
    tune_cache_index.write_text(
        json.dumps(
            {
                "schema_version": "view-risk-tune-cache-index/v1",
                "entries": tune_entries,
            }
        ),
        encoding="utf-8",
    )
    confidence_selection_result = run_view_risk_command(
        parser.parse_args(
            [
                "view-risk-select-confidence",
                "--config", str(config_path),
                "--training-artifact", str(confidence_fit.source_path),
                "--manifest", str(tune_manifest_path),
                "--manifest-binding", str(tune_binding_path),
                "--private-root", str(tmp_path),
                "--tune-cache-index", str(tune_cache_index),
                "--output", str(tmp_path / "correctness-selection.json"),
            ]
        )
    )
    assert confidence_selection_result["status"] == "selected"
    confidence_selection = load_model_artifact_selection(
        tmp_path / "correctness-selection.json",
        config=config,
        classifier=selected.classifier,
    )
    assert confidence_selection.selection_trials == config.epochs
    reload_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json,sys;"
                "from mmdc_clip_f.research.view_risk.training import ResearchRunConfig;"
                "from mmdc_clip_f.research.view_risk.production import "
                "load_selected_classifier_artifact;"
                "from mmdc_clip_f.research.view_risk.evaluation import "
                "load_model_artifact_selection;"
                "cfg=ResearchRunConfig.from_dict(json.load(open(sys.argv[1])));"
                "clf=load_selected_classifier_artifact(sys.argv[2],expected_config=cfg);"
                "sel=load_model_artifact_selection(sys.argv[3],config=cfg,"
                "classifier=clf.classifier);"
                "assert sel.selection_trials==cfg.epochs"
            ),
            str(config_path),
            str(selected.source_path),
            str(confidence_selection.workflow_evidence_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert reload_probe.returncode == 0, reload_probe.stderr
    example = build_exam_cache_with_role_access(
        tune_manifest,
        operation=Operation.TUNE_SELECTION,
        role=Role.TUNE,
        exam_key=tune_manifest.records[0].exam_key,
        sample_key="reload-score",
        parent_loader=lambda _record: realize_parent(images, tuple(images)),
        encoder=encoder,
        implementation_revision="p4b-command-integration",
    )
    for method, control_path in control_paths.items():
        control_selection = load_model_artifact_selection(
            control_path,
            config=config,
            classifier=selected.classifier,
            clean_reference_record_path=confidence_selection.workflow_evidence_path,
        )
        control_score = score_confidence_cache_bundle(control_selection, example)
        assert torch.equal(
            control_score.classifier_prediction, example.features.prediction
        ), method
    confidence_model = load_learned_confidence_model(
        confidence_selection,
        config=config,
        classifier=selected.classifier,
        example=example,
    )
    scored = score_confidence_cache_bundle(
        confidence_selection, example, learned_model=confidence_model
    )
    assert torch.equal(scored.classifier_prediction, example.features.prediction)

    scalar_input.write_text("changed tune fitting evidence", encoding="utf-8")
    with pytest.raises(ValueError, match="control.*evidence|evidence.*changed"):
        load_model_artifact_selection(
            temperature_output,
            config=config,
            classifier=selected.classifier,
            clean_reference_record_path=confidence_selection.workflow_evidence_path,
        )


def test_confidence_fit_artifact_rehashes_model_manifest_and_cache_bytes(tmp_path) -> None:
    config = ResearchRunConfig.default("RSNA")
    manifests, _readiness = _software_fixture(tmp_path)
    classifier_fit_manifest, fit_manifest = manifests[:2]
    initialization = ClassifierProvenance.synthetic_injected("vit_b_32", "a" * 64)
    classifier = ClassifierProvenance.fresh_selected(
        initialization,
        checkpoint_sha256="c" * 64,
        classifier_fit_manifest_sha256=classifier_fit_manifest.manifest_sha256,
        classifier_fit_update_count=1,
        tune_selection_sha256="d" * 64,
        tune_manifest_sha256="e" * 64,
        readiness=audit_software_fixture(classifier_fit_manifest),
    )
    binding = TrainingBinding(
        protocol_sha256=config.protocol_sha256,
        config_sha256=config.sha256,
        search_table_sha256=config.search_table.sha256,
        manifest_sha256=fit_manifest.manifest_sha256,
        classifier_checkpoint_sha256=classifier.checkpoint_sha256,
        method="correctness_mvacn",
        seed=42,
    )
    result = TrainingResult(
        method="correctness_mvacn",
        seed=42,
        completed_epoch=config.epochs,
        update_count=config.epochs,
        exposed_record_count=len(fit_manifest.records),
        parameter_count=2,
        model_state_sha256="placeholder",
        manifest_sha256=fit_manifest.manifest_sha256,
        scaling="declared_identity_no_fitted_scaler",
        selection_trial_budget=config.epochs,
        actual_exposure_verified=True,
        software_only=True,
    )
    manifest_path = tmp_path / "confidence-manifest.json"
    cache_index = tmp_path / "confidence-cache-index.json"
    manifest_path.write_text("role-bound synthetic manifest", encoding="utf-8")
    cache_index.write_text("bounded synthetic cache index", encoding="utf-8")
    checkpoints = {}
    for epoch in config.search_table.checkpoint_epochs:
        model = nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.fill_(epoch)
        checkpoints[epoch] = save_tensor_checkpoint(
            model, tmp_path / f"confidence-{epoch}.safetensors"
        )
    final_state_sha256 = state_dict_sha256(
        load_tensor_checkpoint_state(checkpoints[config.epochs])
    )
    result = replace(result, model_state_sha256=final_state_sha256)
    with pytest.raises(ValueError, match="final checkpoint state"):
        save_confidence_fit_artifact(
            tmp_path / "confidence-fit-forged-final.json",
            config=config,
            classifier=classifier,
            classifier_artifact_path=None,
            manifest=fit_manifest,
            manifest_path=manifest_path,
            input_evidence_path=cache_index,
            binding=binding,
            result=replace(result, model_state_sha256="f" * 64),
            checkpoint_paths=checkpoints,
        )
    artifact = save_confidence_fit_artifact(
        tmp_path / "confidence-fit.json",
        config=config,
        classifier=classifier,
        classifier_artifact_path=None,
        manifest=fit_manifest,
        manifest_path=manifest_path,
        input_evidence_path=cache_index,
        binding=binding,
        result=result,
        checkpoint_paths=checkpoints,
    )
    assert load_confidence_fit_artifact(
        artifact.source_path, expected_config=config
    ).sha256 == artifact.sha256
    cache_index.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest/cache evidence changed"):
        load_confidence_fit_artifact(artifact.source_path, expected_config=config)
