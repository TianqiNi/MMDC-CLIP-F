from __future__ import annotations

from dataclasses import replace
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from mmdc_clip_f.research.view_risk.cache import CachedTargets
from mmdc_clip_f.research.view_risk.features import FrozenViewFeatures
from mmdc_clip_f.research.view_risk.fusion import RSNA_FUSION_PAIRS, fuse_view_logits
from mmdc_clip_f.research.view_risk.roles import (
    Operation,
    PatientMappingDeclaration,
    PrivateExamRecord,
    Role,
    RoleManifest,
    ViewReference,
    ManifestBinding,
    LockedPatientIdentityDenylist,
)
from mmdc_clip_f.research.view_risk.training import (
    MANDATORY_METHODS,
    ClassifierProvenance,
    OptimizerConfig,
    ResearchRunConfig,
    RoleBoundBatch,
    TrainingBinding,
    audit_software_fixture,
    audit_real_data_readiness,
    build_learned_method,
    build_method_training_schedule,
    build_training_schedule,
    checked_intervention_targets,
    fit_role_bound_module,
    fit_fresh_classifier_with_role_access,
    initialize_public_classifier,
    load_pinned_public_clip_classifier,
    module_state_sha256,
    load_role_manifest_for_operation,
)


def _manifest(role: Role, count: int = 4) -> RoleManifest:
    return RoleManifest(
        dataset_namespace="synthetic-p4b",
        source_hashes={"fixture": "a" * 64},
        patient_mapping=PatientMappingDeclaration(
            "synthetic-p4b", "b" * 64, "verified synthetic grouping", "test fixture"
        ),
        records=[
            PrivateExamRecord(
                dataset_namespace="synthetic-p4b",
                exam_key=f"exam-{index}",
                patient_key=f"patient-{index}",
                density=index % 4,
                source_manifest="fixture",
                views={
                    view: ViewReference(
                        image_id=f"image-{index}-{view}", path=f"fixture/{index}/{view}.png"
                    )
                    for view in ("L_CC", "L_MLO", "R_CC", "R_MLO")
                },
                role=role,
            )
            for index in range(count)
        ],
    )


def _binding(manifest: RoleManifest, method: str = "tiny") -> TrainingBinding:
    return TrainingBinding(
        protocol_sha256="1" * 64,
        config_sha256="2" * 64,
        search_table_sha256="3" * 64,
        manifest_sha256=manifest.manifest_sha256,
        classifier_checkpoint_sha256="4" * 64,
        method=method,
        seed=42,
        schedule_version="view-risk-training-schedule/v1",
    )


def test_defaults_are_frozen_complete_and_unknown_configuration_is_rejected() -> None:
    rsna = ResearchRunConfig.default("RSNA", "vit_b_32")
    ddsm = ResearchRunConfig.default("DDSM", "vit_l_14_336")

    assert rsna.optimizer == OptimizerConfig("adam", 1e-4, 0.0, 6)
    assert rsna.epochs == 20
    assert ddsm.epochs == 50
    assert rsna.seeds == (42, 43, 44)
    assert rsna.stress_seed == 4242
    assert tuple(rsna.methods) == MANDATORY_METHODS
    assert rsna.search_table.checkpoint_epochs == tuple(range(1, 21))
    assert ddsm.search_table.checkpoint_epochs == tuple(range(1, 51))

    with pytest.raises(ValueError, match="unknown"):
        ResearchRunConfig.from_dict({"dataset": "RSNA", "backbone": "vit_b_32", "oops": 1})
    with pytest.raises(ValueError, match="mandatory"):
        ResearchRunConfig.from_dict(
            {"dataset": "RSNA", "backbone": "vit_b_32", "methods": ["candidate"]}
        )
    with pytest.raises(ValueError, match="optimizer"):
        ResearchRunConfig.from_dict(
            {
                "dataset": "RSNA",
                "backbone": "vit_b_32",
                "optimizer": {"name": "sgd"},
            }
        )
    with pytest.raises(ValueError, match="learning_rate"):
        ResearchRunConfig.from_dict(
            {
                "dataset": "RSNA",
                "backbone": "vit_b_32",
                "optimizer": {"learning_rate": True},
            }
        )
    with pytest.raises(ValueError, match="20 epochs"):
        ResearchRunConfig.from_dict(
            {"dataset": "RSNA", "backbone": "vit_b_32", "epochs": 20.0}
        )


def test_role_is_rejected_before_training_reader_hook() -> None:
    manifest = _manifest(Role.TUNE)
    called = False

    def batches(_records, _epoch):
        nonlocal called
        called = True
        raise AssertionError("must not run")

    with pytest.raises(PermissionError):
        fit_role_bound_module(
            manifest=manifest,
            operation=Operation.CONFIDENCE_FITTING,
            role=Role.TUNE,
            model=nn.Linear(1, 1),
            optimizer_config=OptimizerConfig(batch_size=2),
            epochs=1,
            binding=_binding(manifest),
            batch_loader=batches,
            loss_fn=lambda model, payload: model(payload).square().mean(),
        )
    assert called is False

    with pytest.raises(PermissionError, match="locked_test"):
        load_role_manifest_for_operation(
            "/definitely/not/a/readable/manifest.json",
            expected=ManifestBinding(
                schema_version="view-risk-role-manifest/v1",
                dataset_namespace="synthetic-p4b",
                source_hashes={"fixture": "a" * 64},
                manifest_sha256="b" * 64,
            ),
            private_root="/definitely/not/a/readable",
            operation=Operation.PILOT_EVALUATION,
            role=Role.LOCKED_TEST,
        )


def test_public_fresh_and_diagnostic_classifier_identities_are_not_interchangeable() -> None:
    fresh = ClassifierProvenance.synthetic_injected("vit_b_32", "a" * 64)
    diagnostic = ClassifierProvenance.diagnostic_original(
        "vit_b_32", "a" * 64, reason="training exposure overlaps repartitioned roles"
    )

    assert fresh.workflow_complete is False
    assert fresh.pilot_eligible is False
    selected = ClassifierProvenance.fresh_selected(
        fresh,
        checkpoint_sha256="c" * 64,
        classifier_fit_manifest_sha256=_manifest(Role.CLASSIFIER_FIT).manifest_sha256,
        classifier_fit_update_count=2,
        tune_selection_sha256="e" * 64,
        tune_manifest_sha256="f" * 64,
        readiness=audit_software_fixture(_manifest(Role.CLASSIFIER_FIT)),
    )
    assert selected.workflow_complete is True
    assert selected.pilot_eligible is False
    assert diagnostic.pilot_eligible is False
    with pytest.raises(PermissionError, match="diagnostic"):
        diagnostic.require_pilot_eligible()
    assert audit_software_fixture(_manifest(Role.CLASSIFIER_FIT)).patient_ready is False


def test_schedule_is_method_independent_and_never_uses_held_out_stresses() -> None:
    records = _manifest(Role.CONFIDENCE_FIT).records
    first = build_training_schedule(records, epoch=3, seed=42)
    second = build_training_schedule(records, epoch=3, seed=42)
    assert first == second
    assert {item.exam_key for item in first} == {record.exam_key for record in records}
    for item in first:
        if item.draw.perturbation is not None:
            assert item.draw.perturbation.family in {"gaussian_noise", "gaussian_blur"}
            assert item.draw.perturbation.severity in {"mild", "moderate"}

    matched = build_method_training_schedule(
        records, epoch=3, seed=42, method="matched_mvacn_tcp"
    )
    original = build_method_training_schedule(
        records, epoch=3, seed=42, method="original_mvacn_tcp"
    )
    correctness = build_method_training_schedule(
        records, epoch=3, seed=42, method="correctness_mvacn"
    )
    assert matched == first
    assert all(item.draw.stratum == "clean4" for item in original)
    assert all(item.draw.perturbation is None for item in correctness)
    assert [item.draw.observed_views for item in correctness] == [
        item.draw.observed_views for item in first
    ]


def _features(observed=("L_CC", "L_MLO"), *, changed: bool = False) -> FrozenViewFeatures:
    logits = {
        view: torch.tensor([[3.0 + index, 1.0, 0.0, -1.0]])
        for index, view in enumerate(("L_CC", "L_MLO", "R_CC", "R_MLO"))
        if view in observed
    }
    if changed:
        logits[observed[0]] = torch.tensor([[-2.0, 8.0, 0.0, -1.0]])
    fused = fuse_view_logits(logits, observed, fusion_pairs=RSNA_FUSION_PAIRS)
    return FrozenViewFeatures(
        logits_by_view=logits,
        hidden_by_view={view: torch.ones(1, 2, 768) for view in observed},
        projected_by_view={view: torch.ones(1, 8) for view in observed},
        normalized_text_embeddings=torch.eye(8)[:4],
        scores=fused.scores,
        probabilities=fused.scores.softmax(1),
        observed_views=tuple(observed),
        fusion_pairs=RSNA_FUSION_PAIRS,
    )


def _cached(features: FrozenViewFeatures, label: int = 0) -> CachedTargets:
    from mmdc_clip_f.research.view_risk.targets import build_intervention_targets

    labels = torch.tensor([label])
    target = build_intervention_targets(
        features.logits_by_view,
        labels,
        features.observed_views,
        fusion_pairs=features.fusion_pairs,
    )
    probabilities = features.scores.softmax(1)
    return CachedTargets(
        labels=labels,
        observed_prediction=target.observed_prediction,
        observed_error=target.observed_error,
        omission_predictions=target.omission_predictions,
        omission_effects=target.omission_effects,
        omission_labels=target.omission_labels,
        valid_removal_mask=target.valid_removal_mask,
        scores=features.scores,
        probabilities=probabilities,
        tcp=probabilities[:, label],
    )


def test_cached_target_adapter_regenerates_current_targets_and_rejects_changed_parent() -> None:
    original = _features()
    cached = _cached(original)
    regenerated = checked_intervention_targets(original, cached)
    assert torch.equal(regenerated.omission_labels, cached.omission_labels)

    with pytest.raises(ValueError, match="stale"):
        checked_intervention_targets(_features(changed=True), cached)


class _TinyDropout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(1, 1)
        self.dropout = nn.Dropout(0.25)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layer(self.dropout(value))


def _tiny_batches(records, epoch):
    # Deliberately mix conceptual singleton/full masks in each optimizer batch;
    # payloads contain actual supplied features, never placeholder images.
    del epoch
    for start in range(0, len(records), 2):
        selected = records[start : start + 2]
        jitter = random.random() + float(np.random.random())
        x = torch.tensor(
            [[float(index + 1) + jitter] for index in range(len(selected))]
        )
        y = torch.tensor([[0.25] for _ in selected])
        masks = (("L_CC",), ("L_CC", "L_MLO", "R_CC", "R_MLO"))[: len(selected)]
        yield RoleBoundBatch(
            exam_keys=tuple(record.exam_key for record in selected),
            payload=(x, y, masks),
        )


def _tiny_loss(model, payload):
    x, y, masks = payload
    assert all(mask for mask in masks)
    return (model(x) - y).square().mean()


def test_optimizer_updates_resume_exactly_and_preserves_frozen_module(tmp_path) -> None:
    manifest = _manifest(Role.CONFIDENCE_FIT)
    binding = _binding(manifest)
    frozen = nn.Linear(1, 1).requires_grad_(False)
    frozen_before = {name: value.detach().clone() for name, value in frozen.state_dict().items()}

    torch.manual_seed(71)
    uninterrupted = _TinyDropout()
    initial = {name: value.detach().clone() for name, value in uninterrupted.state_dict().items()}
    complete = fit_role_bound_module(
        manifest=manifest,
        operation=Operation.CONFIDENCE_FITTING,
        role=Role.CONFIDENCE_FIT,
        model=uninterrupted,
        optimizer_config=OptimizerConfig(batch_size=2),
        epochs=3,
        binding=binding,
        batch_loader=_tiny_batches,
        loss_fn=_tiny_loss,
        frozen_modules=(frozen,),
    )

    resumed = _TinyDropout()
    resumed.load_state_dict(initial)
    checkpoint = tmp_path / "private-resume.pt"
    partial = fit_role_bound_module(
        manifest=manifest,
        operation=Operation.CONFIDENCE_FITTING,
        role=Role.CONFIDENCE_FIT,
        model=resumed,
        optimizer_config=OptimizerConfig(batch_size=2),
        epochs=3,
        binding=binding,
        batch_loader=_tiny_batches,
        loss_fn=_tiny_loss,
        checkpoint_path=checkpoint,
        stop_after_epoch=1,
        frozen_modules=(frozen,),
    )
    assert partial.completed_epoch == 1
    resumed_result = fit_role_bound_module(
        manifest=manifest,
        operation=Operation.CONFIDENCE_FITTING,
        role=Role.CONFIDENCE_FIT,
        model=resumed,
        optimizer_config=OptimizerConfig(batch_size=2),
        epochs=3,
        binding=binding,
        batch_loader=_tiny_batches,
        loss_fn=_tiny_loss,
        checkpoint_path=checkpoint,
        resume=True,
        frozen_modules=(frozen,),
    )

    assert complete.update_count == resumed_result.update_count == 6
    for name, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
        assert not torch.equal(value, initial[name])
    for name, value in frozen.state_dict().items():
        torch.testing.assert_close(value, frozen_before[name], rtol=0, atol=0)


def test_resume_refuses_binding_change_and_projection_ownership_is_separate(tmp_path) -> None:
    manifest = _manifest(Role.CONFIDENCE_FIT, 2)
    checkpoint = tmp_path / "resume.pt"
    first = _TinyDropout()
    fit_role_bound_module(
        manifest=manifest,
        operation=Operation.CONFIDENCE_FITTING,
        role=Role.CONFIDENCE_FIT,
        model=first,
        optimizer_config=OptimizerConfig(batch_size=2),
        epochs=1,
        binding=_binding(manifest),
        batch_loader=_tiny_batches,
        loss_fn=_tiny_loss,
        checkpoint_path=checkpoint,
    )
    with pytest.raises(ValueError, match="binding"):
        fit_role_bound_module(
            manifest=manifest,
            operation=Operation.CONFIDENCE_FITTING,
            role=Role.CONFIDENCE_FIT,
            model=_TinyDropout(),
            optimizer_config=OptimizerConfig(batch_size=2),
            epochs=2,
            binding=replace(_binding(manifest), method="changed"),
            batch_loader=_tiny_batches,
            loss_fn=_tiny_loss,
            checkpoint_path=checkpoint,
            resume=True,
        )

    candidate = build_learned_method("candidate", "vit_b_32", RSNA_FUSION_PAIRS)
    mlp = build_learned_method("same_input_mlp", "vit_b_32", RSNA_FUSION_PAIRS)
    assert candidate.preparation is not mlp.preparation
    assert candidate.preparation.hidden_projection.weight.data_ptr() != (
        mlp.preparation.hidden_projection.weight.data_ptr()
    )


def test_resume_checkpoint_tampering_is_rejected(tmp_path) -> None:
    manifest = _manifest(Role.CONFIDENCE_FIT, 2)
    checkpoint = tmp_path / "resume.pt"
    model = _TinyDropout()
    fit_role_bound_module(
        manifest=manifest,
        operation=Operation.CONFIDENCE_FITTING,
        role=Role.CONFIDENCE_FIT,
        model=model,
        optimizer_config=OptimizerConfig(batch_size=2),
        epochs=1,
        binding=_binding(manifest),
        batch_loader=_tiny_batches,
        loss_fn=_tiny_loss,
        checkpoint_path=checkpoint,
    )
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        fit_role_bound_module(
            manifest=manifest,
            operation=Operation.CONFIDENCE_FITTING,
            role=Role.CONFIDENCE_FIT,
            model=_TinyDropout(),
            optimizer_config=OptimizerConfig(batch_size=2),
            epochs=2,
            binding=_binding(manifest),
            batch_loader=_tiny_batches,
            loss_fn=_tiny_loss,
            checkpoint_path=checkpoint,
            resume=True,
        )


def test_resume_rejects_unignored_staging_path_before_reader(tmp_path) -> None:
    repository = tmp_path / "private-output-repository"
    repository.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    (repository / ".gitignore").write_text("resume.pt\nresume.json\n", encoding="utf-8")
    manifest = _manifest(Role.CONFIDENCE_FIT, 2)
    called = False

    def batches(_records, _epoch):
        nonlocal called
        called = True
        return ()

    with pytest.raises(ValueError, match="gitignore"):
        fit_role_bound_module(
            manifest=manifest,
            operation=Operation.CONFIDENCE_FITTING,
            role=Role.CONFIDENCE_FIT,
            model=_TinyDropout(),
            optimizer_config=OptimizerConfig(batch_size=2),
            epochs=1,
            binding=_binding(manifest),
            batch_loader=batches,
            loss_fn=_tiny_loss,
            checkpoint_path=repository / "resume.pt",
        )
    assert called is False


def test_injected_classifier_is_explicitly_synthetic_and_not_pilot_eligible() -> None:
    received = None

    def factory(public):
        nonlocal received
        received = (
            public.hf_model,
            public.revision,
            public.prompts,
            public.preprocessing,
            public.image_mean,
            public.image_std,
        )
        return nn.Linear(1, 1)

    model, provenance = initialize_public_classifier("vit_b_32", factory)
    assert received == (
        provenance.hf_model,
        provenance.revision,
        provenance.prompts,
        provenance.preprocessing,
        provenance.image_mean,
        provenance.image_std,
    )
    assert provenance.checkpoint_sha256 == module_state_sha256(model)
    assert provenance.workflow_complete is False
    assert provenance.kind == "synthetic_injected"
    with pytest.raises(PermissionError, match="synthetic"):
        provenance.require_pilot_eligible()


def test_production_loader_uses_exact_public_pins_and_weight_content(monkeypatch) -> None:
    calls = []

    class FakeCLIP(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([1.0]))

    class ModelLoader:
        @staticmethod
        def from_pretrained(model, *, revision):
            calls.append(("model", model, revision))
            return FakeCLIP()

    class FakeTokenizer:
        def __call__(self, prompts, *, padding, return_tensors):
            calls.append(("tokenize", tuple(prompts), padding, return_tensors))
            return {"input_ids": torch.arange(8).reshape(4, 2)}

    class TokenizerLoader:
        @staticmethod
        def from_pretrained(model, *, revision):
            calls.append(("tokenizer", model, revision))
            return FakeTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(CLIPModel=ModelLoader, CLIPTokenizer=TokenizerLoader),
    )
    classifier, input_ids, provenance = load_pinned_public_clip_classifier(
        "vit_b_32", "RSNA"
    )
    assert isinstance(classifier, nn.Module)
    assert input_ids.shape == (4, 2)
    assert calls[0] == ("model", provenance.hf_model, provenance.revision)
    assert calls[1] == ("tokenizer", provenance.hf_model, provenance.revision)
    assert provenance.kind == "public_pretrained_fresh"
    assert provenance.public_weight_content_sha256 is not None
    assert provenance.patient_readiness_verified is False


def test_real_readiness_rejects_unverified_empty_locked_evidence_before_paths() -> None:
    manifests = []
    for index, role in enumerate(
        (Role.CLASSIFIER_FIT, Role.CONFIDENCE_FIT, Role.TUNE, Role.PILOT)
    ):
        template = _manifest(role, 1)
        record = template.records[0]
        manifests.append(
            RoleManifest(
                dataset_namespace=template.dataset_namespace,
                source_hashes=template.source_hashes,
                patient_mapping=template.patient_mapping,
                records=(
                    replace(
                        record,
                        exam_key=f"exam-role-{index}",
                        patient_key=f"patient-role-{index}",
                        views={
                            view: ViewReference(
                                image_id=f"image-role-{index}-{view}",
                                path=f"fixture/role-{index}/{view}.png",
                            )
                            for view in ("L_CC", "L_MLO", "R_CC", "R_MLO")
                        },
                    ),
                ),
            )
        )
    denylist = LockedPatientIdentityDenylist(
        dataset_namespace="synthetic-p4b",
        source_sha256="c" * 64,
        patient_identity_digests=frozenset(),
    )
    with pytest.raises(ValueError, match="locked patient identity evidence"):
        audit_real_data_readiness(
            tuple(manifests),
            locked_patient_denylist=denylist,
            source_files={"fixture": "/does/not/exist"},
            patient_mapping_file="/does/not/exist",
            locked_denylist_file="/does/not/exist",
            image_root="/does/not/exist",
        )


def test_fresh_classifier_fits_only_classifier_role_and_reports_software_readiness() -> None:
    manifest = _manifest(Role.CLASSIFIER_FIT, 2)
    model, provenance = initialize_public_classifier(
        "vit_b_32", lambda _public: _TinyDropout()
    )
    binding = replace(
        _binding(manifest, method="fresh_classifier"),
        classifier_checkpoint_sha256=provenance.checkpoint_sha256,
    )
    result = fit_fresh_classifier_with_role_access(
        manifest=manifest,
        model=model,
        classifier=provenance,
        readiness=audit_software_fixture(manifest),
        optimizer_config=OptimizerConfig(batch_size=2),
        epochs=1,
        binding=binding,
        batch_loader=_tiny_batches,
        loss_fn=_tiny_loss,
    )
    assert result.actual_exposure_verified is True
    assert result.exposed_record_count == 2
    assert result.software_only is True
