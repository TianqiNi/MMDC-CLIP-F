from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from mmdc_clip_f.backbones import PROMPTS
from mmdc_clip_f.model import MultiViewCLIPClassifier
from mmdc_clip_f.research.view_risk import cache as cache_module
from mmdc_clip_f.research.view_risk.cache import (
    ArtifactIntegrityError,
    ProvenanceMismatchError,
    build_exam_cache_with_role_access,
    load_cache_bundle,
    save_cache_bundle,
)
from mmdc_clip_f.research.view_risk.features import (
    load_verified_frozen_encoder,
    tensor_sha256,
)
from mmdc_clip_f.research.view_risk.fusion import DDSM_FUSION_PAIRS, RSNA_FUSION_PAIRS
from mmdc_clip_f.research.view_risk.perturbations import (
    PerturbationMetadata,
    PerturbationSpec,
    RealizedParent,
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
)
from mmdc_clip_f.research.view_risk.head import (
    RelationAwareConfidenceHead,
    RelationAwareHeadConfig,
)
from mmdc_clip_f.research.view_risk.training import (
    ClassifierProvenance,
    ConfidenceFitBatch,
    ResearchRunConfig,
    TrainingBinding,
    audit_software_fixture,
    build_learned_method,
    build_method_training_schedule,
    confidence_bundle_loss,
    fit_confidence_method_with_role_access,
    module_state_sha256,
)


class TinyVision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.1, 0.9, 768))
        self.calls = 0

    def forward(self, *, pixel_values: Tensor) -> SimpleNamespace:
        self.calls += 1
        level = pixel_values.mean(dim=(1, 2, 3))
        pooled = level[:, None] * self.weight[None]
        return SimpleNamespace(
            pooler_output=pooled,
            last_hidden_state=torch.stack((pooled, pooled + 0.1), dim=1),
        )


class TinyCLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_model = TinyVision()
        self.visual_projection = nn.Linear(768, 5, bias=False)
        self.text = nn.Parameter(torch.arange(1, 21, dtype=torch.float32).reshape(4, 5))
        self.logit_scale = nn.Parameter(torch.tensor(0.2))

    def get_text_features(self, *, input_ids: Tensor) -> Tensor:
        return self.text

    def get_image_features(self, *, pixel_values: Tensor) -> Tensor:
        output = self.vision_model(pixel_values=pixel_values)
        return self.visual_projection(output.pooler_output)


def _manifest(role: Role = Role.CONFIDENCE_FIT) -> RoleManifest:
    views = {
        name: ViewReference(f"image-{index}", f"relative/view-{index}.png", f"{index + 1:064x}")
        for index, name in enumerate(("L_CC", "L_MLO", "R_CC", "R_MLO"))
    }
    record = PrivateExamRecord(
        dataset_namespace="synthetic-cache",
        exam_key="exam-a",
        patient_key="patient-a",
        density=2,
        source_manifest="fixture",
        views=views,
        role=role,
    )
    return RoleManifest(
        dataset_namespace="synthetic-cache",
        source_hashes={"fixture": "a" * 64},
        patient_mapping=PatientMappingDeclaration(
            "synthetic-cache", "b" * 64, "synthetic fixture mapping", "test author"
        ),
        records=(record,),
    )


@pytest.fixture
def cache_bundle(tmp_path):
    clip = TinyCLIP()
    classifier = MultiViewCLIPClassifier(
        clip, ("L_CC", "L_MLO", "R_CC", "R_MLO"), RSNA_FUSION_PAIRS
    )
    checkpoint = tmp_path / "tiny.safetensors"
    save_file(
        {name: value.detach().contiguous() for name, value in classifier.state_dict().items()},
        str(checkpoint),
    )
    encoder = load_verified_frozen_encoder(
        classifier,
        torch.arange(8, dtype=torch.long).reshape(4, 2),
        checkpoint,
        backbone="vit_b_32",
        prompts=PROMPTS,
    )
    spec = PerturbationSpec("gaussian_noise", "mild", realization_seed=37)

    def load_parent(_record: PrivateExamRecord):
        images = {
            name: torch.full((3, 2, 2), 0.15 + index * 0.15)
            for index, name in enumerate(("L_CC", "L_MLO", "R_CC", "R_MLO"))
        }
        return realize_parent(
            images,
            ("L_CC", "R_CC", "R_MLO"),
            perturbations={"R_CC": spec},
        )

    manifest = _manifest()
    bundle = build_exam_cache_with_role_access(
        manifest,
        operation=Operation.CONFIDENCE_FITTING,
        role=Role.CONFIDENCE_FIT,
        exam_key="exam-a",
        sample_key="sample-a",
        parent_loader=load_parent,
        encoder=encoder,
        implementation_revision="synthetic-revision",
    )
    return bundle, encoder, manifest


def _load_cache(path, expected, manifest):
    return load_cache_bundle(
        path,
        expected_provenance=expected,
        manifest=manifest,
        operation=Operation.CONFIDENCE_FITTING,
        role=Role.CONFIDENCE_FIT,
    )


def test_candidate_optimizer_uses_checked_current_cache_and_keeps_classifier_frozen(
    cache_bundle,
) -> None:
    bundle, encoder, _manifest_value = cache_bundle
    model = RelationAwareConfidenceHead(RelationAwareHeadConfig(backbone="vit_b_32"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    before_classifier = module_state_sha256(encoder.classifier)
    before_head = module_state_sha256(model)

    optimizer.zero_grad(set_to_none=True)
    loss = confidence_bundle_loss(model, "candidate", bundle)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert module_state_sha256(model) != before_head
    assert module_state_sha256(encoder.classifier) == before_classifier


def test_role_bound_confidence_fit_uses_exact_schedule_and_verified_fresh_encoder(
    tmp_path,
) -> None:
    manifest = _manifest(Role.CONFIDENCE_FIT)
    schedule = build_method_training_schedule(
        manifest.records, epoch=1, seed=42, method="candidate"
    )
    draw = schedule[0].draw
    clip = TinyCLIP()
    frozen_classifier = MultiViewCLIPClassifier(
        clip, ("L_CC", "L_MLO", "R_CC", "R_MLO"), RSNA_FUSION_PAIRS
    )
    checkpoint = tmp_path / "scheduled-classifier.safetensors"
    save_file(
        {
            name: value.detach().contiguous()
            for name, value in frozen_classifier.state_dict().items()
        },
        str(checkpoint),
    )
    encoder = load_verified_frozen_encoder(
        frozen_classifier,
        torch.arange(8, dtype=torch.long).reshape(4, 2),
        checkpoint,
        backbone="vit_b_32",
        prompts=PROMPTS,
    )
    images = {
        view: torch.full((3, 2, 2), 0.15 + index * 0.15)
        for index, view in enumerate(("L_CC", "L_MLO", "R_CC", "R_MLO"))
    }
    bundle = build_exam_cache_with_role_access(
        manifest,
        operation=Operation.CONFIDENCE_FITTING,
        role=Role.CONFIDENCE_FIT,
        exam_key="exam-a",
        sample_key="scheduled-sample-a",
        parent_loader=lambda _record: realize_training_parent(images, draw),
        encoder=encoder,
        implementation_revision="synthetic-scheduled-revision",
    )
    classifier_fit_manifest = _manifest(Role.CLASSIFIER_FIT)
    initialization = ClassifierProvenance.synthetic_injected(
        "vit_b_32", encoder.identity.checkpoint_sha256
    )
    classifier = ClassifierProvenance.fresh_selected(
        initialization,
        checkpoint_sha256=encoder.identity.checkpoint_sha256,
        classifier_fit_manifest_sha256=classifier_fit_manifest.manifest_sha256,
        classifier_fit_update_count=1,
        tune_selection_sha256="e" * 64,
        tune_manifest_sha256="f" * 64,
        readiness=audit_software_fixture(classifier_fit_manifest),
    )
    config = ResearchRunConfig.default("RSNA", "vit_b_32")
    binding = TrainingBinding(
        protocol_sha256=config.protocol_sha256,
        config_sha256=config.sha256,
        search_table_sha256=config.search_table.sha256,
        manifest_sha256=manifest.manifest_sha256,
        classifier_checkpoint_sha256=classifier.checkpoint_sha256,
        method="candidate",
        seed=42,
    )
    model = build_learned_method("candidate", "vit_b_32", RSNA_FUSION_PAIRS)
    head_before = module_state_sha256(model)
    classifier_before = module_state_sha256(encoder.classifier)

    def batches(_records, supplied_schedule, batch_size, epoch):
        assert supplied_schedule == schedule
        assert batch_size == 6
        assert epoch == 1
        yield ConfidenceFitBatch((bundle,))

    result = fit_confidence_method_with_role_access(
        manifest=manifest,
        model=model,
        method="candidate",
        classifier=classifier,
        config=config,
        seed=42,
        binding=binding,
        batch_loader=batches,
        frozen_encoder=encoder,
        stop_after_epoch=1,
    )

    assert result.actual_exposure_verified is True
    assert result.update_count == 1
    assert result.exposed_record_count == 1
    assert result.selection_trial_budget == 20
    assert module_state_sha256(model) != head_before
    assert module_state_sha256(encoder.classifier) == classifier_before


def test_confidence_fit_rejects_diagnostic_classifier_before_cache_reader() -> None:
    manifest = _manifest(Role.CONFIDENCE_FIT)
    config = ResearchRunConfig.default("RSNA", "vit_b_32")
    diagnostic = ClassifierProvenance.diagnostic_original(
        "vit_b_32", "d" * 64, reason="overlapping historical training exposure"
    )
    binding = TrainingBinding(
        protocol_sha256=config.protocol_sha256,
        config_sha256=config.sha256,
        search_table_sha256=config.search_table.sha256,
        manifest_sha256=manifest.manifest_sha256,
        classifier_checkpoint_sha256=diagnostic.checkpoint_sha256,
        method="candidate",
        seed=42,
    )
    called = False

    def batches(*_args):
        nonlocal called
        called = True
        raise AssertionError("cache reader must not run")

    with pytest.raises(PermissionError, match="fresh public classifier"):
        fit_confidence_method_with_role_access(
            manifest=manifest,
            model=build_learned_method("candidate", "vit_b_32", RSNA_FUSION_PAIRS),
            method="candidate",
            classifier=diagnostic,
            config=config,
            seed=42,
            binding=binding,
            batch_loader=batches,
            stop_after_epoch=1,
        )
    assert called is False


def test_safe_roundtrip_keeps_exact_targets_and_required_provenance(cache_bundle, tmp_path) -> None:
    bundle, _, manifest = cache_bundle
    paths = save_cache_bundle(bundle, tmp_path / "external-cache.json")
    loaded = _load_cache(paths.metadata, bundle.provenance, manifest)

    assert loaded.provenance == bundle.provenance
    assert loaded.provenance.sample_keys == ("sample-a",)
    assert loaded.provenance.exam_keys == ("exam-a",)
    assert loaded.provenance.patient_keys == ("patient-a",)
    assert loaded.provenance.observed_mask == (True, False, True, True)
    assert loaded.provenance.perturbations["L_CC"]["spec"] == {
        "family": "clean",
        "severity": None,
        "variant": None,
        "realization_seed": 0,
    }
    assert loaded.provenance.perturbations["R_CC"]["spec"] == {
        "family": "gaussian_noise",
        "severity": "mild",
        "variant": None,
        "realization_seed": 37,
    }
    assert torch.equal(loaded.targets.labels, torch.tensor([2]))
    assert torch.equal(loaded.targets.omission_effects, bundle.targets.omission_effects.cpu())
    assert torch.equal(loaded.targets.omission_labels, bundle.targets.omission_labels.cpu())
    assert torch.equal(loaded.targets.scores, loaded.features.scores)
    assert torch.equal(
        loaded.targets.tcp,
        loaded.targets.probabilities.gather(1, loaded.targets.labels[:, None]).squeeze(1),
    )
    document = json.loads(paths.metadata.read_text())
    assert document["feature_content_sha256"]
    assert document["labels_sha256"]
    assert document["record_integrity_sha256"]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("dataset_namespace", "different"),
        ("role", Role.TUNE.value),
        ("operation", Operation.TUNE_SELECTION.value),
        ("manifest_sha256", "c" * 64),
        ("sample_keys", ("other",)),
        ("exam_keys", ("other",)),
        ("patient_keys", ("other",)),
        ("fusion_pairs", DDSM_FUSION_PAIRS),
        ("perturbations", {"L_CC": None, "R_CC": None, "R_MLO": None}),
        ("parent_identity_sha256", "e" * 64),
        ("checkpoint_sha256", "d" * 64),
        ("backbone", "other-backbone"),
        ("hf_model", "other/model"),
        ("backbone_revision", "other-revision"),
        ("image_size", 336),
        ("hidden_size", 1024),
        ("preprocessing", "other-preprocessing"),
        ("image_mean", (0.1, 0.2, 0.3)),
        ("image_std", (0.9, 0.8, 0.7)),
        ("prompt_order", tuple(reversed(PROMPTS))),
        ("text_input_sha256", "f" * 64),
        ("implementation_revision", "other-revision"),
    ),
)
def test_every_expected_provenance_field_is_enforced(
    cache_bundle, tmp_path, field: str, value
) -> None:
    bundle, _, manifest = cache_bundle
    paths = save_cache_bundle(bundle, tmp_path / "bound.json")
    expected = replace(bundle.provenance, **{field: value})
    with pytest.raises(ProvenanceMismatchError, match=field):
        _load_cache(paths.metadata, expected, manifest)


def _rehash_document(metadata_path, tensor_path) -> None:
    document = json.loads(metadata_path.read_text())
    tensors = load_file(str(tensor_path))
    document["tensor_file_sha256"] = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    document["tensor_manifest"] = {
        name: {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": tensor_sha256(tensor),
        }
        for name, tensor in sorted(tensors.items())
    }
    feature_records = [
        [name, tensor_sha256(tensors[name])]
        for name in sorted(tensors)
        if name.startswith("feature.")
    ]
    document["feature_content_sha256"] = hashlib.sha256(
        json.dumps(feature_records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if "target.labels" in tensors:
        document["labels_sha256"] = tensor_sha256(tensors["target.labels"])
    unsigned = dict(document)
    unsigned.pop("record_integrity_sha256", None)
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    document["record_integrity_sha256"] = hashlib.sha256(canonical).hexdigest()
    metadata_path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n")


def test_tensor_and_metadata_tampering_are_rejected(cache_bundle, tmp_path) -> None:
    bundle, _, manifest = cache_bundle
    paths = save_cache_bundle(bundle, tmp_path / "tamper.json")
    data = bytearray(paths.tensors.read_bytes())
    data[-1] ^= 1
    paths.tensors.write_bytes(data)
    with pytest.raises(ArtifactIntegrityError, match="tensor file"):
        _load_cache(paths.metadata, bundle.provenance, manifest)

    paths = save_cache_bundle(bundle, tmp_path / "metadata.json")
    document = json.loads(paths.metadata.read_text())
    document["provenance"]["role"] = Role.TUNE.value
    paths.metadata.write_text(json.dumps(document))
    with pytest.raises(ArtifactIntegrityError, match="metadata"):
        _load_cache(paths.metadata, bundle.provenance, manifest)


def test_rehashed_stale_targets_still_fail_semantic_validation(cache_bundle, tmp_path) -> None:
    bundle, _, manifest = cache_bundle
    paths = save_cache_bundle(bundle, tmp_path / "stale.json")
    tensors = load_file(str(paths.tensors))
    tensors["target.observed_error"] = 1 - tensors["target.observed_error"]
    save_file(tensors, str(paths.tensors))
    _rehash_document(paths.metadata, paths.tensors)

    with pytest.raises(ArtifactIntegrityError, match="observed_error"):
        _load_cache(paths.metadata, bundle.provenance, manifest)


def test_rehashed_self_consistent_wrong_label_is_rejected_by_authorized_manifest(
    cache_bundle, tmp_path
) -> None:
    from mmdc_clip_f.research.view_risk.targets import build_intervention_targets

    bundle, _, manifest = cache_bundle
    paths = save_cache_bundle(bundle, tmp_path / "wrong-label.json")
    tensors = load_file(str(paths.tensors))
    labels = torch.tensor([3], dtype=torch.long)
    logits = {
        view: tensors[f"feature.logits.{view}"] for view in bundle.provenance.observed_views
    }
    generated = build_intervention_targets(
        logits,
        labels,
        bundle.provenance.observed_views,
        fusion_pairs=bundle.provenance.fusion_pairs,
    )
    tensors["target.labels"] = labels
    tensors["target.observed_prediction"] = generated.observed_prediction
    tensors["target.observed_error"] = generated.observed_error
    tensors["target.omission_predictions"] = generated.omission_predictions
    tensors["target.omission_effects"] = generated.omission_effects
    tensors["target.omission_labels"] = generated.omission_labels
    tensors["target.valid_removal_mask"] = generated.valid_removal_mask
    tensors["target.tcp"] = tensors["target.probabilities"].gather(1, labels[:, None]).squeeze(1)
    save_file(tensors, str(paths.tensors))
    _rehash_document(paths.metadata, paths.tensors)

    with pytest.raises(ArtifactIntegrityError, match="authorized manifest densities"):
        _load_cache(paths.metadata, bundle.provenance, manifest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("observed_mask", [True, True, True, True], "observed_mask"),
        ("fusion_pairs", [list(pair) for pair in DDSM_FUSION_PAIRS], "fusion_pairs"),
        ("schema_version", "view-risk-frozen-cache/v0", "schema"),
        ("target_version", "view-risk-intervention-target/v0", "target"),
    ),
)
def test_rehashed_mask_tree_and_version_metadata_are_rejected(
    cache_bundle, tmp_path, field, value, message
) -> None:
    bundle, _, manifest = cache_bundle
    paths = save_cache_bundle(bundle, tmp_path / f"structural-{field}.json")
    document = json.loads(paths.metadata.read_text())
    document["provenance"][field] = value
    paths.metadata.write_text(json.dumps(document))
    _rehash_document(paths.metadata, paths.tensors)
    with pytest.raises((ArtifactIntegrityError, ProvenanceMismatchError), match=message):
        _load_cache(paths.metadata, bundle.provenance, manifest)


def test_unignored_worktree_destination_is_refused(cache_bundle) -> None:
    bundle, _, _ = cache_bundle
    destination = Path.cwd() / "private-cache-metadata.json"
    assert not destination.exists()
    with pytest.raises(ValueError, match="gitignore"):
        save_cache_bundle(bundle, destination)
    assert not destination.exists()


def test_ignored_metadata_with_unignored_tensor_sibling_is_refused_before_write(
    cache_bundle, tmp_path
) -> None:
    bundle, _, _ = cache_bundle
    repository = tmp_path / "private-cache-repository"
    repository.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    (repository / ".gitignore").write_text("artifact.json\n", encoding="utf-8")
    metadata = repository / "artifact.json"
    tensors = repository / "artifact.safetensors"

    with pytest.raises(ValueError, match="gitignore"):
        save_cache_bundle(bundle, metadata)

    assert not metadata.exists()
    assert not tensors.exists()


def test_exact_filename_ignore_rules_that_omit_staging_are_refused_before_write(
    cache_bundle, tmp_path
) -> None:
    bundle, _, _ = cache_bundle
    repository = tmp_path / "ignored-cache-repository"
    repository.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    (repository / ".gitignore").write_text(
        "artifact.json\nartifact.safetensors\n", encoding="utf-8"
    )
    metadata = repository / "artifact.json"

    with pytest.raises(ValueError, match="gitignore"):
        save_cache_bundle(bundle, metadata)

    assert not list(repository.glob("*artifact*"))


def test_cache_in_fully_ignored_directory_covers_final_and_staging_paths(
    cache_bundle, tmp_path
) -> None:
    bundle, _, _ = cache_bundle
    repository = tmp_path / "ignored-cache-repository"
    repository.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    (repository / ".gitignore").write_text("private-cache/\n", encoding="utf-8")
    destination = repository / "private-cache"
    destination.mkdir()

    paths = save_cache_bundle(bundle, destination / "artifact.json")

    assert paths.metadata.exists()
    assert paths.tensors.exists()


def _save_in_child_with_abrupt_replacement(
    bundle, metadata: Path, *, replacement_number: int
) -> int:
    child = os.fork()
    if child == 0:
        original_replace = os.replace
        replacements = 0

        def interrupt(source, destination) -> None:
            nonlocal replacements
            replacements += 1
            if replacements == replacement_number:
                os._exit(88)
            original_replace(source, destination)

        cache_module.os.replace = interrupt
        try:
            save_cache_bundle(bundle, metadata)
        except ValueError as exc:
            os._exit(89 if "gitignore" in str(exc) else 90)
        except BaseException:
            os._exit(91)
        os._exit(92)
    _, status = os.waitpid(child, 0)
    assert os.WIFEXITED(status)
    return os.WEXITSTATUS(status)


def _git_ignored(repository: Path, path: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repository), "check-ignore", "--quiet", "--", str(path)],
        check=False,
    )
    return result.returncode == 0


@pytest.mark.parametrize("replacement_number", (1, 2), ids=("tensor", "metadata"))
def test_atomic_replacement_cannot_strand_unignored_private_staging_files(
    cache_bundle, tmp_path, replacement_number
) -> None:
    bundle, _, _ = cache_bundle

    exact_repository = tmp_path / f"exact-ignore-{replacement_number}"
    exact_repository.mkdir()
    subprocess.run(["git", "init", "--quiet", str(exact_repository)], check=True)
    exact_ignore = "artifact.json\nartifact.safetensors\n"
    if replacement_number == 2:
        exact_ignore += ".artifact.safetensors.*.tmp\n"
    (exact_repository / ".gitignore").write_text(exact_ignore, encoding="utf-8")
    exact_metadata = exact_repository / "artifact.json"

    assert (
        _save_in_child_with_abrupt_replacement(
            bundle, exact_metadata, replacement_number=replacement_number
        )
        == 89
    )
    assert not list(exact_repository.glob("*artifact*"))

    ignored_repository = tmp_path / f"directory-ignore-{replacement_number}"
    ignored_repository.mkdir()
    subprocess.run(["git", "init", "--quiet", str(ignored_repository)], check=True)
    (ignored_repository / ".gitignore").write_text("private-cache/\n", encoding="utf-8")
    ignored_destination = ignored_repository / "private-cache"
    ignored_destination.mkdir()
    ignored_metadata = ignored_destination / "artifact.json"

    assert (
        _save_in_child_with_abrupt_replacement(
            bundle, ignored_metadata, replacement_number=replacement_number
        )
        == 88
    )
    survivors = list(ignored_destination.iterdir())
    assert survivors
    assert all(_git_ignored(ignored_repository, path) for path in survivors)


def test_reordered_private_keys_and_tree_or_mask_metadata_fail_even_if_rehashed(
    cache_bundle, tmp_path
) -> None:
    bundle, _, manifest = cache_bundle
    paths = save_cache_bundle(bundle, tmp_path / "order.json")
    document = json.loads(paths.metadata.read_text())
    document["provenance"]["sample_keys"] = ["second", "sample-a"]
    document["provenance"]["exam_keys"] = ["second", "exam-a"]
    document["provenance"]["patient_keys"] = ["second", "patient-a"]
    unsigned = dict(document)
    unsigned.pop("record_integrity_sha256")
    document["record_integrity_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    paths.metadata.write_text(json.dumps(document))
    with pytest.raises((ArtifactIntegrityError, ProvenanceMismatchError), match="keys|batch"):
        _load_cache(paths.metadata, bundle.provenance, manifest)


def test_missing_view_tensor_and_nonfinite_feature_are_rejected(cache_bundle, tmp_path) -> None:
    bundle, _, manifest = cache_bundle
    for case in ("missing", "nonfinite"):
        paths = save_cache_bundle(bundle, tmp_path / f"{case}.json")
        tensors = load_file(str(paths.tensors))
        if case == "missing":
            tensors.pop("feature.hidden.R_CC")
        else:
            tensors["feature.hidden.R_CC"][0, 0, 0] = float("nan")
        save_file(tensors, str(paths.tensors))
        _rehash_document(paths.metadata, paths.tensors)
        with pytest.raises(ArtifactIntegrityError, match="tensor|finite|feature"):
            _load_cache(paths.metadata, bundle.provenance, manifest)


def test_held_out_fit_stress_is_refused_before_encoding(cache_bundle) -> None:
    _, encoder, manifest = cache_bundle
    calls = encoder.classifier.clip_model.vision_model.calls

    def held_out_parent(_record):
        images = {name: torch.full((3, 2, 2), 0.5) for name in ("L_CC", "L_MLO", "R_CC", "R_MLO")}
        return realize_parent(
            images,
            ("L_CC", "R_CC"),
            perturbations={"L_CC": PerturbationSpec("contrast", "mild", realization_seed=9)},
        )

    with pytest.raises(ValueError, match="held out from confidence-fit"):
        build_exam_cache_with_role_access(
            manifest,
            operation=Operation.CONFIDENCE_FITTING,
            role=Role.CONFIDENCE_FIT,
            exam_key="exam-a",
            sample_key="sample-a",
            parent_loader=held_out_parent,
            encoder=encoder,
            implementation_revision="synthetic-revision",
        )
    assert encoder.classifier.clip_model.vision_model.calls == calls


def test_realized_severity_parameters_cannot_disguise_strong_noise_as_mild(
    cache_bundle,
) -> None:
    _, encoder, manifest = cache_bundle
    calls = encoder.classifier.clip_model.vision_model.calls

    def disguised_parent(_record):
        spec = PerturbationSpec("gaussian_noise", "mild", realization_seed=9)
        return RealizedParent(
            images={"L_CC": torch.full((3, 8, 8), 0.5)},
            metadata={"L_CC": PerturbationMetadata(spec, (("sigma", 0.1),))},
        )

    with pytest.raises(ValueError, match="realized perturbation parameters"):
        build_exam_cache_with_role_access(
            manifest,
            operation=Operation.CONFIDENCE_FITTING,
            role=Role.CONFIDENCE_FIT,
            exam_key="exam-a",
            sample_key="sample-a",
            parent_loader=disguised_parent,
            encoder=encoder,
            implementation_revision="synthetic-revision",
        )
    assert encoder.classifier.clip_model.vision_model.calls == calls


@pytest.mark.parametrize(
    ("spec", "parameters"),
    (
        (
            PerturbationSpec("gaussian_noise", "mild", realization_seed=9),
            (("sigma", 0.02), ("unrecorded_gain", 2.0)),
        ),
        (
            PerturbationSpec("crop", "mild", realization_seed=9),
            (("retained_area", 0.9), ("crop_side", 7), ("top", 0)),
        ),
        (
            PerturbationSpec("crop", "mild", realization_seed=9),
            (("retained_area", 0.9), ("crop_side", 7), ("top", 2), ("left", 0)),
        ),
    ),
)
def test_realized_parameters_reject_unknown_missing_or_out_of_bounds_fields(
    cache_bundle, spec, parameters
) -> None:
    _, encoder, _ = cache_bundle
    manifest = _manifest(Role.PILOT)
    calls = encoder.classifier.clip_model.vision_model.calls

    def invalid_parent(_record):
        return RealizedParent(
            images={"L_CC": torch.full((3, 8, 8), 0.5)},
            metadata={"L_CC": PerturbationMetadata(spec, parameters)},
        )

    with pytest.raises(ValueError, match="realized perturbation parameters"):
        build_exam_cache_with_role_access(
            manifest,
            operation=Operation.PILOT_EVALUATION,
            role=Role.PILOT,
            exam_key="exam-a",
            sample_key="sample-a",
            parent_loader=invalid_parent,
            encoder=encoder,
            implementation_revision="synthetic-revision",
        )
    assert encoder.classifier.clip_model.vision_model.calls == calls


def test_forbidden_cache_load_role_is_checked_before_file_access(cache_bundle, tmp_path) -> None:
    bundle, _, _ = cache_bundle
    with pytest.raises(PermissionError):
        load_cache_bundle(
            tmp_path / "does-not-exist.json",
            expected_provenance=bundle.provenance,
            manifest=_manifest(Role.PILOT),
            operation=Operation.CONFIDENCE_FITTING,
            role=Role.PILOT,
        )


def test_forbidden_role_is_refused_before_parent_loader_or_encoder(tmp_path) -> None:
    manifest = _manifest(Role.PILOT)
    callback_calls = 0
    model_calls = 0

    class ExplodingEncoder:
        def extract_realized_parent(self, _parent):
            nonlocal model_calls
            model_calls += 1
            raise AssertionError("must not encode")

    def parent_loader(_record):
        nonlocal callback_calls
        callback_calls += 1
        raise AssertionError("must not load")

    with pytest.raises(PermissionError):
        build_exam_cache_with_role_access(
            manifest,
            operation=Operation.CONFIDENCE_FITTING,
            role=Role.PILOT,
            exam_key="exam-a",
            sample_key="sample-a",
            parent_loader=parent_loader,
            encoder=ExplodingEncoder(),
            implementation_revision="synthetic-revision",
        )
    assert callback_calls == 0
    assert model_calls == 0


def test_cache_builder_does_not_accept_caller_supplied_targets(cache_bundle) -> None:
    import inspect

    signature = inspect.signature(build_exam_cache_with_role_access)
    assert "targets" not in signature.parameters
    assert "labels" not in signature.parameters
