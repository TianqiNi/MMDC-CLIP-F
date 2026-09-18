"""Persisted production classifier evidence and role-bound image adapters."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F

from mmdc_clip_f.data import build_transforms
from mmdc_clip_f.provenance import sha256_file

from .cache import (
    _require_external_or_ignored_destination,
    verify_cache_file_binding,
)
from .fusion import DDSM_FUSION_PAIRS, RSNA_FUSION_PAIRS
from .inputs import CANONICAL_VIEWS
from .roles import Operation, PrivateExamRecord, Role, RoleManifest, run_with_role_access
from .training import (
    ClassifierProvenance,
    LEARNED_TORCH_METHODS,
    OptimizerConfig,
    ReadinessAudit,
    ResearchRunConfig,
    RoleBoundBatch,
    TrainingBinding,
    TrainingResult,
    fit_fresh_classifier_with_role_access,
    load_pinned_public_clip_classifier,
    load_verified_readiness_audit,
    module_state_sha256,
    pinned_public_clip_configuration,
    state_dict_sha256,
)
from .features import (
    FrozenEncoderIdentity,
    frozen_encoder_identity_from_inputs,
    tensor_sha256,
)


PUBLIC_INITIALIZATION_VERSION = "view-risk-public-initialization/v1"
SELECTED_CLASSIFIER_VERSION = "view-risk-selected-classifier/v2"
CLASSIFIER_FIT_VERSION = "view-risk-classifier-fit/v1"
CONFIDENCE_FIT_VERSION = "view-risk-confidence-fit/v3"


def _require_config_dataset_namespace(
    config: ResearchRunConfig, namespace: str
) -> None:
    aliases = {
        "RSNA": "RSNA",
        "RSNA-SMBC": "RSNA",
        "DDSM": "DDSM",
        "MINI-DDSM": "DDSM",
    }
    if aliases.get(namespace.upper()) != config.dataset:
        raise ValueError("role manifest dataset namespace disagrees with run configuration")


def _require_initialization_config(
    config: ResearchRunConfig, initialization: PublicInitializationArtifact
) -> None:
    if (
        initialization.dataset != config.dataset
        or initialization.classifier.backbone != config.backbone
    ):
        raise ValueError("public initialization dataset/backbone disagrees with run configuration")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _atomic_json(path: Path, document: Mapping[str, object]) -> None:
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    for candidate in (path, staging):
        _require_external_or_ignored_destination(candidate)
    try:
        with staging.open("xb") as handle:
            handle.write(_canonical_json(document) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(staging, path)
        path.chmod(0o600)
    finally:
        staging.unlink(missing_ok=True)


def _atomic_safetensors(path: Path, tensors: Mapping[str, Tensor]) -> None:
    from safetensors.torch import save_file

    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    for candidate in (path, staging):
        _require_external_or_ignored_destination(candidate)
    try:
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in tensors.items()},
            str(staging),
        )
        os.link(staging, path)
        path.chmod(0o600)
    finally:
        staging.unlink(missing_ok=True)


def save_tensor_checkpoint(model: nn.Module, path: str | Path) -> Path:
    """Exclusively save tensor-only model state to an external/git-ignored path."""

    target = Path(path).resolve()
    if target.suffix != ".safetensors" or not target.parent.exists():
        raise ValueError("model checkpoint must be safetensors in an existing directory")
    _atomic_safetensors(target, model.state_dict())
    return target


def load_tensor_checkpoint_state(path: str | Path) -> Mapping[str, Tensor]:
    from safetensors.torch import load_file

    source = Path(path).resolve()
    _require_external_or_ignored_destination(source)
    try:
        return load_file(str(source), device="cpu")
    except Exception as exc:
        raise ValueError("tensor checkpoint is unavailable or invalid") from exc


@dataclass(frozen=True)
class PublicInitializationArtifact:
    dataset: str
    classifier: ClassifierProvenance
    weights_path: Path
    input_ids_path: Path
    sha256: str
    source_path: Path


def _initialization_payload(
    *,
    dataset: str,
    provenance: ClassifierProvenance,
    weights_path: Path,
    input_ids_path: Path,
) -> dict[str, object]:
    weights = load_tensor_checkpoint_state(weights_path)
    tokens = load_tensor_checkpoint_state(input_ids_path)
    if set(tokens) != {"input_ids"}:
        raise ValueError("public initialization token artifact is invalid")
    weight_state_sha256 = state_dict_sha256(weights)
    input_ids_sha256 = tensor_sha256(tokens["input_ids"])
    pairs = RSNA_FUSION_PAIRS if dataset == "RSNA" else DDSM_FUSION_PAIRS
    source_identity = _sha256_json(
        {
            "hf_model": provenance.hf_model,
            "revision": provenance.revision,
            "weight_content_sha256": weight_state_sha256,
            "token_ids_sha256": input_ids_sha256,
            "prompts": list(provenance.prompts),
            "fusion_pairs": [list(pair) for pair in pairs],
        }
    )
    if (
        provenance.kind != "public_pretrained_fresh"
        or provenance.workflow_complete
        or provenance.checkpoint_sha256 != weight_state_sha256
        or provenance.public_weight_content_sha256 != weight_state_sha256
        or provenance.initialization_sha256 != source_identity
    ):
        raise ValueError("public initialization provenance disagrees with persisted tensor content")
    return {
        "schema_version": PUBLIC_INITIALIZATION_VERSION,
        "dataset": dataset,
        "classifier": provenance.to_dict(),
        "weights_path": str(weights_path),
        "weights_file_sha256": sha256_file(weights_path),
        "weights_state_sha256": weight_state_sha256,
        "input_ids_path": str(input_ids_path),
        "input_ids_file_sha256": sha256_file(input_ids_path),
        "input_ids_sha256": input_ids_sha256,
        "fusion_pairs": [list(pair) for pair in pairs],
    }


def save_public_initialization_artifact(
    path: str | Path,
    *,
    model: nn.Module,
    input_ids: Tensor,
    provenance: ClassifierProvenance,
    dataset: str,
) -> PublicInitializationArtifact:
    """Persist the exact loaded public state, prompt tokens, pins, and fusion tree."""

    target = Path(path).resolve()
    normalized = dataset.upper()
    if normalized not in ("RSNA", "DDSM") or target.suffix != ".json" or not target.parent.exists():
        raise ValueError("public initialization artifact path or dataset is invalid")
    weights_path = target.with_name(f"{target.stem}.weights.safetensors")
    input_ids_path = target.with_name(f"{target.stem}.input-ids.safetensors")
    if module_state_sha256(model) != provenance.checkpoint_sha256:
        raise ValueError("loaded public model state disagrees with provenance")
    _atomic_safetensors(weights_path, model.state_dict())
    try:
        _atomic_safetensors(input_ids_path, {"input_ids": input_ids})
        payload = _initialization_payload(
            dataset=normalized,
            provenance=provenance,
            weights_path=weights_path,
            input_ids_path=input_ids_path,
        )
        digest = _sha256_json(payload)
        _atomic_json(target, {"artifact_sha256": digest, "artifact": payload})
    except BaseException:
        weights_path.unlink(missing_ok=True)
        input_ids_path.unlink(missing_ok=True)
        raise
    return PublicInitializationArtifact(
        normalized, provenance, weights_path, input_ids_path, digest, target
    )


def _classifier_from_verified_initialization(
    payload: Mapping[str, object], *, source_path: Path
) -> PublicInitializationArtifact:
    expected = {
        "schema_version",
        "dataset",
        "classifier",
        "weights_path",
        "weights_file_sha256",
        "weights_state_sha256",
        "input_ids_path",
        "input_ids_file_sha256",
        "input_ids_sha256",
        "fusion_pairs",
    }
    if set(payload) != expected or payload["schema_version"] != PUBLIC_INITIALIZATION_VERSION:
        raise ValueError("public initialization artifact schema is invalid")
    dataset = payload["dataset"]
    if dataset not in ("RSNA", "DDSM"):
        raise ValueError("public initialization dataset is invalid")
    raw = payload["classifier"]
    if not isinstance(raw, Mapping):
        raise ValueError("public initialization classifier record is invalid")
    public = pinned_public_clip_configuration(str(raw.get("backbone")))
    weights_path = Path(str(payload["weights_path"])).resolve()
    input_ids_path = Path(str(payload["input_ids_path"])).resolve()
    weights = load_tensor_checkpoint_state(weights_path)
    tokens = load_tensor_checkpoint_state(input_ids_path)
    if set(tokens) != {"input_ids"}:
        raise ValueError("public initialization token artifact is invalid")
    weight_state = state_dict_sha256(weights)
    token_state = tensor_sha256(tokens["input_ids"])
    pairs = RSNA_FUSION_PAIRS if dataset == "RSNA" else DDSM_FUSION_PAIRS
    source_identity = _sha256_json(
        {
            "hf_model": public.hf_model,
            "revision": public.revision,
            "weight_content_sha256": weight_state,
            "token_ids_sha256": token_state,
            "prompts": list(public.prompts),
            "fusion_pairs": [list(pair) for pair in pairs],
        }
    )
    checks = (
        sha256_file(weights_path) == payload["weights_file_sha256"],
        sha256_file(input_ids_path) == payload["input_ids_file_sha256"],
        weight_state == payload["weights_state_sha256"],
        token_state == payload["input_ids_sha256"],
        payload["fusion_pairs"] == [list(pair) for pair in pairs],
    )
    if not all(checks):
        raise ValueError("public initialization tensor content or fusion binding changed")
    provenance = ClassifierProvenance._public_pretrained(
        public.backbone,
        weight_state,
        weight_content_sha256=weight_state,
        source_identity_sha256=source_identity,
    )
    if dict(raw) != provenance.to_dict():
        raise ValueError("public initialization provenance record is forged or stale")
    return PublicInitializationArtifact(
        dataset, provenance, weights_path, input_ids_path, "", source_path
    )


def load_public_initialization_artifact(path: str | Path) -> PublicInitializationArtifact:
    source = Path(path).resolve()
    _require_external_or_ignored_destination(source)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
        if set(document) != {"artifact_sha256", "artifact"} or not isinstance(
            document["artifact"], Mapping
        ):
            raise ValueError
        digest = _sha256_json(document["artifact"])
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("public initialization artifact is unavailable or invalid") from exc
    if document["artifact_sha256"] != digest:
        raise ValueError("public initialization artifact integrity check failed")
    result = _classifier_from_verified_initialization(document["artifact"], source_path=source)
    return PublicInitializationArtifact(
        result.dataset,
        result.classifier,
        result.weights_path,
        result.input_ids_path,
        digest,
        source,
    )


@dataclass(frozen=True)
class SelectedClassifierArtifact:
    classifier: ClassifierProvenance
    initialization: PublicInitializationArtifact
    readiness: ReadinessAudit
    selected_checkpoint_path: Path
    encoder_identity: FrozenEncoderIdentity
    sha256: str
    source_path: Path
    config_sha256: str


@dataclass(frozen=True)
class ClassifierFitArtifact:
    initialization: PublicInitializationArtifact
    readiness: ReadinessAudit
    binding: TrainingBinding
    result: TrainingResult
    checkpoint_paths: tuple[tuple[int, Path], ...]
    sha256: str
    source_path: Path
    config: ResearchRunConfig


@dataclass(frozen=True)
class ConfidenceFitArtifact:
    classifier: ClassifierProvenance
    encoder_identity: FrozenEncoderIdentity | None
    binding: TrainingBinding
    result: TrainingResult
    checkpoint_paths: tuple[tuple[int, Path], ...]
    sha256: str
    source_path: Path
    config: ResearchRunConfig


def _selected_encoder_identity(
    initialization: PublicInitializationArtifact, checkpoint_path: str | Path
) -> FrozenEncoderIdentity:
    """Derive selected identity from current verified token and checkpoint bytes."""

    current = load_public_initialization_artifact(initialization.source_path)
    tokens = load_tensor_checkpoint_state(current.input_ids_path)
    if set(tokens) != {"input_ids"}:
        raise ValueError("public initialization token artifact is invalid")
    checkpoint = Path(checkpoint_path).resolve()
    load_tensor_checkpoint_state(checkpoint)
    return frozen_encoder_identity_from_inputs(
        checkpoint_sha256=sha256_file(checkpoint),
        backbone=current.classifier.backbone,
        prompts=current.classifier.prompts,
        input_ids=tokens["input_ids"],
        fusion_pairs=(
            RSNA_FUSION_PAIRS
            if current.dataset == "RSNA"
            else DDSM_FUSION_PAIRS
        ),
    )


def _verify_cache_file_binding_bytes(binding: object) -> None:
    """Rehash bound cache files without opening private metadata before role access."""

    expected = {"metadata_path", "metadata_sha256", "tensors_path", "tensors_sha256"}
    if not isinstance(binding, Mapping) or set(binding) != expected:
        raise ValueError("confidence cache-file binding schema is invalid")
    for path_field, digest_field in (
        ("metadata_path", "metadata_sha256"),
        ("tensors_path", "tensors_sha256"),
    ):
        source = Path(str(binding[path_field])).resolve()
        _require_external_or_ignored_destination(source)
        if sha256_file(source) != binding[digest_field]:
            raise ValueError("confidence cache-file evidence changed")


def _confidence_cache_index_paths(
    path: Path, classifier: ClassifierProvenance
) -> tuple[Path, ...]:
    """Read only the already-authorized training index and return exact cache paths."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("confidence cache index is unavailable or invalid") from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema_version", "classifier", "entries"}
        or value["schema_version"] != "view-risk-training-cache-index/v1"
        or value["classifier"] != classifier.to_dict()
        or not isinstance(value["entries"], list)
    ):
        raise ValueError("confidence cache index workflow binding is invalid")
    result = []
    for entry in value["entries"]:
        if not isinstance(entry, Mapping) or set(entry) != {
            "epoch", "metadata_path", "provenance"
        }:
            raise ValueError("confidence cache index entry schema is invalid")
        raw_path = entry["metadata_path"]
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise ValueError("confidence cache index path is invalid")
        result.append(Path(raw_path).resolve())
    if len(set(result)) != len(result):
        raise ValueError("confidence cache index contains duplicate paths")
    return tuple(result)


def save_confidence_fit_artifact(
    path: str | Path,
    *,
    config: ResearchRunConfig,
    classifier: ClassifierProvenance,
    classifier_artifact_path: str | Path | None,
    manifest: RoleManifest,
    manifest_path: str | Path,
    input_evidence_path: str | Path,
    binding: TrainingBinding,
    result: TrainingResult,
    checkpoint_paths: Mapping[int, str | Path],
    input_cache_file_bindings: Sequence[Mapping[str, object]] | None = None,
) -> ConfidenceFitArtifact:
    """Persist a learned confidence fit with actual checkpoint bytes."""

    target = Path(path).resolve()
    _require_config_dataset_namespace(config, manifest.dataset_namespace)
    if target.suffix != ".json" or not target.parent.exists():
        raise ValueError("confidence fit artifact must be JSON in an existing directory")
    classifier_path = None
    classifier_file_sha256 = None
    encoder_identity = None
    if classifier.kind == "public_pretrained_fresh":
        if classifier_artifact_path is None:
            raise ValueError("production confidence fit requires selected classifier evidence")
        classifier_artifact = load_selected_classifier_artifact(
            classifier_artifact_path, expected_config=config
        )
        if classifier_artifact.classifier != classifier:
            raise ValueError("confidence fit classifier artifact is stale")
        if manifest.manifest_sha256 not in classifier_artifact.readiness.manifest_sha256s:
            raise ValueError(
                "confidence-fit manifest is absent from classifier readiness evidence"
            )
        classifier_path = classifier_artifact.source_path
        classifier_file_sha256 = sha256_file(classifier_path)
        encoder_identity = classifier_artifact.encoder_identity
    elif classifier.kind != "synthetic_injected" or classifier_artifact_path is not None:
        raise ValueError("confidence fit classifier evidence kind is invalid")
    role_path = Path(manifest_path).resolve()
    input_path = Path(input_evidence_path).resolve()
    _require_external_or_ignored_destination(role_path)
    _require_external_or_ignored_destination(input_path)
    if (
        {record.role for record in manifest.records} != {Role.CONFIDENCE_FIT}
        or binding.config_sha256 != config.sha256
        or binding.protocol_sha256 != config.protocol_sha256
        or binding.search_table_sha256 != config.search_table.sha256
        or binding.manifest_sha256 != manifest.manifest_sha256
        or binding.manifest_sha256 != result.manifest_sha256
        or binding.classifier_checkpoint_sha256 != classifier.checkpoint_sha256
        or binding.method != result.method
        or binding.seed != result.seed
        or result.method not in LEARNED_TORCH_METHODS
        or result.completed_epoch < 1
        or result.completed_epoch > config.epochs
        or not result.actual_exposure_verified
        or result.update_count < result.completed_epoch
        or result.software_only
        != (
            classifier.kind == "synthetic_injected"
            or not classifier.patient_readiness_verified
        )
    ):
        raise ValueError("confidence fit workflow evidence is incomplete or stale")
    if set(checkpoint_paths) != set(range(1, result.completed_epoch + 1)):
        raise ValueError("confidence fit artifact requires every completed epoch checkpoint")
    records = []
    normalized = []
    for epoch in range(1, result.completed_epoch + 1):
        checkpoint = Path(checkpoint_paths[epoch]).resolve()
        state = load_tensor_checkpoint_state(checkpoint)
        records.append(
            {
                "epoch": epoch,
                "path": str(checkpoint),
                "file_sha256": sha256_file(checkpoint),
                "state_sha256": state_dict_sha256(state),
            }
        )
        normalized.append((epoch, checkpoint))
    if records[-1]["state_sha256"] != result.model_state_sha256:
        raise ValueError("confidence final checkpoint state disagrees with training result")
    cache_file_bindings = None
    if input_cache_file_bindings is not None:
        normalized_bindings = []
        seen_metadata_paths = set()
        for binding_record in input_cache_file_bindings:
            verify_cache_file_binding(binding_record)
            normalized_binding = dict(binding_record)
            metadata_path = normalized_binding["metadata_path"]
            if metadata_path in seen_metadata_paths:
                raise ValueError("confidence cache-file bindings contain duplicates")
            seen_metadata_paths.add(metadata_path)
            normalized_bindings.append(normalized_binding)
        cache_file_bindings = normalized_bindings
    expected_cache_count = config.epochs * result.exposed_record_count
    if classifier.kind == "public_pretrained_fresh" and (
        cache_file_bindings is None
        or len(cache_file_bindings) != expected_cache_count
    ):
        raise ValueError(
            "production confidence fit requires every realized cache file binding"
        )
    if cache_file_bindings is not None:
        indexed_paths = _confidence_cache_index_paths(input_path, classifier)
        bound_paths = tuple(
            Path(str(record["metadata_path"])).resolve()
            for record in cache_file_bindings
        )
        if indexed_paths != bound_paths:
            raise ValueError(
                "confidence cache-file bindings disagree with the authorized index"
            )
    payload = {
        "schema_version": CONFIDENCE_FIT_VERSION,
        "config": config.to_dict(),
        "config_sha256": config.sha256,
        "classifier": classifier.to_dict(),
        "encoder_identity": (
            None if encoder_identity is None else encoder_identity.to_dict()
        ),
        "classifier_artifact_path": None if classifier_path is None else str(classifier_path),
        "classifier_artifact_file_sha256": classifier_file_sha256,
        "manifest_path": str(role_path),
        "manifest_file_sha256": sha256_file(role_path),
        "input_evidence_path": str(input_path),
        "input_evidence_file_sha256": sha256_file(input_path),
        "input_cache_files": cache_file_bindings,
        "binding": binding.to_dict(),
        "result": result.to_dict(),
        "checkpoints": records,
    }
    digest = _sha256_json(payload)
    _atomic_json(target, {"artifact_sha256": digest, "artifact": payload})
    return ConfidenceFitArtifact(
        classifier, encoder_identity, binding, result, tuple(normalized), digest, target, config
    )


def load_confidence_fit_artifact(
    path: str | Path, *, expected_config: ResearchRunConfig | None = None
) -> ConfidenceFitArtifact:
    """Reload a learned confidence fit and re-hash all referenced bytes."""

    source = Path(path).resolve()
    _require_external_or_ignored_destination(source)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
        payload = document["artifact"]
        if set(document) != {"artifact_sha256", "artifact"} or not isinstance(payload, Mapping):
            raise ValueError
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("confidence fit artifact is unavailable or invalid") from exc
    expected = {
        "schema_version", "config", "config_sha256", "classifier",
        "encoder_identity", "classifier_artifact_path",
        "classifier_artifact_file_sha256", "manifest_path",
        "manifest_file_sha256", "input_evidence_path", "input_evidence_file_sha256",
        "input_cache_files", "binding", "result", "checkpoints",
    }
    digest = _sha256_json(payload)
    if document["artifact_sha256"] != digest or set(payload) != expected or payload[
        "schema_version"
    ] != CONFIDENCE_FIT_VERSION:
        raise ValueError("confidence fit artifact integrity/schema is invalid")
    config = ResearchRunConfig.from_dict(payload["config"])
    if config.sha256 != payload["config_sha256"] or (
        expected_config is not None and config != expected_config
    ):
        raise ValueError("confidence fit configuration is stale")
    raw_classifier = payload["classifier"]
    if not isinstance(raw_classifier, Mapping):
        raise ValueError("confidence fit classifier record is invalid")
    if raw_classifier.get("kind") == "public_pretrained_fresh":
        classifier_path = Path(str(payload["classifier_artifact_path"])).resolve()
        if sha256_file(classifier_path) != payload["classifier_artifact_file_sha256"]:
            raise ValueError("confidence fit selected-classifier evidence changed")
        selected_classifier = load_selected_classifier_artifact(
            classifier_path, expected_config=config
        )
        classifier = selected_classifier.classifier
        encoder_identity = selected_classifier.encoder_identity
        if (
            classifier.to_dict() != raw_classifier
            or encoder_identity.to_dict() != payload["encoder_identity"]
        ):
            raise ValueError("confidence fit classifier binding is stale")
    elif raw_classifier.get("kind") == "synthetic_injected":
        if payload["classifier_artifact_path"] is not None or payload[
            "classifier_artifact_file_sha256"
        ] is not None:
            raise ValueError("synthetic confidence fit cannot claim production evidence")
        if payload["encoder_identity"] is not None:
            raise ValueError("synthetic confidence fit cannot claim production encoder identity")
        encoder_identity = None
        classifier = ClassifierProvenance._restore_frozen_record(
            **{
                **raw_classifier,
                "prompts": tuple(raw_classifier["prompts"]),
                "image_mean": tuple(raw_classifier["image_mean"]),
                "image_std": tuple(raw_classifier["image_std"]),
            }
        )
    else:
        raise ValueError("confidence fit classifier evidence kind is invalid")
    manifest_path = Path(str(payload["manifest_path"])).resolve()
    input_path = Path(str(payload["input_evidence_path"])).resolve()
    if (
        sha256_file(manifest_path) != payload["manifest_file_sha256"]
        or sha256_file(input_path) != payload["input_evidence_file_sha256"]
    ):
        raise ValueError("confidence fit manifest/cache evidence changed")
    raw_cache_files = payload["input_cache_files"]
    if raw_cache_files is not None:
        if not isinstance(raw_cache_files, list):
            raise ValueError("confidence cache-file binding table is invalid")
        for cache_binding in raw_cache_files:
            _verify_cache_file_binding_bytes(cache_binding)
        metadata_paths = [
            cache_binding["metadata_path"] for cache_binding in raw_cache_files
        ]
        if len(set(metadata_paths)) != len(metadata_paths):
            raise ValueError("confidence cache-file bindings contain duplicates")
    try:
        binding = TrainingBinding(**payload["binding"])
        result = TrainingResult(**payload["result"])
    except (KeyError, TypeError) as exc:
        raise ValueError("confidence fit result/binding is invalid") from exc
    if (
        binding.config_sha256 != config.sha256
        or binding.protocol_sha256 != config.protocol_sha256
        or binding.search_table_sha256 != config.search_table.sha256
        or binding.manifest_sha256 != result.manifest_sha256
        or (
            classifier.kind == "public_pretrained_fresh"
            and binding.manifest_sha256
            not in selected_classifier.readiness.manifest_sha256s
        )
        or binding.classifier_checkpoint_sha256 != classifier.checkpoint_sha256
        or binding.method != result.method
        or binding.seed != result.seed
        or result.method not in LEARNED_TORCH_METHODS
        or result.completed_epoch < 1
        or result.completed_epoch > config.epochs
        or not result.actual_exposure_verified
        or result.update_count < result.completed_epoch
        or result.software_only
        != (
            classifier.kind == "synthetic_injected"
            or not classifier.patient_readiness_verified
        )
        or (
            classifier.kind == "public_pretrained_fresh"
            and (
                raw_cache_files is None
                or len(raw_cache_files)
                != config.epochs * result.exposed_record_count
            )
        )
    ):
        raise ValueError("confidence fit workflow cannot reproduce its bindings")
    raw_records = payload["checkpoints"]
    if not isinstance(raw_records, list) or len(raw_records) != result.completed_epoch:
        raise ValueError("confidence fit checkpoint history is incomplete")
    paths = []
    for epoch, record in enumerate(raw_records, 1):
        if not isinstance(record, Mapping) or set(record) != {
            "epoch", "path", "file_sha256", "state_sha256"
        } or record["epoch"] != epoch:
            raise ValueError("confidence fit checkpoint record is invalid")
        checkpoint = Path(str(record["path"])).resolve()
        if sha256_file(checkpoint) != record["file_sha256"] or state_dict_sha256(
            load_tensor_checkpoint_state(checkpoint)
        ) != record["state_sha256"]:
            raise ValueError("confidence fit checkpoint content changed")
        paths.append((epoch, checkpoint))
    if raw_records[-1]["state_sha256"] != result.model_state_sha256:
        raise ValueError("confidence final checkpoint state disagrees with training result")
    return ConfidenceFitArtifact(
        classifier, encoder_identity, binding, result, tuple(paths), digest, source, config
    )


def save_classifier_fit_artifact(
    path: str | Path,
    *,
    config: ResearchRunConfig,
    initialization: PublicInitializationArtifact,
    readiness_artifact_path: str | Path,
    manifest: RoleManifest,
    manifest_path: str | Path,
    binding: TrainingBinding,
    result: TrainingResult,
    checkpoint_paths: Mapping[int, str | Path],
) -> ClassifierFitArtifact:
    """Persist actual classifier-fit exposure and every completed epoch state."""

    target = Path(path).resolve()
    _require_config_dataset_namespace(config, manifest.dataset_namespace)
    if target.suffix != ".json" or not target.parent.exists():
        raise ValueError("classifier fit artifact must be JSON in an existing directory")
    current_initialization = load_public_initialization_artifact(initialization.source_path)
    _require_initialization_config(config, current_initialization)
    readiness_path = Path(readiness_artifact_path).resolve()
    readiness = load_verified_readiness_audit(
        readiness_path, require_patient_ready=False
    )
    role_path = Path(manifest_path).resolve()
    _require_external_or_ignored_destination(role_path)
    if (
        {record.role for record in manifest.records} != {Role.CLASSIFIER_FIT}
        or manifest.manifest_sha256 != binding.manifest_sha256
        or manifest.manifest_sha256 != result.manifest_sha256
        or manifest.manifest_sha256 not in readiness.manifest_sha256s
        or binding.config_sha256 != config.sha256
        or binding.protocol_sha256 != config.protocol_sha256
        or binding.search_table_sha256 != config.search_table.sha256
        or binding.classifier_checkpoint_sha256
        != current_initialization.classifier.checkpoint_sha256
        or binding.method != "fresh_classifier"
        or result.method != "fresh_classifier"
        or result.seed != binding.seed
        or result.completed_epoch < 1
        or result.completed_epoch > config.epochs
        or result.update_count < result.completed_epoch
        or not result.actual_exposure_verified
        or result.software_only != (not readiness.patient_ready)
    ):
        raise ValueError("classifier fit workflow evidence is incomplete or stale")
    if set(checkpoint_paths) != set(range(1, result.completed_epoch + 1)):
        raise ValueError("classifier fit artifact requires every completed epoch checkpoint")
    checkpoints = []
    normalized_paths = []
    for epoch in range(1, result.completed_epoch + 1):
        checkpoint = Path(checkpoint_paths[epoch]).resolve()
        _require_external_or_ignored_destination(checkpoint)
        state = load_tensor_checkpoint_state(checkpoint)
        checkpoints.append(
            {
                "epoch": epoch,
                "path": str(checkpoint),
                "file_sha256": sha256_file(checkpoint),
                "state_sha256": state_dict_sha256(state),
            }
        )
        normalized_paths.append((epoch, checkpoint))
    if checkpoints[-1]["state_sha256"] != result.model_state_sha256:
        raise ValueError("classifier final checkpoint state disagrees with training result")
    payload = {
        "schema_version": CLASSIFIER_FIT_VERSION,
        "config": config.to_dict(),
        "config_sha256": config.sha256,
        "initialization_path": str(current_initialization.source_path),
        "initialization_file_sha256": sha256_file(current_initialization.source_path),
        "readiness_path": str(readiness_path),
        "readiness_file_sha256": sha256_file(readiness_path),
        "manifest_path": str(role_path),
        "manifest_file_sha256": sha256_file(role_path),
        "manifest_sha256": manifest.manifest_sha256,
        "binding": binding.to_dict(),
        "result": result.to_dict(),
        "checkpoints": checkpoints,
    }
    digest = _sha256_json(payload)
    _atomic_json(target, {"artifact_sha256": digest, "artifact": payload})
    return ClassifierFitArtifact(
        current_initialization,
        readiness,
        binding,
        result,
        tuple(normalized_paths),
        digest,
        target,
        config,
    )


def load_classifier_fit_artifact(
    path: str | Path, *, expected_config: ResearchRunConfig | None = None
) -> ClassifierFitArtifact:
    """Reload classifier fitting only after re-hashing all referenced evidence."""

    source = Path(path).resolve()
    _require_external_or_ignored_destination(source)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
        payload = document["artifact"]
        if set(document) != {"artifact_sha256", "artifact"} or not isinstance(payload, Mapping):
            raise ValueError
        digest = _sha256_json(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("classifier fit artifact is unavailable or invalid") from exc
    expected = {
        "schema_version", "config", "config_sha256", "initialization_path",
        "initialization_file_sha256", "readiness_path", "readiness_file_sha256",
        "manifest_path", "manifest_file_sha256", "manifest_sha256", "binding",
        "result", "checkpoints",
    }
    if (
        document["artifact_sha256"] != digest
        or set(payload) != expected
        or payload["schema_version"] != CLASSIFIER_FIT_VERSION
    ):
        raise ValueError("classifier fit artifact integrity/schema is invalid")
    config = ResearchRunConfig.from_dict(payload["config"])
    if config.sha256 != payload["config_sha256"] or (
        expected_config is not None and config != expected_config
    ):
        raise ValueError("classifier fit configuration changed")
    initialization_path = Path(str(payload["initialization_path"])).resolve()
    readiness_path = Path(str(payload["readiness_path"])).resolve()
    manifest_path = Path(str(payload["manifest_path"])).resolve()
    for evidence_path, expected_hash in (
        (initialization_path, payload["initialization_file_sha256"]),
        (readiness_path, payload["readiness_file_sha256"]),
        (manifest_path, payload["manifest_file_sha256"]),
    ):
        if sha256_file(evidence_path) != expected_hash:
            raise ValueError("classifier fit referenced evidence changed")
    initialization = load_public_initialization_artifact(initialization_path)
    _require_initialization_config(config, initialization)
    readiness = load_verified_readiness_audit(
        readiness_path, require_patient_ready=False
    )
    try:
        binding = TrainingBinding(**payload["binding"])
        result = TrainingResult(**payload["result"])
    except (KeyError, TypeError) as exc:
        raise ValueError("classifier fit binding/result is invalid") from exc
    if (
        binding.config_sha256 != config.sha256
        or binding.protocol_sha256 != config.protocol_sha256
        or binding.search_table_sha256 != config.search_table.sha256
        or binding.manifest_sha256 != payload["manifest_sha256"]
        or result.manifest_sha256 != payload["manifest_sha256"]
        or payload["manifest_sha256"] not in readiness.manifest_sha256s
        or binding.classifier_checkpoint_sha256 != initialization.classifier.checkpoint_sha256
        or binding.method != "fresh_classifier"
        or result.method != "fresh_classifier"
        or result.seed != binding.seed
        or not result.actual_exposure_verified
        or result.software_only != (not readiness.patient_ready)
        or result.completed_epoch < 1
        or result.completed_epoch > config.epochs
        or result.update_count < result.completed_epoch
    ):
        raise ValueError("classifier fit workflow cannot reproduce its bindings")
    raw_checkpoints = payload["checkpoints"]
    if not isinstance(raw_checkpoints, list) or len(raw_checkpoints) != result.completed_epoch:
        raise ValueError("classifier fit checkpoint history is incomplete")
    paths = []
    for expected_epoch, record in enumerate(raw_checkpoints, 1):
        if not isinstance(record, Mapping) or set(record) != {
            "epoch", "path", "file_sha256", "state_sha256"
        } or record["epoch"] != expected_epoch:
            raise ValueError("classifier fit checkpoint history is invalid")
        checkpoint = Path(str(record["path"])).resolve()
        if (
            sha256_file(checkpoint) != record["file_sha256"]
            or state_dict_sha256(load_tensor_checkpoint_state(checkpoint))
            != record["state_sha256"]
        ):
            raise ValueError("classifier fit checkpoint content changed")
        paths.append((expected_epoch, checkpoint))
    if raw_checkpoints[-1]["state_sha256"] != result.model_state_sha256:
        raise ValueError("classifier final checkpoint state disagrees with training result")
    return ClassifierFitArtifact(
        initialization, readiness, binding, result, tuple(paths), digest, source, config
    )


def _classifier_selection_digest(
    tune_manifest_sha256: str, checkpoints: Sequence[object], selected_sha256: str
) -> str:
    return _sha256_json(
        {
            "metric": "multiclass_nll",
            "role": "tune",
            "manifest_sha256": tune_manifest_sha256,
            "candidates": [
                {
                    "epoch": item.epoch,
                    "artifact_sha256": item.artifact_sha256,
                    "tune_nll": item.tune_nll,
                }
                for item in checkpoints
            ],
            "selected_artifact_sha256": selected_sha256,
        }
    )


def save_selected_classifier_artifact(
    path: str | Path,
    *,
    config: ResearchRunConfig,
    initialization: PublicInitializationArtifact,
    readiness_artifact_path: str | Path,
    fit_artifact_path: str | Path,
    fit_binding: TrainingBinding,
    fit_result: TrainingResult,
    tune_manifest: RoleManifest,
    tune_manifest_path: str | Path,
    tune_checkpoints: Sequence[object],
    checkpoint_paths: Mapping[int, str | Path],
    selected: ClassifierProvenance,
) -> SelectedClassifierArtifact:
    """Persist classifier fit/tune history and every independent evidence path."""

    from .evaluation import ClassifierTuneCheckpoint

    target = Path(path).resolve()
    _require_config_dataset_namespace(config, tune_manifest.dataset_namespace)
    if target.suffix != ".json" or not target.parent.exists():
        raise ValueError("selected classifier artifact must be JSON in an existing directory")
    current_initialization = load_public_initialization_artifact(initialization.source_path)
    _require_initialization_config(config, current_initialization)
    readiness_path = Path(readiness_artifact_path).resolve()
    readiness = load_verified_readiness_audit(
        readiness_path, require_patient_ready=False
    )
    if tune_manifest.manifest_sha256 not in readiness.manifest_sha256s:
        raise ValueError("tune manifest is absent from the readiness audit")
    fit_artifact = load_classifier_fit_artifact(fit_artifact_path, expected_config=config)
    tune_role_path = Path(tune_manifest_path).resolve()
    _require_external_or_ignored_destination(tune_role_path)
    checkpoints = tuple(tune_checkpoints)
    if not checkpoints or any(not isinstance(item, ClassifierTuneCheckpoint) for item in checkpoints):
        raise ValueError("classifier tune evidence is missing or invalid")
    if len({item.epoch for item in checkpoints}) != len(checkpoints):
        raise ValueError("classifier tune evidence contains duplicate epochs")
    if tuple(sorted(item.epoch for item in checkpoints)) != config.search_table.checkpoint_epochs:
        raise ValueError("classifier tune evidence does not cover the frozen epoch budget")
    if set(checkpoint_paths) != {item.epoch for item in checkpoints}:
        raise ValueError("classifier checkpoint path table is incomplete")
    checkpoint_records = []
    for item in checkpoints:
        checkpoint_path = Path(checkpoint_paths[item.epoch]).resolve()
        _require_external_or_ignored_destination(checkpoint_path)
        if sha256_file(checkpoint_path) != item.artifact_sha256:
            raise ValueError("classifier tune result disagrees with checkpoint bytes")
        checkpoint_records.append(
            {
                "epoch": item.epoch,
                "tune_nll": item.tune_nll,
                "artifact_sha256": item.artifact_sha256,
                "path": str(checkpoint_path),
                "state_sha256": state_dict_sha256(load_tensor_checkpoint_state(checkpoint_path)),
            }
        )
    chosen = min(checkpoints, key=lambda item: (item.tune_nll, item.epoch))
    selected_path = Path(checkpoint_paths[chosen.epoch]).resolve()
    encoder_identity = _selected_encoder_identity(
        current_initialization, selected_path
    )
    if (
        selected.kind != "public_pretrained_fresh"
        or not selected.workflow_complete
        or selected.patient_readiness_verified != readiness.patient_ready
        or selected.checkpoint_sha256 != chosen.artifact_sha256
        or selected.initialization_sha256
        != current_initialization.classifier.initialization_sha256
        or selected.public_weight_content_sha256
        != current_initialization.classifier.public_weight_content_sha256
        or selected.readiness_audit_sha256 != readiness.sha256
        or selected.classifier_fit_manifest_sha256 != fit_result.manifest_sha256
        or selected.classifier_fit_update_count != fit_result.update_count
        or selected.tune_manifest_sha256 != tune_manifest.manifest_sha256
        or selected.tune_selection_sha256
        != _classifier_selection_digest(
            tune_manifest.manifest_sha256, checkpoints, chosen.artifact_sha256
        )
    ):
        raise ValueError("selected classifier provenance disagrees with fit/tune evidence")
    if (
        fit_binding.config_sha256 != config.sha256
        or fit_binding.search_table_sha256 != config.search_table.sha256
        or fit_binding.protocol_sha256 != config.protocol_sha256
        or fit_binding.manifest_sha256 != fit_result.manifest_sha256
        or fit_binding.classifier_checkpoint_sha256
        != current_initialization.classifier.checkpoint_sha256
        or fit_binding.method != "fresh_classifier"
        or fit_result.method != "fresh_classifier"
        or not fit_result.actual_exposure_verified
        or fit_result.software_only != (not readiness.patient_ready)
        or fit_result.update_count < 1
        or fit_result.completed_epoch != config.epochs
        or fit_artifact.binding != fit_binding
        or fit_artifact.result != fit_result
        or fit_artifact.initialization.sha256 != current_initialization.sha256
        or fit_artifact.readiness.sha256 != readiness.sha256
    ):
        raise ValueError("classifier fit evidence is incomplete or stale")
    payload = {
        "schema_version": SELECTED_CLASSIFIER_VERSION,
        "config_sha256": config.sha256,
        "classifier": selected.to_dict(),
        "encoder_identity": encoder_identity.to_dict(),
        "initialization_path": str(current_initialization.source_path),
        "initialization_file_sha256": sha256_file(current_initialization.source_path),
        "initialization_sha256": current_initialization.sha256,
        "readiness_path": str(readiness_path),
        "readiness_file_sha256": sha256_file(readiness_path),
        "readiness_sha256": readiness.sha256,
        "fit_artifact_path": str(fit_artifact.source_path),
        "fit_artifact_file_sha256": sha256_file(fit_artifact.source_path),
        "fit_artifact_sha256": fit_artifact.sha256,
        "fit_binding": fit_binding.to_dict(),
        "fit_result": fit_result.to_dict(),
        "tune_manifest_path": str(tune_role_path),
        "tune_manifest_file_sha256": sha256_file(tune_role_path),
        "tune_manifest_sha256": tune_manifest.manifest_sha256,
        "tune_checkpoints": checkpoint_records,
        "selected_epoch": chosen.epoch,
        "selected_checkpoint_path": str(selected_path),
    }
    digest = _sha256_json(payload)
    _atomic_json(target, {"artifact_sha256": digest, "artifact": payload})
    return SelectedClassifierArtifact(
        selected,
        current_initialization,
        readiness,
        selected_path,
        encoder_identity,
        digest,
        target,
        config.sha256,
    )


def load_selected_classifier_artifact(
    path: str | Path,
    *,
    expected_config_sha256: str | None = None,
    expected_config: ResearchRunConfig | None = None,
) -> SelectedClassifierArtifact:
    """Reload all public/readiness/fit/tune/checkpoint evidence and derive provenance."""

    from .evaluation import ClassifierTuneCheckpoint

    source = Path(path).resolve()
    _require_external_or_ignored_destination(source)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
        payload = document["artifact"]
        if set(document) != {"artifact_sha256", "artifact"} or not isinstance(payload, Mapping):
            raise ValueError
        digest = _sha256_json(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("selected classifier artifact is unavailable or invalid") from exc
    if document["artifact_sha256"] != digest:
        raise ValueError("selected classifier artifact integrity check failed")
    expected = {
        "schema_version",
        "config_sha256",
        "classifier",
        "encoder_identity",
        "initialization_path",
        "initialization_file_sha256",
        "initialization_sha256",
        "readiness_path",
        "readiness_file_sha256",
        "readiness_sha256",
        "fit_artifact_path",
        "fit_artifact_file_sha256",
        "fit_artifact_sha256",
        "fit_binding",
        "fit_result",
        "tune_manifest_path",
        "tune_manifest_file_sha256",
        "tune_manifest_sha256",
        "tune_checkpoints",
        "selected_epoch",
        "selected_checkpoint_path",
    }
    if set(payload) != expected or payload["schema_version"] != SELECTED_CLASSIFIER_VERSION:
        raise ValueError("selected classifier artifact schema is invalid")
    if expected_config is not None:
        if expected_config_sha256 is not None and expected_config_sha256 != expected_config.sha256:
            raise ValueError("conflicting selected classifier configuration expectations")
        expected_config_sha256 = expected_config.sha256
    if expected_config_sha256 is not None and payload["config_sha256"] != expected_config_sha256:
        raise ValueError("selected classifier configuration binding is stale")
    initialization_path = Path(str(payload["initialization_path"])).resolve()
    readiness_path = Path(str(payload["readiness_path"])).resolve()
    if sha256_file(initialization_path) != payload["initialization_file_sha256"]:
        raise ValueError("public initialization evidence changed")
    initialization = load_public_initialization_artifact(initialization_path)
    if expected_config is not None:
        _require_initialization_config(expected_config, initialization)
    if initialization.sha256 != payload["initialization_sha256"]:
        raise ValueError("public initialization evidence identity changed")
    if sha256_file(readiness_path) != payload["readiness_file_sha256"]:
        raise ValueError("readiness evidence changed")
    readiness = load_verified_readiness_audit(
        readiness_path, require_patient_ready=False
    )
    if readiness.sha256 != payload["readiness_sha256"]:
        raise ValueError("readiness evidence identity changed")
    fit_artifact_path = Path(str(payload["fit_artifact_path"])).resolve()
    tune_manifest_path = Path(str(payload["tune_manifest_path"])).resolve()
    if (
        sha256_file(fit_artifact_path) != payload["fit_artifact_file_sha256"]
        or sha256_file(tune_manifest_path) != payload["tune_manifest_file_sha256"]
    ):
        raise ValueError("classifier fit or tune manifest evidence changed")
    fit_artifact = load_classifier_fit_artifact(
        fit_artifact_path, expected_config=expected_config
    )
    if (
        fit_artifact.sha256 != payload["fit_artifact_sha256"]
        or fit_artifact.config.sha256 != payload["config_sha256"]
    ):
        raise ValueError("classifier fit evidence identity changed")
    persisted_config = fit_artifact.config
    try:
        binding = TrainingBinding(**payload["fit_binding"])
        fit_result = TrainingResult(**payload["fit_result"])
        records = tuple(payload["tune_checkpoints"])
        if any(
            not isinstance(item, Mapping)
            or set(item)
            != {"epoch", "tune_nll", "artifact_sha256", "path", "state_sha256"}
            for item in records
        ):
            raise ValueError
        checkpoints = tuple(
            ClassifierTuneCheckpoint(
                epoch=item["epoch"],
                artifact_sha256=item["artifact_sha256"],
                tune_nll=item["tune_nll"],
            )
            for item in records
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("selected classifier fit/tune history is invalid") from exc
    if not checkpoints or len({item.epoch for item in checkpoints}) != len(checkpoints):
        raise ValueError("selected classifier tune history is incomplete")
    if tuple(sorted(item.epoch for item in checkpoints)) != (
        persisted_config.search_table.checkpoint_epochs
    ):
        raise ValueError("selected classifier tune budget is incomplete")
    for item, record in zip(checkpoints, records):
        checkpoint_path = Path(str(record["path"])).resolve()
        if (
            sha256_file(checkpoint_path) != item.artifact_sha256
            or state_dict_sha256(load_tensor_checkpoint_state(checkpoint_path))
            != record["state_sha256"]
        ):
            raise ValueError("classifier checkpoint evidence changed")
    chosen = min(checkpoints, key=lambda item: (item.tune_nll, item.epoch))
    if chosen.epoch != payload["selected_epoch"]:
        raise ValueError("classifier selected epoch is stale")
    selected = ClassifierProvenance.fresh_selected(
        initialization.classifier,
        checkpoint_sha256=chosen.artifact_sha256,
        classifier_fit_manifest_sha256=fit_result.manifest_sha256,
        classifier_fit_update_count=fit_result.update_count,
        tune_selection_sha256=_classifier_selection_digest(
            str(payload["tune_manifest_sha256"]), checkpoints, chosen.artifact_sha256
        ),
        tune_manifest_sha256=str(payload["tune_manifest_sha256"]),
        readiness=readiness,
    )
    if (
        selected.to_dict() != payload["classifier"]
        or binding.config_sha256 != payload["config_sha256"]
        or binding.manifest_sha256 != fit_result.manifest_sha256
        or binding.classifier_checkpoint_sha256 != initialization.classifier.checkpoint_sha256
        or binding.method != "fresh_classifier"
        or fit_result.method != "fresh_classifier"
        or not fit_result.actual_exposure_verified
        or fit_result.software_only != (not readiness.patient_ready)
        or fit_result.update_count < 1
        or binding != fit_artifact.binding
        or fit_result != fit_artifact.result
        or fit_artifact.initialization.sha256 != initialization.sha256
        or fit_artifact.readiness.sha256 != readiness.sha256
        or str(payload["tune_manifest_sha256"]) not in readiness.manifest_sha256s
    ):
        raise ValueError("selected classifier history cannot reproduce its provenance")
    if (
        binding.protocol_sha256 != persisted_config.protocol_sha256
        or binding.search_table_sha256 != persisted_config.search_table.sha256
        or binding.seed not in persisted_config.seeds
        or fit_result.completed_epoch != persisted_config.epochs
    ):
        raise ValueError("selected classifier training binding disagrees with configuration")
    selected_path = Path(str(payload["selected_checkpoint_path"])).resolve()
    if selected_path != Path(str(records[checkpoints.index(chosen)]["path"])).resolve():
        raise ValueError("selected classifier checkpoint path is stale")
    encoder_identity = _selected_encoder_identity(initialization, selected_path)
    if encoder_identity.to_dict() != payload["encoder_identity"]:
        raise ValueError("selected classifier encoder/token identity changed")
    return SelectedClassifierArtifact(
        selected,
        initialization,
        readiness,
        selected_path,
        encoder_identity,
        digest,
        source,
        str(payload["config_sha256"]),
    )


@dataclass(frozen=True)
class ClassifierImageBatch:
    views: Mapping[str, Tensor]
    labels: Tensor


def default_private_image_reader(
    record: PrivateExamRecord, view: str, image_root: str | Path
) -> Image.Image:
    """Open exactly one already-authorized image using its role-manifest path."""

    root = Path(image_root).resolve()
    reference = record.views[view]
    path = (root / reference.path).resolve()
    if path != root and root not in path.parents:
        raise ValueError("private image path escapes the configured image root")
    if reference.content_sha256 is not None and sha256_file(path) != reference.content_sha256:
        raise ValueError("private image content changed after manifest creation")
    if path.suffix.lower() == ".dcm":
        try:
            import pydicom
        except ImportError as exc:  # pragma: no cover - installation dependent
            raise RuntimeError("DICOM loading requires pydicom and its configured decoder") from exc
        dcm = pydicom.dcmread(path)
        pixels = dcm.pixel_array.astype(np.float32)
        pixels -= pixels.min()
        pixels /= pixels.max() + 1e-5
        return Image.fromarray((pixels * 255).astype(np.uint8)).convert("RGB")
    with Image.open(path) as image:
        return image.convert("RGB")


def role_image_batch_loader(
    records: Sequence[PrivateExamRecord],
    *,
    image_root: str | Path,
    image_size: int,
    optimizer: OptimizerConfig,
    epoch: int,
    seed: int,
    training: bool,
    image_reader: Callable[[PrivateExamRecord, str, str | Path], Image.Image] = default_private_image_reader,
) -> Iterable[RoleBoundBatch]:
    """Yield bounded four-view batches without opening any unselected role."""

    transform, eval_transform = build_transforms(
        image_size, {"name": "randaugment"} if training else None
    )
    selected_transform = transform if training else eval_transform
    generator = torch.Generator().manual_seed(seed + epoch * 1_000_003)
    order = torch.randperm(len(records), generator=generator).tolist() if training else list(range(len(records)))
    for start in range(0, len(order), optimizer.batch_size):
        indexes = order[start : start + optimizer.batch_size]
        batch_records = [records[index] for index in indexes]
        views = {
            view: torch.stack(
                [
                    selected_transform(image_reader(record, view, image_root))
                    for record in batch_records
                ]
            )
            for view in CANONICAL_VIEWS
        }
        labels = torch.tensor([record.density for record in batch_records], dtype=torch.long)
        yield RoleBoundBatch(
            tuple(record.exam_key for record in batch_records),
            ClassifierImageBatch(views, labels),
        )


def reload_verified_public_classifier(
    initialization: PublicInitializationArtifact,
    *,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, Tensor]:
    """Recreate the pinned architecture and require the persisted public bytes."""

    current = load_public_initialization_artifact(initialization.source_path)
    model, input_ids, provenance = load_pinned_public_clip_classifier(
        current.classifier.backbone, current.dataset, device=device
    )
    saved_tokens = load_tensor_checkpoint_state(current.input_ids_path)["input_ids"]
    if provenance != current.classifier or not torch.equal(input_ids.cpu(), saved_tokens):
        raise ValueError("current public model/token source differs from frozen initialization")
    model.load_state_dict(load_tensor_checkpoint_state(current.weights_path), strict=True)
    return model, saved_tokens.to(device)


def fit_classifier_images_with_role_access(
    *,
    artifact_path: str | Path,
    manifest: RoleManifest,
    manifest_path: str | Path,
    model: nn.Module,
    input_ids: Tensor,
    initialization: PublicInitializationArtifact,
    readiness_artifact_path: str | Path,
    config: ResearchRunConfig,
    image_root: str | Path,
    checkpoint_directory: str | Path,
    resume_checkpoint_path: str | Path,
    resume: bool = False,
    stop_after_epoch: int | None = None,
    image_reader: Callable[
        [PrivateExamRecord, str, str | Path], Image.Image
    ] = default_private_image_reader,
) -> ClassifierFitArtifact:
    """Fit the fresh classifier on classifier_fit images and persist each epoch."""

    _require_config_dataset_namespace(config, manifest.dataset_namespace)
    current_initialization = load_public_initialization_artifact(initialization.source_path)
    _require_initialization_config(config, current_initialization)
    readiness = load_verified_readiness_audit(
        readiness_artifact_path, require_patient_ready=False
    )
    saved_input_ids = load_tensor_checkpoint_state(
        current_initialization.input_ids_path
    )["input_ids"]
    if module_state_sha256(model) != current_initialization.classifier.checkpoint_sha256:
        raise ValueError("classifier model does not contain the frozen public initialization")
    if not torch.equal(input_ids.detach().cpu(), saved_input_ids):
        raise ValueError("classifier prompt tokens disagree with the frozen initialization")
    try:
        device = next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("classifier model has no trainable parameters") from exc
    input_ids = input_ids.to(device)
    directory = Path(checkpoint_directory).resolve()
    if not directory.is_dir():
        raise ValueError("classifier checkpoint directory must already exist")
    checkpoint_paths: dict[int, Path] = {}
    for epoch in range(1, (stop_after_epoch or config.epochs) + 1):
        candidate = directory / f"classifier-epoch-{epoch}.safetensors"
        _require_external_or_ignored_destination(candidate)
        if candidate.exists():
            checkpoint_paths[epoch] = candidate

    binding = TrainingBinding(
        protocol_sha256=config.protocol_sha256,
        config_sha256=config.sha256,
        search_table_sha256=config.search_table.sha256,
        manifest_sha256=manifest.manifest_sha256,
        classifier_checkpoint_sha256=current_initialization.classifier.checkpoint_sha256,
        method="fresh_classifier",
        seed=config.seeds[0],
    )

    def batches(records: tuple[PrivateExamRecord, ...], epoch: int) -> Iterable[RoleBoundBatch]:
        return role_image_batch_loader(
            records,
            image_root=image_root,
            image_size=current_initialization.classifier.image_size,
            optimizer=config.optimizer,
            epoch=epoch,
            seed=binding.seed,
            training=True,
            image_reader=image_reader,
        )

    def loss(current: nn.Module, payload: object) -> Tensor:
        if not isinstance(payload, ClassifierImageBatch):
            raise TypeError("classifier batch adapter returned an invalid payload")
        views = {name: value.to(device) for name, value in payload.views.items()}
        labels = payload.labels.to(device)
        return F.cross_entropy(current(views, input_ids), labels)

    def save_epoch(epoch: int, current: nn.Module) -> None:
        destination = directory / f"classifier-epoch-{epoch}.safetensors"
        if destination.exists():
            raise FileExistsError("classifier epoch checkpoint already exists")
        checkpoint_paths[epoch] = save_tensor_checkpoint(current, destination)

    result = fit_fresh_classifier_with_role_access(
        manifest=manifest,
        model=model,
        classifier=current_initialization.classifier,
        readiness=readiness,
        optimizer_config=config.optimizer,
        epochs=config.epochs,
        binding=binding,
        batch_loader=batches,
        loss_fn=loss,
        checkpoint_path=resume_checkpoint_path,
        resume=resume,
        stop_after_epoch=stop_after_epoch,
        epoch_callback=save_epoch,
    )
    return save_classifier_fit_artifact(
        artifact_path,
        config=config,
        initialization=current_initialization,
        readiness_artifact_path=readiness_artifact_path,
        manifest=manifest,
        manifest_path=manifest_path,
        binding=binding,
        result=result,
        checkpoint_paths=checkpoint_paths,
    )


def evaluate_classifier_checkpoints_with_role_access(
    *,
    manifest: RoleManifest,
    model: nn.Module,
    input_ids: Tensor,
    checkpoint_paths: Mapping[int, str | Path],
    image_root: str | Path,
    config: ResearchRunConfig,
    image_reader: Callable[
        [PrivateExamRecord, str, str | Path], Image.Image
    ] = default_private_image_reader,
) -> tuple[object, ...]:
    """Compute tune NLL for every persisted classifier epoch from authorized images."""

    from .evaluation import ClassifierTuneCheckpoint

    if set(checkpoint_paths) != set(config.search_table.checkpoint_epochs):
        raise ValueError("classifier tune evaluation requires the full frozen epoch table")
    try:
        device = next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("classifier model has no parameters") from exc
    input_ids = input_ids.to(device)

    def authorized(records: tuple[PrivateExamRecord, ...]) -> tuple[ClassifierTuneCheckpoint, ...]:
        expected_keys = {record.exam_key for record in records}
        results = []
        for epoch in config.search_table.checkpoint_epochs:
            checkpoint = Path(checkpoint_paths[epoch]).resolve()
            _require_external_or_ignored_destination(checkpoint)
            model.load_state_dict(load_tensor_checkpoint_state(checkpoint), strict=True)
            model.eval()
            total = 0.0
            count = 0
            seen: set[str] = set()
            with torch.no_grad():
                for batch in role_image_batch_loader(
                    records,
                    image_root=image_root,
                    image_size=pinned_public_clip_configuration(config.backbone).image_size,
                    optimizer=config.optimizer,
                    epoch=epoch,
                    seed=config.seeds[0],
                    training=False,
                    image_reader=image_reader,
                ):
                    if not isinstance(batch.payload, ClassifierImageBatch):
                        raise TypeError("classifier tune batch is invalid")
                    if any(key in seen for key in batch.exam_keys):
                        raise ValueError("classifier tune cohort contains duplicate exposure")
                    seen.update(batch.exam_keys)
                    logits = model(
                        {name: value.to(device) for name, value in batch.payload.views.items()},
                        input_ids,
                    )
                    labels = batch.payload.labels.to(device)
                    total += float(F.cross_entropy(logits, labels, reduction="sum"))
                    count += labels.numel()
            if seen != expected_keys or count != len(records):
                raise ValueError("classifier tune checkpoint omitted authorized rows")
            results.append(
                ClassifierTuneCheckpoint(epoch, sha256_file(checkpoint), total / count)
            )
        return tuple(results)

    return run_with_role_access(
        manifest,
        operation=Operation.TUNE_SELECTION,
        roles=(Role.TUNE,),
        loader=authorized,
    )


def select_classifier_from_images_with_role_access(
    *,
    output_path: str | Path,
    fit_artifact_path: str | Path,
    tune_manifest: RoleManifest,
    tune_manifest_path: str | Path,
    image_root: str | Path,
    config: ResearchRunConfig,
    device: str | torch.device = "cpu",
    image_reader: Callable[
        [PrivateExamRecord, str, str | Path], Image.Image
    ] = default_private_image_reader,
) -> SelectedClassifierArtifact:
    """Reload fit evidence, perform tune inference, select, and persist the classifier."""

    _require_config_dataset_namespace(config, tune_manifest.dataset_namespace)
    from .evaluation import select_fresh_classifier_on_tune

    fit = load_classifier_fit_artifact(fit_artifact_path, expected_config=config)
    if fit.result.completed_epoch != config.epochs:
        raise ValueError("classifier fit has not completed the frozen epoch budget")
    model, input_ids = reload_verified_public_classifier(fit.initialization, device=device)
    checkpoints = evaluate_classifier_checkpoints_with_role_access(
        manifest=tune_manifest,
        model=model,
        input_ids=input_ids,
        checkpoint_paths=dict(fit.checkpoint_paths),
        image_root=image_root,
        config=config,
        image_reader=image_reader,
    )
    selected = select_fresh_classifier_on_tune(
        tune_manifest,
        initialization=fit.initialization.classifier,
        classifier_fit_result=fit.result,
        readiness=fit.readiness,
        checkpoint_evaluator=lambda _records: checkpoints,
    )
    readiness_path = json.loads(fit.source_path.read_text(encoding="utf-8"))["artifact"][
        "readiness_path"
    ]
    return save_selected_classifier_artifact(
        output_path,
        config=config,
        initialization=fit.initialization,
        readiness_artifact_path=readiness_path,
        fit_artifact_path=fit.source_path,
        fit_binding=fit.binding,
        fit_result=fit.result,
        tune_manifest=tune_manifest,
        tune_manifest_path=tune_manifest_path,
        tune_checkpoints=checkpoints,
        checkpoint_paths=dict(fit.checkpoint_paths),
        selected=selected,
    )


__all__ = [
    "CLASSIFIER_FIT_VERSION",
    "CONFIDENCE_FIT_VERSION",
    "ClassifierImageBatch",
    "ClassifierFitArtifact",
    "ConfidenceFitArtifact",
    "PUBLIC_INITIALIZATION_VERSION",
    "PublicInitializationArtifact",
    "SELECTED_CLASSIFIER_VERSION",
    "SelectedClassifierArtifact",
    "default_private_image_reader",
    "evaluate_classifier_checkpoints_with_role_access",
    "fit_classifier_images_with_role_access",
    "load_classifier_fit_artifact",
    "load_confidence_fit_artifact",
    "load_public_initialization_artifact",
    "load_selected_classifier_artifact",
    "load_tensor_checkpoint_state",
    "role_image_batch_loader",
    "reload_verified_public_classifier",
    "save_classifier_fit_artifact",
    "save_confidence_fit_artifact",
    "save_public_initialization_artifact",
    "save_selected_classifier_artifact",
    "save_tensor_checkpoint",
    "select_classifier_from_images_with_role_access",
]
