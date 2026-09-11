"""Provenance-bound safetensors caches for frozen view-risk features and targets.

These unsigned SHA-256 records detect stale or mismatched artifacts under a
caller's trusted expected provenance.  They are not authentication against a
hostile artifact author.  Every persistence API requires an explicit destination;
private keys and labels must remain in caller-managed external/ignored storage.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

import torch
from torch import Tensor

from mmdc_clip_f.provenance import sha256_file

from .features import (
    FrozenEncoderIdentity,
    FrozenViewFeatures,
    VerifiedFrozenEncoder,
    tensor_sha256,
)
from .fusion import FusionPairs, fuse_view_logits, validate_fusion_pairs
from .inputs import CANONICAL_VIEWS, NUM_CLASSES, canonicalize_observed_views
from .perturbations import (
    PerturbationMetadata,
    PerturbationSpec,
    RealizedParent,
    resolve_parameters,
    validate_training_perturbation,
)
from .roles import Operation, PrivateExamRecord, Role, RoleManifest, run_with_role_access
from .targets import InterventionTargets, build_intervention_targets


CACHE_SCHEMA_VERSION = "view-risk-frozen-cache/v1"
TARGET_VERSION = "view-risk-intervention-target/v1"
_CACHE_OPERATIONS = frozenset(
    {
        Operation.CONFIDENCE_FITTING,
        Operation.TUNE_SELECTION,
        Operation.PILOT_EVALUATION,
    }
)


class ArtifactIntegrityError(ValueError):
    """Raised when artifact bytes or their scientific relationships are stale."""


class ProvenanceMismatchError(ValueError):
    """Raised when an artifact differs from caller-supplied expected provenance."""


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_sha256(value: object, field: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return str(value)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _hash_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _private_keys(values: object, field: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field} must be an ordered sequence, not one string")
    try:
        result = tuple(_required_string(value, field) for value in values)  # type: ignore[union-attr]
    except TypeError as exc:
        raise TypeError(f"{field} must be an ordered sequence") from exc
    if not result:
        raise ValueError(f"{field} cannot be empty")
    return result


def _snapshot_perturbations(
    values: Mapping[str, Mapping[str, object] | None], observed: tuple[str, ...]
) -> Mapping[str, Mapping[str, object] | None]:
    if not isinstance(values, Mapping) or set(values) != set(observed):
        raise ValueError("perturbations must describe every and only observed parent view")
    snapshot: dict[str, Mapping[str, object] | None] = {}
    for view in observed:
        value = values[view]
        if value is None:
            snapshot[view] = None
            continue
        if not isinstance(value, Mapping):
            raise TypeError("perturbation records must be mappings or null")
        if set(value) != {"parameters", "spec"}:
            raise ValueError("perturbation metadata has missing or unknown fields")
        raw_spec = value["spec"]
        raw_parameters = value["parameters"]
        if not isinstance(raw_spec, Mapping) or set(raw_spec) != {
            "family",
            "severity",
            "variant",
            "realization_seed",
        }:
            raise ValueError("perturbation spec binding is incomplete")
        if not isinstance(raw_parameters, Mapping):
            raise ValueError("realized perturbation parameters must be a mapping")
        spec = dict(raw_spec)
        parameters = dict(raw_parameters)
        # Canonical JSON validation rejects tensors and other non-metadata objects.
        json.loads(_canonical_json({"parameters": parameters, "spec": spec}).decode("utf-8"))
        seed = spec["realization_seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("perturbation realization seed is invalid")
        snapshot[view] = MappingProxyType(
            {
                "parameters": MappingProxyType(parameters),
                "spec": MappingProxyType(dict(spec)),
            }
        )
    return MappingProxyType(snapshot)


@dataclass(frozen=True)
class CacheProvenance:
    """Exact parent, batch order, role, model, transform, and schema binding."""

    dataset_namespace: str
    role: str
    operation: str
    manifest_sha256: str
    sample_keys: tuple[str, ...]
    exam_keys: tuple[str, ...]
    patient_keys: tuple[str, ...]
    observed_views: tuple[str, ...]
    observed_mask: tuple[bool, ...]
    fusion_pairs: FusionPairs
    perturbations: Mapping[str, Mapping[str, object] | None]
    parent_identity_sha256: str
    checkpoint_sha256: str
    backbone: str
    hf_model: str
    backbone_revision: str
    image_size: int
    hidden_size: int
    preprocessing: str
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    prompt_order: tuple[str, ...]
    text_input_sha256: str
    implementation_revision: str
    schema_version: str = CACHE_SCHEMA_VERSION
    target_version: str = TARGET_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dataset_namespace", _required_string(self.dataset_namespace, "dataset_namespace")
        )
        try:
            role = Role(self.role)
        except (TypeError, ValueError) as exc:
            raise ValueError("role is invalid") from exc
        if role == Role.LOCKED_TEST:
            raise PermissionError("locked_test cannot be bound to an ordinary cache")
        try:
            operation = Operation(self.operation)
        except (TypeError, ValueError) as exc:
            raise ValueError("operation is invalid") from exc
        if operation not in _CACHE_OPERATIONS:
            raise PermissionError("operation is not authorized for target-cache construction")
        object.__setattr__(self, "role", role.value)
        object.__setattr__(self, "operation", operation.value)
        object.__setattr__(
            self, "manifest_sha256", _require_sha256(self.manifest_sha256, "manifest_sha256")
        )
        sample_keys = _private_keys(self.sample_keys, "sample_keys")
        exam_keys = _private_keys(self.exam_keys, "exam_keys")
        patient_keys = _private_keys(self.patient_keys, "patient_keys")
        if not (len(sample_keys) == len(exam_keys) == len(patient_keys)):
            raise ValueError("ordered private keys must have the same batch length")
        object.__setattr__(self, "sample_keys", sample_keys)
        object.__setattr__(self, "exam_keys", exam_keys)
        object.__setattr__(self, "patient_keys", patient_keys)
        observed = canonicalize_observed_views(self.observed_views)
        if tuple(self.observed_views) != observed:
            raise ValueError("observed_views must use canonical order")
        mask = tuple(self.observed_mask)
        expected_mask = tuple(view in observed for view in CANONICAL_VIEWS)
        if mask != expected_mask or any(not isinstance(value, bool) for value in mask):
            raise ValueError("observed_mask disagrees with observed_views")
        object.__setattr__(self, "observed_views", observed)
        object.__setattr__(self, "observed_mask", mask)
        object.__setattr__(self, "fusion_pairs", validate_fusion_pairs(self.fusion_pairs))
        object.__setattr__(
            self, "perturbations", _snapshot_perturbations(self.perturbations, observed)
        )
        for field_name in (
            "parent_identity_sha256",
            "checkpoint_sha256",
            "text_input_sha256",
        ):
            object.__setattr__(
                self, field_name, _require_sha256(getattr(self, field_name), field_name)
            )
        for field_name in (
            "backbone",
            "hf_model",
            "backbone_revision",
            "preprocessing",
            "implementation_revision",
        ):
            object.__setattr__(
                self, field_name, _required_string(getattr(self, field_name), field_name)
            )
        if (
            isinstance(self.image_size, bool)
            or not isinstance(self.image_size, int)
            or self.image_size < 1
        ):
            raise ValueError("image_size must be a positive integer")
        if (
            isinstance(self.hidden_size, bool)
            or not isinstance(self.hidden_size, int)
            or self.hidden_size < 1
        ):
            raise ValueError("hidden_size must be a positive integer")
        means = tuple(float(value) for value in self.image_mean)
        stds = tuple(float(value) for value in self.image_std)
        if len(means) != 3 or len(stds) != 3 or any(value <= 0 for value in stds):
            raise ValueError("image normalization must contain three channels and positive stds")
        object.__setattr__(self, "image_mean", means)
        object.__setattr__(self, "image_std", stds)
        prompts = tuple(_required_string(value, "prompt_order") for value in self.prompt_order)
        if len(prompts) != NUM_CLASSES:
            raise ValueError("prompt_order must bind four class prompts")
        object.__setattr__(self, "prompt_order", prompts)
        if self.schema_version != CACHE_SCHEMA_VERSION:
            raise ValueError("unsupported cache schema version")
        if self.target_version != TARGET_VERSION:
            raise ValueError("unsupported target version")

    @property
    def batch_size(self) -> int:
        return len(self.sample_keys)

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_namespace": self.dataset_namespace,
            "role": self.role,
            "operation": self.operation,
            "manifest_sha256": self.manifest_sha256,
            "sample_keys": list(self.sample_keys),
            "exam_keys": list(self.exam_keys),
            "patient_keys": list(self.patient_keys),
            "observed_views": list(self.observed_views),
            "observed_mask": list(self.observed_mask),
            "fusion_pairs": [list(pair) for pair in self.fusion_pairs],
            "perturbations": {
                view: None
                if self.perturbations[view] is None
                else {
                    "parameters": dict(self.perturbations[view]["parameters"]),  # type: ignore[index]
                    "spec": dict(self.perturbations[view]["spec"]),  # type: ignore[index]
                }
                for view in self.observed_views
            },
            "parent_identity_sha256": self.parent_identity_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "backbone": self.backbone,
            "hf_model": self.hf_model,
            "backbone_revision": self.backbone_revision,
            "image_size": self.image_size,
            "hidden_size": self.hidden_size,
            "preprocessing": self.preprocessing,
            "image_mean": list(self.image_mean),
            "image_std": list(self.image_std),
            "prompt_order": list(self.prompt_order),
            "text_input_sha256": self.text_input_sha256,
            "implementation_revision": self.implementation_revision,
            "schema_version": self.schema_version,
            "target_version": self.target_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CacheProvenance":
        if not isinstance(value, dict):
            raise ArtifactIntegrityError("cache provenance must be a JSON object")
        expected = {field.name for field in fields(cls)}
        if set(value) != expected:
            raise ArtifactIntegrityError("cache provenance has missing or unknown fields")
        try:
            return cls(**value)  # type: ignore[arg-type]
        except (TypeError, ValueError, PermissionError) as exc:
            raise ArtifactIntegrityError(f"invalid cache provenance: {exc}") from exc


@dataclass(frozen=True)
class CachedTargets:
    """Label-supervised outputs kept separate from label-free inference features."""

    labels: Tensor
    observed_prediction: Tensor
    observed_error: Tensor
    omission_predictions: Tensor
    omission_effects: Tensor
    omission_labels: Tensor
    valid_removal_mask: Tensor
    scores: Tensor
    probabilities: Tensor
    tcp: Tensor


_CACHE_BUNDLE_TOKEN = object()


@dataclass(frozen=True, init=False)
class CacheBundle:
    provenance: CacheProvenance
    features: FrozenViewFeatures
    targets: CachedTargets

    def __init__(
        self,
        provenance: CacheProvenance,
        features: FrozenViewFeatures,
        targets: CachedTargets,
        *,
        _factory_token: object,
    ) -> None:
        if _factory_token is not _CACHE_BUNDLE_TOKEN:
            raise RuntimeError("CacheBundle must be built from regenerated or validated targets")
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "targets", targets)


@dataclass(frozen=True)
class CachePaths:
    metadata: Path
    tensors: Path


def _encoder_provenance(identity: FrozenEncoderIdentity) -> dict[str, object]:
    return identity.to_dict()


def _parent_identity(parent: RealizedParent) -> str:
    return _hash_json(
        [
            {
                "view": view,
                "tensor_sha256": tensor_sha256(parent.images[view]),
            }
            for view in parent.observed_views
        ]
    )


def _parent_perturbations(
    parent: RealizedParent,
) -> Mapping[str, Mapping[str, object] | None]:
    clean = {"spec": PerturbationSpec("clean").to_dict(), "parameters": {}}
    return {
        view: clean if view not in parent.metadata else parent.metadata[view].to_dict()
        for view in parent.observed_views
    }


def _validate_realized_perturbation_parameters(parent: RealizedParent) -> None:
    """Validate recorded transform parameters, not caller-authored pixel authenticity."""

    for view, metadata in parent.metadata.items():
        image = parent.images[view]
        if (
            not isinstance(image, Tensor)
            or image.ndim != 3
            or image.shape[0] != 3
            or image.shape[1] != image.shape[2]
        ):
            raise ValueError(
                "realized perturbation parameters require a square RGB parent image"
            )
        try:
            entries = tuple(metadata.parameters)
        except TypeError as exc:
            raise ValueError("realized perturbation parameters must be key/value pairs") from exc
        if any(not isinstance(entry, tuple) or len(entry) != 2 for entry in entries):
            raise ValueError("realized perturbation parameters must be key/value pairs")
        keys = tuple(entry[0] for entry in entries)
        if any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("realized perturbation parameters require unique string fields")
        actual = dict(entries)
        expected = resolve_parameters(metadata.spec, int(image.shape[-1]))
        extra_fields = {"top", "left"} if metadata.spec.family == "crop" else set()
        if set(actual) != set(expected) | extra_fields:
            raise ValueError(
                "realized perturbation parameters have missing or unknown fields"
            )
        for field, expected_value in expected.items():
            actual_value = actual[field]
            if type(actual_value) is not type(expected_value) or actual_value != expected_value:
                raise ValueError(
                    f"realized perturbation parameters disagree with spec: {field}"
                )
        if metadata.spec.family == "crop":
            maximum_offset = int(image.shape[-1]) - int(expected["crop_side"])
            for field in ("top", "left"):
                value = actual[field]
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(
                        f"realized perturbation parameters require integer {field}"
                    )
                if value < 0 or value > maximum_offset:
                    raise ValueError(
                        f"realized perturbation parameters contain out-of-bounds {field}"
                    )


def _provenance_for_exam(
    *,
    manifest: RoleManifest,
    record: PrivateExamRecord,
    role: Role,
    operation: Operation,
    sample_key: str,
    parent: RealizedParent,
    features: FrozenViewFeatures,
    identity: FrozenEncoderIdentity,
    implementation_revision: str,
) -> CacheProvenance:
    encoder = _encoder_provenance(identity)
    return CacheProvenance(
        dataset_namespace=manifest.dataset_namespace,
        role=role.value,
        operation=operation.value,
        manifest_sha256=manifest.manifest_sha256,
        sample_keys=(sample_key,),
        exam_keys=(record.exam_key,),
        patient_keys=(record.patient_key,),
        observed_views=features.observed_views,
        observed_mask=tuple(view in features.observed_views for view in CANONICAL_VIEWS),
        fusion_pairs=features.fusion_pairs,
        perturbations=_parent_perturbations(parent),
        parent_identity_sha256=_parent_identity(parent),
        checkpoint_sha256=str(encoder["checkpoint_sha256"]),
        backbone=str(encoder["backbone"]),
        hf_model=str(encoder["hf_model"]),
        backbone_revision=str(encoder["backbone_revision"]),
        image_size=int(encoder["image_size"]),
        hidden_size=int(encoder["hidden_size"]),
        preprocessing=str(encoder["preprocessing"]),
        image_mean=tuple(encoder["image_mean"]),  # type: ignore[arg-type]
        image_std=tuple(encoder["image_std"]),  # type: ignore[arg-type]
        prompt_order=tuple(encoder["prompt_order"]),  # type: ignore[arg-type]
        text_input_sha256=str(encoder["text_input_sha256"]),
        implementation_revision=implementation_revision,
    )


def _build_cache_bundle(
    features: FrozenViewFeatures,
    labels: Tensor,
    provenance: CacheProvenance,
) -> CacheBundle:
    """Regenerate target tensors from supplied realized logits and labels only."""

    if not isinstance(features, FrozenViewFeatures):
        raise TypeError("features must be FrozenViewFeatures")
    if not isinstance(provenance, CacheProvenance):
        raise TypeError("provenance must be CacheProvenance")
    if provenance.batch_size != features.batch_size:
        raise ValueError("provenance private-key order must match the feature batch")
    if provenance.observed_views != features.observed_views:
        raise ValueError("provenance observed mask disagrees with features")
    if provenance.fusion_pairs != features.fusion_pairs:
        raise ValueError("provenance fusion tree disagrees with features")
    generated = build_intervention_targets(
        features.logits_by_view,
        labels,
        features.observed_views,
        fusion_pairs=features.fusion_pairs,
    )
    probabilities = torch.softmax(features.scores, dim=1)
    tcp = probabilities.gather(1, labels[:, None]).squeeze(1)
    targets = CachedTargets(
        labels=labels.detach(),
        observed_prediction=generated.observed_prediction.detach(),
        observed_error=generated.observed_error.detach(),
        omission_predictions=generated.omission_predictions.detach(),
        omission_effects=generated.omission_effects.detach(),
        omission_labels=generated.omission_labels.detach(),
        valid_removal_mask=generated.valid_removal_mask.detach(),
        scores=features.scores.detach(),
        probabilities=probabilities.detach(),
        tcp=tcp.detach(),
    )
    bundle = CacheBundle(
        provenance=provenance,
        features=features,
        targets=targets,
        _factory_token=_CACHE_BUNDLE_TOKEN,
    )
    _validate_bundle(bundle)
    return bundle


def build_exam_cache_with_role_access(
    manifest: RoleManifest,
    *,
    operation: Operation,
    role: Role,
    exam_key: str,
    sample_key: str,
    parent_loader: Callable[[PrivateExamRecord], RealizedParent],
    encoder: VerifiedFrozenEncoder,
    implementation_revision: str,
) -> CacheBundle:
    """Authorize one exam before loading its parent or invoking its frozen model."""

    try:
        normalized_operation = Operation(operation)
    except (TypeError, ValueError) as exc:
        raise PermissionError("unknown research operation") from exc
    if normalized_operation not in _CACHE_OPERATIONS:
        raise PermissionError("operation is not authorized for target-cache construction")
    try:
        normalized_role = Role(role)
    except (TypeError, ValueError) as exc:
        raise PermissionError("unknown research role") from exc
    key = _required_string(exam_key, "exam_key")
    private_sample_key = _required_string(sample_key, "sample_key")

    def authorized(records: tuple[PrivateExamRecord, ...]) -> CacheBundle:
        matching = tuple(record for record in records if record.exam_key == key)
        if len(matching) != 1:
            raise ValueError("authorized manifest must contain exactly one requested exam")
        record = matching[0]
        parent = parent_loader(record)
        if not isinstance(parent, RealizedParent):
            raise TypeError("parent_loader must return a RealizedParent")
        parent_views = canonicalize_observed_views(parent.images)
        if tuple(parent.images) != parent_views:
            raise ValueError("realized parent views must use canonical order")
        if not set(parent.metadata).issubset(parent_views):
            raise ValueError("realized parent metadata cannot describe an absent view")
        if any(
            not isinstance(metadata, PerturbationMetadata) for metadata in parent.metadata.values()
        ):
            raise TypeError("realized parent metadata must contain PerturbationMetadata")
        _validate_realized_perturbation_parameters(parent)
        if normalized_operation in {
            Operation.CONFIDENCE_FITTING,
            Operation.TUNE_SELECTION,
        }:
            perturbation_operation = {
                Operation.CONFIDENCE_FITTING: "confidence-fit",
                Operation.TUNE_SELECTION: "tune",
            }[normalized_operation]
            for metadata in parent.metadata.values():
                validate_training_perturbation(metadata.spec, operation=perturbation_operation)
        if not isinstance(encoder, VerifiedFrozenEncoder):
            raise TypeError("encoder must come from load_verified_frozen_encoder")
        features = encoder.extract_realized_parent(parent)
        provenance = _provenance_for_exam(
            manifest=manifest,
            record=record,
            role=normalized_role,
            operation=normalized_operation,
            sample_key=private_sample_key,
            parent=parent,
            features=features,
            identity=encoder.identity,
            implementation_revision=implementation_revision,
        )
        labels = torch.tensor([record.density], dtype=torch.long, device=features.scores.device)
        return _build_cache_bundle(features, labels, provenance)

    return run_with_role_access(
        manifest,
        operation=normalized_operation,
        roles=(normalized_role,),
        loader=authorized,
    )


def _expected_target_tensors(targets: CachedTargets) -> dict[str, Tensor]:
    return {
        "target.labels": targets.labels,
        "target.observed_prediction": targets.observed_prediction,
        "target.observed_error": targets.observed_error,
        "target.omission_predictions": targets.omission_predictions,
        "target.omission_effects": targets.omission_effects,
        "target.omission_labels": targets.omission_labels,
        "target.valid_removal_mask": targets.valid_removal_mask,
        "target.scores": targets.scores,
        "target.probabilities": targets.probabilities,
        "target.tcp": targets.tcp,
    }


def _bundle_tensors(bundle: CacheBundle) -> dict[str, Tensor]:
    features = bundle.features
    result: dict[str, Tensor] = {
        "feature.normalized_text_embeddings": features.normalized_text_embeddings,
    }
    for view in features.observed_views:
        result[f"feature.logits.{view}"] = features.logits_by_view[view]
        result[f"feature.hidden.{view}"] = features.hidden_by_view[view]
        result[f"feature.projected.{view}"] = features.projected_by_view[view]
    result.update(_expected_target_tensors(bundle.targets))
    return result


def _tensor_manifest(tensors: Mapping[str, Tensor]) -> dict[str, dict[str, object]]:
    return {
        name: {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": tensor_sha256(tensor),
        }
        for name, tensor in sorted(tensors.items())
    }


def _selected_tensor_hash(tensors: Mapping[str, Tensor], prefix: str) -> str:
    return _hash_json(
        [
            [name, tensor_sha256(tensors[name])]
            for name in sorted(tensors)
            if name.startswith(prefix)
        ]
    )


def _same_tensor(actual: Tensor, expected: Tensor, field: str) -> None:
    if (
        actual.dtype != expected.dtype
        or actual.shape != expected.shape
        or not torch.equal(actual, expected)
    ):
        raise ArtifactIntegrityError(f"cached {field} is stale or inconsistent")


def _validate_bundle(bundle: CacheBundle) -> None:
    if not isinstance(bundle.provenance, CacheProvenance):
        raise TypeError("bundle provenance is invalid")
    if not isinstance(bundle.features, FrozenViewFeatures):
        raise TypeError("bundle features are invalid")
    if not isinstance(bundle.targets, CachedTargets):
        raise TypeError("bundle targets are invalid")
    provenance = bundle.provenance
    features = bundle.features
    targets = bundle.targets
    if provenance.batch_size != features.batch_size:
        raise ArtifactIntegrityError("ordered private keys disagree with feature batch")
    if provenance.observed_views != features.observed_views:
        raise ArtifactIntegrityError("observed mask disagrees with cached features")
    if provenance.fusion_pairs != features.fusion_pairs:
        raise ArtifactIntegrityError("fusion tree disagrees with cached features")
    if any(
        features.hidden_by_view[view].shape[2] != provenance.hidden_size
        for view in features.observed_views
    ):
        raise ArtifactIntegrityError("hidden feature width disagrees with backbone provenance")
    if targets.labels.device != features.scores.device:
        raise ArtifactIntegrityError("labels and inference tensors must share a device")
    generated: InterventionTargets
    try:
        generated = build_intervention_targets(
            features.logits_by_view,
            targets.labels,
            features.observed_views,
            fusion_pairs=features.fusion_pairs,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(f"cached labels or logits are invalid: {exc}") from exc
    _same_tensor(targets.observed_prediction, generated.observed_prediction, "observed_prediction")
    _same_tensor(targets.observed_error, generated.observed_error, "observed_error")
    _same_tensor(
        targets.omission_predictions, generated.omission_predictions, "omission_predictions"
    )
    _same_tensor(targets.omission_effects, generated.omission_effects, "omission_effects")
    _same_tensor(targets.omission_labels, generated.omission_labels, "omission_labels")
    _same_tensor(targets.valid_removal_mask, generated.valid_removal_mask, "valid_removal_mask")
    fused = fuse_view_logits(
        features.logits_by_view,
        features.observed_views,
        fusion_pairs=features.fusion_pairs,
    )
    _same_tensor(features.scores, fused.scores, "feature scores")
    _same_tensor(targets.scores, fused.scores, "scores")
    probabilities = torch.softmax(fused.scores, dim=1)
    _same_tensor(features.probabilities, probabilities, "feature probabilities")
    _same_tensor(targets.probabilities, probabilities, "probabilities")
    tcp = probabilities.gather(1, targets.labels[:, None]).squeeze(1)
    _same_tensor(targets.tcp, tcp, "tcp")


def _write_atomic_json(path: Path, value: Mapping[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json(value))
            handle.write(b"\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _require_external_or_ignored_destination(path: Path) -> None:
    resolved = path.resolve()
    repository: Path | None = None
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / ".git").exists():
            repository = candidate
            break
    if repository is None:
        return
    check = subprocess.run(
        ["git", "-C", str(repository), "check-ignore", "--quiet", "--", str(resolved)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if check.returncode != 0:
        raise ValueError(
            "private cache destination inside a worktree must be covered by .gitignore"
        )


def save_cache_bundle(bundle: CacheBundle, metadata_path: str | Path) -> CachePaths:
    """Write a cache only to an explicit caller-supplied JSON destination."""

    from safetensors.torch import save_file

    if not isinstance(bundle, CacheBundle):
        raise TypeError("bundle must be a CacheBundle")
    _validate_bundle(bundle)
    metadata = Path(metadata_path).resolve()
    if metadata.suffix != ".json":
        raise ValueError("cache metadata destination must end in .json")
    if not metadata.parent.exists():
        raise ValueError("caller-supplied cache destination directory must already exist")
    tensors_path = metadata.with_suffix(".safetensors")
    _require_external_or_ignored_destination(metadata)
    _require_external_or_ignored_destination(tensors_path)
    tensors = {
        name: tensor.detach().cpu().contiguous() for name, tensor in _bundle_tensors(bundle).items()
    }
    temporary = tensors_path.with_name(f".{tensors_path.name}.{os.getpid()}.tmp")
    try:
        save_file(tensors, str(temporary))
        os.replace(temporary, tensors_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    manifest = _tensor_manifest(tensors)
    document: dict[str, object] = {
        "provenance": bundle.provenance.to_dict(),
        "tensor_file": tensors_path.name,
        "tensor_file_sha256": sha256_file(tensors_path),
        "tensor_manifest": manifest,
        "feature_content_sha256": _selected_tensor_hash(tensors, "feature."),
        "labels_sha256": tensor_sha256(tensors["target.labels"]),
    }
    document["record_integrity_sha256"] = _hash_json(document)
    _write_atomic_json(metadata, document)
    return CachePaths(metadata=metadata, tensors=tensors_path)


def _read_document(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError("cache metadata is unreadable") from exc
    expected = {
        "provenance",
        "tensor_file",
        "tensor_file_sha256",
        "tensor_manifest",
        "feature_content_sha256",
        "labels_sha256",
        "record_integrity_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ArtifactIntegrityError("cache metadata has missing or unknown fields")
    integrity = value.get("record_integrity_sha256")
    unsigned = dict(value)
    unsigned.pop("record_integrity_sha256")
    if not _is_sha256(integrity) or integrity != _hash_json(unsigned):
        raise ArtifactIntegrityError("cache metadata integrity check failed")
    return value


def _verify_expected(actual: CacheProvenance, expected: CacheProvenance) -> None:
    for field in fields(CacheProvenance):
        name = field.name
        if getattr(actual, name) != getattr(expected, name):
            raise ProvenanceMismatchError(f"expected provenance mismatch: {name}")


def _target_from_tensors(tensors: Mapping[str, Tensor]) -> CachedTargets:
    return CachedTargets(
        labels=tensors["target.labels"],
        observed_prediction=tensors["target.observed_prediction"],
        observed_error=tensors["target.observed_error"],
        omission_predictions=tensors["target.omission_predictions"],
        omission_effects=tensors["target.omission_effects"],
        omission_labels=tensors["target.omission_labels"],
        valid_removal_mask=tensors["target.valid_removal_mask"],
        scores=tensors["target.scores"],
        probabilities=tensors["target.probabilities"],
        tcp=tensors["target.tcp"],
    )


def _load_cache_bundle_authorized(
    metadata_path: str | Path,
    *,
    expected_provenance: CacheProvenance,
    authorized_densities: tuple[int, ...],
) -> CacheBundle:
    """Load after the public entry point has authorized the requested role."""

    from safetensors.torch import load_file

    if not isinstance(expected_provenance, CacheProvenance):
        raise TypeError("expected_provenance must be supplied as CacheProvenance")
    metadata = Path(metadata_path)
    document = _read_document(metadata)
    actual = CacheProvenance.from_dict(document["provenance"])
    _verify_expected(actual, expected_provenance)
    tensor_file = document["tensor_file"]
    if not isinstance(tensor_file, str) or Path(tensor_file).name != tensor_file:
        raise ArtifactIntegrityError("cache tensor filename is unsafe")
    tensors_path = metadata.parent / tensor_file
    try:
        actual_file_sha256 = sha256_file(tensors_path)
    except OSError as exc:
        raise ArtifactIntegrityError("cache tensor file is unavailable") from exc
    if document["tensor_file_sha256"] != actual_file_sha256:
        raise ArtifactIntegrityError("cache tensor file integrity check failed")
    try:
        tensors = load_file(str(tensors_path), device="cpu")
    except Exception as exc:
        raise ArtifactIntegrityError("cache safetensors file is unreadable") from exc
    raw_manifest = document["tensor_manifest"]
    if not isinstance(raw_manifest, dict) or set(raw_manifest) != set(tensors):
        raise ArtifactIntegrityError("cache tensor set is missing or unexpected")
    for name, tensor in tensors.items():
        record = raw_manifest[name]
        if not isinstance(record, dict) or set(record) != {"dtype", "shape", "sha256"}:
            raise ArtifactIntegrityError(f"invalid tensor manifest entry for {name}")
        if record["dtype"] != str(tensor.dtype) or record["shape"] != list(tensor.shape):
            raise ArtifactIntegrityError(f"tensor shape/dtype mismatch for {name}")
        if record["sha256"] != tensor_sha256(tensor):
            raise ArtifactIntegrityError(f"tensor content hash mismatch for {name}")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ArtifactIntegrityError(f"tensor must be finite: {name}")
    if document["feature_content_sha256"] != _selected_tensor_hash(tensors, "feature."):
        raise ArtifactIntegrityError("feature tensor content hash mismatch")
    if document["labels_sha256"] != tensor_sha256(tensors["target.labels"]):
        raise ArtifactIntegrityError("labels content hash mismatch")

    expected_names = {"feature.normalized_text_embeddings"}
    for view in actual.observed_views:
        expected_names.update(
            {
                f"feature.logits.{view}",
                f"feature.hidden.{view}",
                f"feature.projected.{view}",
            }
        )
    target_names = {
        "target.labels",
        "target.observed_prediction",
        "target.observed_error",
        "target.omission_predictions",
        "target.omission_effects",
        "target.omission_labels",
        "target.valid_removal_mask",
        "target.scores",
        "target.probabilities",
        "target.tcp",
    }
    if set(tensors) != expected_names | target_names:
        raise ArtifactIntegrityError(
            "cache tensor set disagrees with observed view representations"
        )
    targets = _target_from_tensors(tensors)
    authorized_labels = torch.tensor(authorized_densities, dtype=torch.long)
    if (
        targets.labels.dtype != authorized_labels.dtype
        or targets.labels.shape != authorized_labels.shape
        or not torch.equal(targets.labels, authorized_labels)
    ):
        raise ArtifactIntegrityError(
            "cached labels disagree with authorized manifest densities or order"
        )
    try:
        features = FrozenViewFeatures(
            logits_by_view={
                view: tensors[f"feature.logits.{view}"] for view in actual.observed_views
            },
            hidden_by_view={
                view: tensors[f"feature.hidden.{view}"] for view in actual.observed_views
            },
            projected_by_view={
                view: tensors[f"feature.projected.{view}"] for view in actual.observed_views
            },
            normalized_text_embeddings=tensors["feature.normalized_text_embeddings"],
            scores=targets.scores,
            probabilities=targets.probabilities,
            observed_views=actual.observed_views,
            fusion_pairs=actual.fusion_pairs,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(f"invalid cached feature tensors: {exc}") from exc
    bundle = CacheBundle(
        provenance=actual,
        features=features,
        targets=targets,
        _factory_token=_CACHE_BUNDLE_TOKEN,
    )
    _validate_bundle(bundle)
    return bundle


def load_cache_bundle(
    metadata_path: str | Path,
    *,
    expected_provenance: CacheProvenance,
    manifest: RoleManifest,
    operation: Operation,
    role: Role,
) -> CacheBundle:
    """Authorize role and ordered exam identities before reading private cache files."""

    if not isinstance(expected_provenance, CacheProvenance):
        raise TypeError("expected_provenance must be supplied as CacheProvenance")
    try:
        normalized_operation = Operation(operation)
    except (TypeError, ValueError) as exc:
        raise PermissionError("unknown research operation") from exc
    if normalized_operation not in _CACHE_OPERATIONS:
        raise PermissionError("operation is not authorized for target-cache loading")
    try:
        normalized_role = Role(role)
    except (TypeError, ValueError) as exc:
        raise PermissionError("unknown research role") from exc

    def authorized(records: tuple[PrivateExamRecord, ...]) -> CacheBundle:
        if expected_provenance.dataset_namespace != manifest.dataset_namespace:
            raise ProvenanceMismatchError("expected provenance mismatch: dataset_namespace")
        if expected_provenance.manifest_sha256 != manifest.manifest_sha256:
            raise ProvenanceMismatchError("expected provenance mismatch: manifest_sha256")
        if expected_provenance.role != normalized_role.value:
            raise ProvenanceMismatchError("expected provenance mismatch: role")
        if expected_provenance.operation != normalized_operation.value:
            raise ProvenanceMismatchError("expected provenance mismatch: operation")
        by_exam = {record.exam_key: record for record in records}
        if any(exam_key not in by_exam for exam_key in expected_provenance.exam_keys):
            raise ProvenanceMismatchError("expected provenance mismatch: exam_keys")
        ordered_records = tuple(by_exam[key] for key in expected_provenance.exam_keys)
        patients = tuple(record.patient_key for record in ordered_records)
        if patients != expected_provenance.patient_keys:
            raise ProvenanceMismatchError("expected provenance mismatch: patient_keys")
        return _load_cache_bundle_authorized(
            metadata_path,
            expected_provenance=expected_provenance,
            authorized_densities=tuple(record.density for record in ordered_records),
        )

    return run_with_role_access(
        manifest,
        operation=normalized_operation,
        roles=(normalized_role,),
        loader=authorized,
    )
