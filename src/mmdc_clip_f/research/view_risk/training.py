"""Role-bound training orchestration for the view-risk study.

This module deliberately consumes injected readers and modules.  Importing it
does not open a manifest, image, cache, checkpoint, or remote model.  The public
entry points authorize a role before invoking those hooks, which keeps the same
code usable for tiny software fixtures and later, separately audited P5 runs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from mmdc_clip_f.backbones import PROMPTS, get_backbone
from mmdc_clip_f.provenance import sha256_file

from .baselines import (
    BASELINE_DEFINITIONS,
    MaskedMVACNAdapter,
    MVACNObjective,
    SameInputDensityControl,
    SameInputMLP,
    ViLUFailureAdapter,
    compute_density_control_loss,
    compute_mvacn_objective,
    compute_same_input_error_loss,
    compute_vilu_failure_loss,
    p3a_ablation_definitions,
)
from .cache import (
    CacheBundle,
    CachedTargets,
    _require_external_or_ignored_destination,
)
from .features import (
    LEGACY_IMAGE_MEAN,
    LEGACY_IMAGE_STD,
    LEGACY_PREPROCESSING,
    FrozenViewFeatures,
    VerifiedFrozenEncoder,
    tensor_sha256,
)
from .fusion import DDSM_FUSION_PAIRS, FusionPairs, RSNA_FUSION_PAIRS
from .head import (
    RelationAwareConfidenceHead,
    RelationAwareHeadConfig,
    compute_view_risk_loss,
    count_trainable_parameters,
)
from .head_inputs import prepare_raw_head_inputs
from .inputs import CANONICAL_VIEWS
from .perturbations import PerturbationSpec, TrainingSampleSpec, sample_training_spec
from .roles import (
    NON_TEST_ROLES,
    OPERATION_ROLE_ALLOWLIST,
    LockedPatientIdentityDenylist,
    ManifestBinding,
    Operation,
    PrivateExamRecord,
    Role,
    RoleManifest,
    load_private_manifest,
    run_with_role_access,
    validate_inventory,
)
from .targets import InterventionTargets, build_intervention_targets


RUN_CONFIG_VERSION = "view-risk-run-config/v1"
SEARCH_TABLE_VERSION = "view-risk-search-table/v1"
TRAINING_SCHEDULE_VERSION = "view-risk-training-schedule/v1"
RESUME_VERSION = "view-risk-resume/v1"
METRIC_VERSION = "view-risk-p4a-metrics/v1"
DEFAULT_PROTOCOL_SHA256 = "4231c168d6dc21d0b33e02fe4adb247af6c8031a5c2eb8bace992b12b2eff435"

ABLATION_METHODS = (
    "no_effect_supervision",
    "no_intervention_features",
    "no_view_relations",
    "magnitude_effects",
    "corruption_auxiliary",
    "hidden_only",
    "evidence_only",
    "clean_fitting",
)
MANDATORY_METHODS = ("candidate", *tuple(BASELINE_DEFINITIONS), *ABLATION_METHODS)
LEARNED_TORCH_METHODS = frozenset(
    {
        "candidate",
        "original_mvacn_tcp",
        "correctness_mvacn",
        "matched_mvacn_tcp",
        "matched_mvacn_correctness",
        "vilu",
        "same_input_mlp",
        "same_input_density",
        *ABLATION_METHODS,
    }
)
_FRESH_CLASSIFIER_FIT_TOKEN = object()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adam"
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 6

    def __post_init__(self) -> None:
        if self.name != "adam":
            raise ValueError("optimizer must be the protocol Adam optimizer")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not math.isfinite(self.learning_rate)
            or not self.learning_rate > 0
        ):
            raise ValueError("optimizer learning_rate must be positive")
        if (
            isinstance(self.weight_decay, bool)
            or not isinstance(self.weight_decay, (int, float))
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("optimizer weight_decay must be nonnegative")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise ValueError("optimizer batch_size must be an integer")
        if self.batch_size < 1:
            raise ValueError("optimizer batch_size must be positive")


@dataclass(frozen=True)
class FrozenSearchTable:
    checkpoint_epochs: tuple[int, ...]
    ds_regularizations: tuple[float, ...] = (0.0, 1e-4, 1e-3, 1e-2)
    architecture_trials_per_method: int = 1
    version: str = SEARCH_TABLE_VERSION

    def __post_init__(self) -> None:
        if self.version != SEARCH_TABLE_VERSION:
            raise ValueError("unsupported search table version")
        epochs = tuple(self.checkpoint_epochs)
        if (
            not epochs
            or any(isinstance(value, bool) or not isinstance(value, int) for value in epochs)
            or epochs != tuple(range(1, len(epochs) + 1))
        ):
            raise ValueError("checkpoint search epochs must be the finite consecutive table")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in self.ds_regularizations
        ):
            raise ValueError("DS regularizations must be numeric")
        regularizations = tuple(float(value) for value in self.ds_regularizations)
        if (
            not regularizations
            or len(set(regularizations)) != len(regularizations)
            or any(not np.isfinite(value) or value < 0 for value in regularizations)
        ):
            raise ValueError("DS regularization search must be finite, unique, and nonnegative")
        if (
            isinstance(self.architecture_trials_per_method, bool)
            or not isinstance(self.architecture_trials_per_method, int)
            or self.architecture_trials_per_method != 1
        ):
            raise ValueError("every learned method has one frozen architecture trial")
        object.__setattr__(self, "checkpoint_epochs", epochs)
        object.__setattr__(self, "ds_regularizations", regularizations)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "checkpoint_epochs": list(self.checkpoint_epochs),
            "ds_regularizations": list(self.ds_regularizations),
            "architecture_trials_per_method": self.architecture_trials_per_method,
        }

    @property
    def sha256(self) -> str:
        return _sha256_json(self.to_dict())


@dataclass(frozen=True)
class ResearchRunConfig:
    dataset: str
    backbone: str
    optimizer: OptimizerConfig
    epochs: int
    seeds: tuple[int, ...]
    methods: tuple[str, ...]
    stress_seed: int
    metric_version: str
    protocol_sha256: str
    search_table: FrozenSearchTable
    schema_version: str = RUN_CONFIG_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RUN_CONFIG_VERSION:
            raise ValueError("unsupported research run configuration version")
        if not isinstance(self.dataset, str) or not isinstance(self.backbone, str):
            raise ValueError("dataset and backbone must be strings")
        dataset = self.dataset.upper()
        if dataset not in ("RSNA", "DDSM"):
            raise ValueError("dataset must be RSNA or DDSM")
        spec = get_backbone(self.backbone)
        expected_epochs = 20 if dataset == "RSNA" else 50
        if (
            isinstance(self.epochs, bool)
            or not isinstance(self.epochs, int)
            or self.epochs != expected_epochs
        ):
            raise ValueError(f"{dataset} confidence training must use {expected_epochs} epochs")
        if tuple(self.seeds) != (42, 43, 44):
            raise ValueError("training seeds must be the frozen tuple (42, 43, 44)")
        if tuple(self.methods) != MANDATORY_METHODS:
            raise ValueError("methods must contain every mandatory control and ablation")
        if (
            isinstance(self.stress_seed, bool)
            or not isinstance(self.stress_seed, int)
            or self.stress_seed != 4242
        ):
            raise ValueError("stress_seed must be the frozen evaluation seed 4242")
        if self.metric_version != METRIC_VERSION:
            raise ValueError("metric_version is not the accepted P4A binding")
        if not _is_sha256(self.protocol_sha256):
            raise ValueError("protocol_sha256 must be a lowercase SHA-256 digest")
        if self.search_table.checkpoint_epochs != tuple(range(1, expected_epochs + 1)):
            raise ValueError("search table checkpoint budget does not match the dataset")
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "backbone", spec.name)

    @classmethod
    def default(
        cls,
        dataset: str,
        backbone: str = "vit_b_32",
        *,
        protocol_sha256: str = DEFAULT_PROTOCOL_SHA256,
    ) -> "ResearchRunConfig":
        normalized = dataset.upper()
        epochs = 20 if normalized == "RSNA" else 50 if normalized == "DDSM" else 0
        return cls(
            dataset=normalized,
            backbone=backbone,
            optimizer=OptimizerConfig(),
            epochs=epochs,
            seeds=(42, 43, 44),
            methods=MANDATORY_METHODS,
            stress_seed=4242,
            metric_version=METRIC_VERSION,
            protocol_sha256=protocol_sha256,
            search_table=FrozenSearchTable(tuple(range(1, epochs + 1))),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ResearchRunConfig":
        if not isinstance(value, Mapping):
            raise TypeError("research configuration must be a mapping")
        allowed = {
            "schema_version",
            "dataset",
            "backbone",
            "optimizer",
            "epochs",
            "seeds",
            "methods",
            "stress_seed",
            "metric_version",
            "protocol_sha256",
            "search_table",
        }
        unknown = set(value).difference(allowed)
        if unknown:
            raise ValueError("research configuration contains unknown fields")
        if "dataset" not in value or "backbone" not in value:
            raise ValueError("research configuration requires dataset and backbone")
        if not isinstance(value["dataset"], str) or not isinstance(value["backbone"], str):
            raise ValueError("dataset and backbone must be strings")
        default = cls.default(
            str(value["dataset"]),
            str(value["backbone"]),
            protocol_sha256=str(value.get("protocol_sha256", DEFAULT_PROTOCOL_SHA256)),
        )
        optimizer_value = value.get("optimizer")
        optimizer = default.optimizer
        if optimizer_value is not None:
            if not isinstance(optimizer_value, Mapping):
                raise ValueError("optimizer configuration must be a mapping")
            optimizer_fields = {field.name for field in fields(OptimizerConfig)}
            if set(optimizer_value).difference(optimizer_fields):
                raise ValueError("optimizer configuration contains unknown fields")
            optimizer = OptimizerConfig(**optimizer_value)  # type: ignore[arg-type]
        search_value = value.get("search_table")
        search = default.search_table
        if search_value is not None:
            if not isinstance(search_value, Mapping):
                raise ValueError("search_table must be a mapping")
            allowed_search = {field.name for field in fields(FrozenSearchTable)}
            if set(search_value).difference(allowed_search):
                raise ValueError("search_table contains unknown fields")
            try:
                search = FrozenSearchTable(
                    checkpoint_epochs=tuple(search_value["checkpoint_epochs"]),  # type: ignore[arg-type]
                    ds_regularizations=tuple(
                        search_value.get("ds_regularizations", default.search_table.ds_regularizations)  # type: ignore[arg-type]
                    ),
                    architecture_trials_per_method=search_value.get(
                        "architecture_trials_per_method", 1
                    ),  # type: ignore[arg-type]
                    version=search_value.get("version", SEARCH_TABLE_VERSION),  # type: ignore[arg-type]
                )
            except (KeyError, TypeError) as exc:
                raise ValueError("search_table has an invalid schema") from exc
        return cls(
            dataset=default.dataset,
            backbone=default.backbone,
            optimizer=optimizer,
            epochs=value.get("epochs", default.epochs),  # type: ignore[arg-type]
            seeds=tuple(value.get("seeds", default.seeds)),  # type: ignore[arg-type]
            methods=tuple(value.get("methods", default.methods)),  # type: ignore[arg-type]
            stress_seed=value.get("stress_seed", default.stress_seed),  # type: ignore[arg-type]
            metric_version=value.get("metric_version", default.metric_version),  # type: ignore[arg-type]
            protocol_sha256=value.get("protocol_sha256", default.protocol_sha256),  # type: ignore[arg-type]
            search_table=search,
            schema_version=value.get("schema_version", RUN_CONFIG_VERSION),  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "backbone": self.backbone,
            "optimizer": asdict(self.optimizer),
            "epochs": self.epochs,
            "seeds": list(self.seeds),
            "methods": list(self.methods),
            "stress_seed": self.stress_seed,
            "metric_version": self.metric_version,
            "protocol_sha256": self.protocol_sha256,
            "search_table": self.search_table.to_dict(),
        }

    @property
    def sha256(self) -> str:
        return _sha256_json(self.to_dict())


def load_research_run_config(path: str | Path) -> ResearchRunConfig:
    """Load strict JSON/YAML without accepting unknown or executable choices."""

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
        if source.suffix.lower() == ".json":
            value = json.loads(text)
        elif source.suffix.lower() in (".yaml", ".yml"):
            import yaml

            value = yaml.safe_load(text)
        else:
            raise ValueError("research configuration must be JSON or YAML")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("research configuration could not be read") from exc
    return ResearchRunConfig.from_dict(value)


def load_role_manifest_for_operation(
    path: str | Path,
    *,
    expected: ManifestBinding,
    private_root: str | Path,
    operation: Operation,
    role: Role,
) -> RoleManifest:
    """Authorize a single-role request before opening its private manifest."""

    try:
        normalized_operation = Operation(operation)
        normalized_role = Role(role)
    except (TypeError, ValueError) as exc:
        raise PermissionError("unknown research role or operation") from exc
    if normalized_role is Role.LOCKED_TEST:
        raise PermissionError("locked_test remains inaccessible without an orchestrator release")
    if normalized_role not in OPERATION_ROLE_ALLOWLIST[normalized_operation]:
        raise PermissionError("requested role is forbidden for this operation")
    manifest = load_private_manifest(path, expected=expected, private_root=private_root)
    if {record.role for record in manifest.records} != {normalized_role}:
        raise PermissionError("private workflow manifests must contain only the authorized role")
    return manifest


@dataclass(frozen=True)
class PublicCLIPConfiguration:
    backbone: str
    hf_model: str
    revision: str
    image_size: int
    hidden_size: int
    prompts: tuple[str, ...]
    preprocessing: str
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]


def pinned_public_clip_configuration(backbone: str) -> PublicCLIPConfiguration:
    spec = get_backbone(backbone)
    return PublicCLIPConfiguration(
        backbone=spec.name,
        hf_model=spec.hf_model,
        revision=spec.revision,
        image_size=spec.image_size,
        hidden_size=spec.hidden_size,
        prompts=tuple(PROMPTS),
        preprocessing=LEGACY_PREPROCESSING,
        image_mean=LEGACY_IMAGE_MEAN,
        image_std=LEGACY_IMAGE_STD,
    )


@dataclass(frozen=True)
class ClassifierProvenance:
    kind: str
    backbone: str
    hf_model: str
    revision: str
    image_size: int
    hidden_size: int
    prompts: tuple[str, ...]
    preprocessing: str
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    checkpoint_sha256: str
    diagnostic_reason: str | None
    initialization_sha256: str | None = None
    classifier_fit_manifest_sha256: str | None = None
    classifier_fit_update_count: int = 0
    tune_selection_sha256: str | None = None
    tune_manifest_sha256: str | None = None
    patient_readiness_verified: bool = False
    readiness_audit_sha256: str | None = None

    def __post_init__(self) -> None:
        spec = get_backbone(self.backbone)
        if self.hf_model != spec.hf_model or self.revision != spec.revision:
            raise ValueError("classifier provenance disagrees with the pinned public backbone")
        if self.image_size != spec.image_size or self.hidden_size != spec.hidden_size:
            raise ValueError("classifier provenance disagrees with pinned backbone dimensions")
        if tuple(self.prompts) != tuple(PROMPTS):
            raise ValueError("classifier provenance disagrees with pinned prompt ordering")
        if (
            self.preprocessing != LEGACY_PREPROCESSING
            or tuple(self.image_mean) != LEGACY_IMAGE_MEAN
            or tuple(self.image_std) != LEGACY_IMAGE_STD
        ):
            raise ValueError("classifier provenance disagrees with accepted preprocessing")
        if not _is_sha256(self.checkpoint_sha256):
            raise ValueError("classifier checkpoint identity must be SHA-256")
        if self.kind == "public_pretrained_fresh":
            if self.diagnostic_reason is not None:
                raise ValueError("fresh classifier provenance cannot have a diagnostic reason")
            initialization = self.initialization_sha256 or self.checkpoint_sha256
            if not _is_sha256(initialization):
                raise ValueError("fresh classifier initialization must be SHA-256 bound")
            object.__setattr__(self, "initialization_sha256", initialization)
            bound = self.classifier_fit_manifest_sha256 is not None
            if bound != (
                self.tune_selection_sha256 is not None
                and self.tune_manifest_sha256 is not None
            ):
                raise ValueError("fresh fitting and tune-selection bindings must appear together")
            if bound:
                if not _is_sha256(self.classifier_fit_manifest_sha256) or not _is_sha256(
                    self.tune_selection_sha256
                ) or not _is_sha256(self.tune_manifest_sha256):
                    raise ValueError("fresh fitting/selection identities must be SHA-256")
                if (
                    isinstance(self.classifier_fit_update_count, bool)
                    or not isinstance(self.classifier_fit_update_count, int)
                    or self.classifier_fit_update_count < 1
                ):
                    raise ValueError("fresh classifier exposure must record optimizer updates")
                if not _is_sha256(self.readiness_audit_sha256):
                    raise ValueError("fresh selection requires a readiness-audit binding")
            elif (
                self.classifier_fit_update_count != 0
                or self.patient_readiness_verified
                or self.readiness_audit_sha256 is not None
                or self.tune_manifest_sha256 is not None
            ):
                raise ValueError("unfitted public initialization cannot claim training readiness")
        elif self.kind == "original_finetuned_diagnostic":
            if not isinstance(self.diagnostic_reason, str) or not self.diagnostic_reason.strip():
                raise ValueError("diagnostic classifier provenance requires an exposure reason")
        else:
            raise ValueError("unknown classifier provenance kind")
        if not isinstance(self.patient_readiness_verified, bool):
            raise ValueError("classifier patient readiness must be boolean")

    @classmethod
    def public_pretrained(cls, backbone: str, checkpoint_sha256: str) -> "ClassifierProvenance":
        spec = get_backbone(backbone)
        return cls(
            "public_pretrained_fresh",
            spec.name,
            spec.hf_model,
            spec.revision,
            spec.image_size,
            spec.hidden_size,
            tuple(PROMPTS),
            LEGACY_PREPROCESSING,
            LEGACY_IMAGE_MEAN,
            LEGACY_IMAGE_STD,
            checkpoint_sha256,
            None,
        )

    @classmethod
    def fresh_selected(
        cls,
        initialization: "ClassifierProvenance",
        *,
        checkpoint_sha256: str,
        classifier_fit_manifest_sha256: str,
        classifier_fit_update_count: int,
        tune_selection_sha256: str,
        tune_manifest_sha256: str,
        readiness: "ReadinessAudit",
    ) -> "ClassifierProvenance":
        if initialization.kind != "public_pretrained_fresh" or initialization.workflow_complete:
            raise ValueError("fresh selection requires an unfitted pinned public initialization")
        if not isinstance(readiness, ReadinessAudit):
            raise TypeError("fresh selection requires a factory-created readiness audit")
        if classifier_fit_manifest_sha256 not in readiness.manifest_sha256s:
            raise ValueError("classifier-fit manifest is absent from the readiness audit")
        return cls(
            kind=initialization.kind,
            backbone=initialization.backbone,
            hf_model=initialization.hf_model,
            revision=initialization.revision,
            image_size=initialization.image_size,
            hidden_size=initialization.hidden_size,
            prompts=initialization.prompts,
            preprocessing=initialization.preprocessing,
            image_mean=initialization.image_mean,
            image_std=initialization.image_std,
            checkpoint_sha256=checkpoint_sha256,
            diagnostic_reason=None,
            initialization_sha256=initialization.initialization_sha256,
            classifier_fit_manifest_sha256=classifier_fit_manifest_sha256,
            classifier_fit_update_count=classifier_fit_update_count,
            tune_selection_sha256=tune_selection_sha256,
            tune_manifest_sha256=tune_manifest_sha256,
            patient_readiness_verified=readiness.patient_ready,
            readiness_audit_sha256=readiness.sha256,
        )

    @classmethod
    def diagnostic_original(
        cls, backbone: str, checkpoint_sha256: str, *, reason: str
    ) -> "ClassifierProvenance":
        spec = get_backbone(backbone)
        return cls(
            "original_finetuned_diagnostic",
            spec.name,
            spec.hf_model,
            spec.revision,
            spec.image_size,
            spec.hidden_size,
            tuple(PROMPTS),
            LEGACY_PREPROCESSING,
            LEGACY_IMAGE_MEAN,
            LEGACY_IMAGE_STD,
            checkpoint_sha256,
            reason,
        )

    @property
    def pilot_eligible(self) -> bool:
        return self.workflow_complete and self.patient_readiness_verified

    @property
    def workflow_complete(self) -> bool:
        return (
            self.kind == "public_pretrained_fresh"
            and self.classifier_fit_manifest_sha256 is not None
            and self.classifier_fit_update_count > 0
            and self.tune_selection_sha256 is not None
            and self.tune_manifest_sha256 is not None
        )

    def require_pilot_eligible(self) -> None:
        if self.kind == "original_finetuned_diagnostic":
            raise PermissionError("diagnostic original classifier cannot qualify a held-out pilot")
        if not self.workflow_complete:
            raise PermissionError("fresh classifier fitting/tune workflow is incomplete")
        if not self.patient_readiness_verified:
            raise PermissionError("real patient readiness is unverified")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "backbone": self.backbone,
            "hf_model": self.hf_model,
            "revision": self.revision,
            "image_size": self.image_size,
            "hidden_size": self.hidden_size,
            "prompts": list(self.prompts),
            "preprocessing": self.preprocessing,
            "image_mean": list(self.image_mean),
            "image_std": list(self.image_std),
            "checkpoint_sha256": self.checkpoint_sha256,
            "diagnostic_reason": self.diagnostic_reason,
            "initialization_sha256": self.initialization_sha256,
            "classifier_fit_manifest_sha256": self.classifier_fit_manifest_sha256,
            "classifier_fit_update_count": self.classifier_fit_update_count,
            "tune_selection_sha256": self.tune_selection_sha256,
            "tune_manifest_sha256": self.tune_manifest_sha256,
            "patient_readiness_verified": self.patient_readiness_verified,
            "readiness_audit_sha256": self.readiness_audit_sha256,
        }


def initialize_public_classifier(
    backbone: str,
    factory: Callable[[PublicCLIPConfiguration], nn.Module],
) -> tuple[nn.Module, ClassifierProvenance]:
    """Construct a fresh classifier through a public-pin-aware injected factory."""

    public = pinned_public_clip_configuration(backbone)
    model = factory(public)
    if not isinstance(model, nn.Module) or not tuple(model.parameters()):
        raise TypeError("public classifier factory must return a parameterized torch module")
    digest = module_state_sha256(model)
    return model, ClassifierProvenance.public_pretrained(public.backbone, digest)


_READINESS_AUDIT_TOKEN = object()


@dataclass(frozen=True, init=False)
class ReadinessAudit:
    kind: str
    patient_ready: bool
    dataset_count: int
    exam_count: int
    patient_count: int
    role_exam_counts: Mapping[str, int]
    locked_isolation_verified: bool
    manifest_sha256s: tuple[str, ...]
    locked_denylist_sha256: str | None
    sha256: str

    def __init__(
        self,
        kind: str,
        patient_ready: bool,
        dataset_count: int,
        exam_count: int,
        patient_count: int,
        role_exam_counts: Mapping[str, int],
        locked_isolation_verified: bool,
        manifest_sha256s: Sequence[str],
        locked_denylist_sha256: str | None,
        *,
        _factory_token: object,
    ) -> None:
        if _factory_token is not _READINESS_AUDIT_TOKEN:
            raise RuntimeError("ReadinessAudit must be produced by an inventory audit")
        manifests = tuple(sorted(manifest_sha256s))
        if not manifests or any(not _is_sha256(value) for value in manifests):
            raise ValueError("readiness audit requires manifest SHA-256 bindings")
        if locked_denylist_sha256 is not None and not _is_sha256(locked_denylist_sha256):
            raise ValueError("readiness audit locked denylist binding must be SHA-256")
        counts = MappingProxyType(dict(sorted(role_exam_counts.items())))
        payload = {
            "kind": kind,
            "patient_ready": patient_ready,
            "dataset_count": dataset_count,
            "exam_count": exam_count,
            "patient_count": patient_count,
            "role_exam_counts": dict(counts),
            "locked_isolation_verified": locked_isolation_verified,
            "manifest_sha256s": list(manifests),
            "locked_denylist_sha256": locked_denylist_sha256,
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "role_exam_counts", counts)
        object.__setattr__(self, "manifest_sha256s", manifests)
        object.__setattr__(self, "sha256", _sha256_json(payload))


def audit_software_fixture(manifest: RoleManifest) -> ReadinessAudit:
    """Validate fixture structure while explicitly withholding patient readiness."""

    summary = validate_inventory((manifest,))
    return ReadinessAudit(
        "synthetic_software",
        False,
        summary.dataset_count,
        summary.exam_count,
        summary.patient_count,
        MappingProxyType({role.value: summary.role_exam_counts[role] for role in Role}),
        False,
        (manifest.manifest_sha256,),
        None,
        _factory_token=_READINESS_AUDIT_TOKEN,
    )


def audit_real_data_readiness(
    manifests: Sequence[RoleManifest],
    *,
    locked_patient_denylist: LockedPatientIdentityDenylist,
) -> ReadinessAudit:
    """Audit all non-test roles plus identity-only locked-patient isolation."""

    materialized = tuple(manifests)
    summary = validate_inventory(
        materialized, locked_patient_denylists=(locked_patient_denylist,)
    )
    represented = {
        record.role for manifest in materialized for record in manifest.records
    }
    if represented != set(NON_TEST_ROLES):
        raise ValueError("real-data readiness requires every non-test role")
    return ReadinessAudit(
        "real_data",
        True,
        summary.dataset_count,
        summary.exam_count,
        summary.patient_count,
        MappingProxyType({role.value: summary.role_exam_counts[role] for role in Role}),
        True,
        tuple(manifest.manifest_sha256 for manifest in materialized),
        locked_patient_denylist.source_sha256,
        _factory_token=_READINESS_AUDIT_TOKEN,
    )


@dataclass(frozen=True)
class ScheduledExample:
    exam_key: str
    draw: TrainingSampleSpec


def build_training_schedule(
    records: Sequence[PrivateExamRecord], *, epoch: int, seed: int
) -> tuple[ScheduledExample, ...]:
    """Build the method-independent, deterministic per-epoch fit schedule."""

    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError("epoch must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    result = []
    for record in records:
        if not isinstance(record, PrivateExamRecord):
            raise TypeError("training schedule requires private exam records")
        private_key = f"{record.patient_key}\0{record.exam_key}"
        result.append(
            ScheduledExample(
                record.exam_key,
                sample_training_spec(
                    private_key,
                    f"epoch/{epoch}",
                    seed,
                    operation="confidence-fit",
                ),
            )
        )
    return tuple(result)


def build_method_training_schedule(
    records: Sequence[PrivateExamRecord], *, epoch: int, seed: int, method: str
) -> tuple[ScheduledExample, ...]:
    """Apply only each control's declared exposure change to the common draws."""

    schedule = build_training_schedule(records, epoch=epoch, seed=seed)
    if method not in MANDATORY_METHODS:
        raise ValueError("method is outside the frozen training schedule")
    if method == "original_mvacn_tcp":
        clean_four = TrainingSampleSpec(
            stratum="clean4",
            observed_views=tuple(CANONICAL_VIEWS),
            stress_mode=None,
            stressed_views=(),
            perturbation=None,
        )
        return tuple(ScheduledExample(item.exam_key, clean_four) for item in schedule)
    if method in ("correctness_mvacn", "clean_fitting"):
        stripped = []
        for item in schedule:
            full = item.draw.observed_views == tuple(CANONICAL_VIEWS)
            draw = TrainingSampleSpec(
                stratum="clean4" if full else "clean_proper_mask",
                observed_views=item.draw.observed_views,
                stress_mode=None,
                stressed_views=(),
                perturbation=None,
            )
            stripped.append(ScheduledExample(item.exam_key, draw))
        return tuple(stripped)
    return schedule


def _same_tensor(actual: Tensor, expected: Tensor) -> bool:
    return actual.dtype == expected.dtype and actual.shape == expected.shape and torch.equal(
        actual, expected
    )


def checked_intervention_targets(
    features: FrozenViewFeatures, cached: CachedTargets
) -> InterventionTargets:
    """Regenerate exact current targets and reject a stale ``CachedTargets`` adapter."""

    if not isinstance(features, FrozenViewFeatures) or not isinstance(cached, CachedTargets):
        raise TypeError("target adaptation requires frozen features and CachedTargets")
    generated = build_intervention_targets(
        features.logits_by_view,
        cached.labels,
        features.observed_views,
        fusion_pairs=features.fusion_pairs,
    )
    comparisons = (
        (cached.observed_prediction, generated.observed_prediction),
        (cached.observed_error, generated.observed_error),
        (cached.omission_predictions, generated.omission_predictions),
        (cached.omission_effects, generated.omission_effects),
        (cached.omission_labels, generated.omission_labels),
        (cached.valid_removal_mask, generated.valid_removal_mask),
        (cached.scores, features.scores),
        (cached.probabilities, features.scores.softmax(1)),
        (
            cached.tcp,
            features.scores.softmax(1).gather(1, cached.labels[:, None]).squeeze(1),
        ),
    )
    if any(not _same_tensor(actual, expected) for actual, expected in comparisons):
        raise ValueError("CachedTargets are stale for the current realized parent")
    return generated


def _legacy_order(fusion_pairs: FusionPairs) -> tuple[str, ...]:
    if tuple(fusion_pairs) == RSNA_FUSION_PAIRS:
        return tuple(CANONICAL_VIEWS)
    if tuple(fusion_pairs) == DDSM_FUSION_PAIRS:
        return ("L_CC", "R_CC", "L_MLO", "R_MLO")
    raise ValueError("confidence method requires an accepted RSNA or DDSM fusion tree")


def build_learned_method(
    method: str,
    backbone: str,
    fusion_pairs: FusionPairs,
    *,
    visual_width: int | None = None,
    text_embeddings: Tensor | None = None,
) -> nn.Module:
    """Instantiate one method with method-owned projections/scalers."""

    if method not in LEARNED_TORCH_METHODS:
        raise ValueError("method is not a learned torch confidence method")
    if method == "candidate":
        return RelationAwareConfidenceHead(RelationAwareHeadConfig(backbone=backbone))
    ablations = p3a_ablation_definitions(RelationAwareHeadConfig(backbone=backbone))
    if method in ablations:
        return RelationAwareConfidenceHead(ablations[method].head_config)
    if method in {
        "original_mvacn_tcp",
        "correctness_mvacn",
        "matched_mvacn_tcp",
        "matched_mvacn_correctness",
    }:
        return MaskedMVACNAdapter.for_backbone(
            backbone, legacy_view_order=_legacy_order(fusion_pairs)
        )
    if method == "same_input_mlp":
        return SameInputMLP(backbone)
    if method == "same_input_density":
        return SameInputDensityControl(backbone)
    if method == "vilu":
        if visual_width is None or text_embeddings is None:
            raise ValueError("ViLU construction requires frozen visual width and text embeddings")
        return ViLUFailureAdapter(
            visual_width=visual_width, text_embeddings=text_embeddings.detach()
        )
    raise RuntimeError("unhandled learned method")


def _projected_slots(features: FrozenViewFeatures) -> tuple[Tensor, Tensor]:
    reference = features.projected_by_view[features.observed_views[0]]
    slots = reference.new_zeros(features.batch_size, 4, reference.shape[1])
    mask = torch.zeros(features.batch_size, 4, dtype=torch.bool, device=reference.device)
    for column, view in enumerate(CANONICAL_VIEWS):
        if view in features.observed_views:
            slots[:, column] = features.projected_by_view[view]
            mask[:, column] = True
    return slots.detach(), mask


def confidence_bundle_loss(
    model: nn.Module,
    method: str,
    bundle: CacheBundle,
    *,
    corruption_targets: Tensor | None = None,
) -> Tensor:
    """Compute one accepted learned method loss from a validated current cache."""

    if not isinstance(bundle, CacheBundle):
        raise TypeError("confidence fitting requires validated CacheBundle values")
    try:
        device = next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("confidence model must contain trainable parameters") from exc
    source = bundle.features
    features = FrozenViewFeatures(
        logits_by_view={view: source.logits_by_view[view].to(device) for view in source.observed_views},
        hidden_by_view={view: source.hidden_by_view[view].to(device) for view in source.observed_views},
        projected_by_view={
            view: source.projected_by_view[view].to(device) for view in source.observed_views
        },
        normalized_text_embeddings=source.normalized_text_embeddings.to(device),
        scores=source.scores.to(device),
        probabilities=source.probabilities.to(device),
        observed_views=source.observed_views,
        fusion_pairs=source.fusion_pairs,
    )
    cached = CachedTargets(
        **{
            field.name: getattr(bundle.targets, field.name).to(device)
            for field in fields(CachedTargets)
        }
    )
    targets = checked_intervention_targets(features, cached)
    raw = prepare_raw_head_inputs(features, backbone=bundle.provenance.backbone)
    if isinstance(model, RelationAwareConfidenceHead):
        definition = p3a_ablation_definitions(model.config).get(method)
        auxiliary_weight = 1.0 if definition is None else definition.auxiliary_weight
        return compute_view_risk_loss(
            model(raw),
            targets,
            auxiliary_weight=auxiliary_weight,
            corruption_targets=(
                None if corruption_targets is None else corruption_targets.to(device)
            ),
        ).total
    if isinstance(model, MaskedMVACNAdapter):
        output = model(
            features.hidden_by_view,
            features.observed_views,
            classifier_prediction=features.prediction,
            current_scores=features.scores,
        )
        objective = (
            MVACNObjective.TCP_MSE
            if method in ("original_mvacn_tcp", "matched_mvacn_tcp")
            else MVACNObjective.CORRECTNESS_BCE
        )
        return compute_mvacn_objective(output, cached, objective)
    if isinstance(model, SameInputMLP):
        return compute_same_input_error_loss(model(raw), cached)
    if isinstance(model, SameInputDensityControl):
        return compute_density_control_loss(model(raw), cached)
    if isinstance(model, ViLUFailureAdapter):
        projected, mask = _projected_slots(features)
        return compute_vilu_failure_loss(
            model(
                projected,
                mask,
                classifier_prediction=features.prediction,
                current_scores=features.scores,
            ),
            cached,
        )
    raise TypeError("model type is not bound to an accepted learned confidence objective")


@dataclass(frozen=True)
class RoleBoundBatch:
    """One bounded optimizer batch with private identities kept out of reports."""

    exam_keys: tuple[str, ...]
    payload: object


@dataclass(frozen=True)
class TrainingBinding:
    protocol_sha256: str
    config_sha256: str
    search_table_sha256: str
    manifest_sha256: str
    classifier_checkpoint_sha256: str
    method: str
    seed: int
    schedule_version: str = TRAINING_SCHEDULE_VERSION

    def __post_init__(self) -> None:
        for name in (
            "protocol_sha256",
            "config_sha256",
            "search_table_sha256",
            "manifest_sha256",
            "classifier_checkpoint_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be SHA-256")
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("method binding must be nonempty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("training seed must be nonnegative")
        if self.schedule_version != TRAINING_SCHEDULE_VERSION:
            raise ValueError("unsupported training schedule binding")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingResult:
    method: str
    seed: int
    completed_epoch: int
    update_count: int
    exposed_record_count: int
    parameter_count: int
    model_state_sha256: str
    manifest_sha256: str
    scaling: str
    selection_trial_budget: int
    actual_exposure_verified: bool
    software_only: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def module_state_sha256(module: nn.Module) -> str:
    manifest = [
        (name, str(value.dtype), list(value.shape), tensor_sha256(value.detach()))
        for name, value in sorted(module.state_dict().items())
    ]
    return _sha256_json(manifest)


def _rng_state() -> dict[str, object]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy_name": numpy_state[0],
        # Torch 2.3 cannot serialize a uint32 TypedStorage. Preserve the exact
        # MT19937 words losslessly in int64 and restore uint32 below.
        "numpy_keys": torch.from_numpy(numpy_state[1].astype(np.int64, copy=True)),
        "numpy_position": int(numpy_state[2]),
        "numpy_has_gauss": int(numpy_state[3]),
        "numpy_cached_gaussian": float(numpy_state[4]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(
        (
            str(state["numpy_name"]),
            state["numpy_keys"].cpu().numpy().astype(np.uint32),  # type: ignore[union-attr]
            int(state["numpy_position"]),
            int(state["numpy_has_gauss"]),
            float(state["numpy_cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch"])  # type: ignore[arg-type]
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])  # type: ignore[arg-type]


def _checkpoint_paths(path: str | Path) -> tuple[Path, Path, Path, Path]:
    checkpoint = Path(path).resolve()
    if checkpoint.suffix != ".pt":
        raise ValueError("resume checkpoint path must end in .pt")
    if not checkpoint.parent.exists():
        raise ValueError("resume checkpoint parent directory must already exist")
    metadata = checkpoint.with_suffix(".json")
    temporary = checkpoint.with_name(f".{checkpoint.name}.{os.getpid()}.tmp")
    metadata_temporary = metadata.with_name(f".{metadata.name}.{os.getpid()}.tmp")
    for candidate in (checkpoint, metadata, temporary, metadata_temporary):
        _require_external_or_ignored_destination(candidate)
    return checkpoint, metadata, temporary, metadata_temporary


def _save_resume(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    binding: TrainingBinding,
    completed_epoch: int,
    update_count: int,
) -> None:
    checkpoint, metadata, temporary, metadata_temporary = _checkpoint_paths(path)
    payload = {
        "resume_version": RESUME_VERSION,
        "binding": binding.to_dict(),
        "completed_epoch": completed_epoch,
        "update_count": update_count,
        "schedule_state": {
            "version": binding.schedule_version,
            "next_epoch": completed_epoch + 1,
        },
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": _rng_state(),
    }
    try:
        torch.save(payload, temporary)
        digest = sha256_file(temporary)
        document = {
            "resume_version": RESUME_VERSION,
            "binding": binding.to_dict(),
            "checkpoint_file": checkpoint.name,
            "checkpoint_sha256": digest,
        }
        with metadata_temporary.open("xb") as handle:
            handle.write(_canonical_json(document) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, checkpoint)
        os.replace(metadata_temporary, metadata)
    finally:
        temporary.unlink(missing_ok=True)
        metadata_temporary.unlink(missing_ok=True)


def _load_resume(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    binding: TrainingBinding,
) -> tuple[int, int]:
    checkpoint, metadata, _temporary, _metadata_temporary = _checkpoint_paths(path)
    try:
        document = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("resume metadata is unavailable or invalid") from exc
    expected_fields = {
        "resume_version",
        "binding",
        "checkpoint_file",
        "checkpoint_sha256",
    }
    if not isinstance(document, dict) or set(document) != expected_fields:
        raise ValueError("resume metadata schema is invalid")
    if document["resume_version"] != RESUME_VERSION or document["binding"] != binding.to_dict():
        raise ValueError("resume binding does not match the requested run")
    if document["checkpoint_file"] != checkpoint.name:
        raise ValueError("resume checkpoint filename binding is invalid")
    if document["checkpoint_sha256"] != sha256_file(checkpoint):
        raise ValueError("resume checkpoint integrity check failed")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("resume checkpoint is unreadable") from exc
    if payload.get("resume_version") != RESUME_VERSION or payload.get("binding") != binding.to_dict():
        raise ValueError("resume payload binding does not match the requested run")
    completed = int(payload["completed_epoch"])
    schedule = payload.get("schedule_state")
    if schedule != {"version": binding.schedule_version, "next_epoch": completed + 1}:
        raise ValueError("resume schedule state is stale")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    _restore_rng_state(payload["rng_state"])
    return completed, int(payload["update_count"])


def fit_role_bound_module(
    *,
    manifest: RoleManifest,
    operation: Operation,
    role: Role,
    model: nn.Module,
    optimizer_config: OptimizerConfig,
    epochs: int,
    binding: TrainingBinding,
    batch_loader: Callable[
        [tuple[PrivateExamRecord, ...], int], Iterable[RoleBoundBatch]
    ],
    loss_fn: Callable[[nn.Module, object], Tensor],
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    stop_after_epoch: int | None = None,
    frozen_modules: Sequence[nn.Module] = (),
    _classifier_fit_token: object | None = None,
) -> TrainingResult:
    """Run deterministic Adam updates after role authorization, with exact resume."""

    if operation not in (Operation.CLASSIFIER_FITTING, Operation.CONFIDENCE_FITTING):
        raise PermissionError("optimizer operation must be classifier or confidence fitting")
    if (
        operation is Operation.CLASSIFIER_FITTING
        and _classifier_fit_token is not _FRESH_CLASSIFIER_FIT_TOKEN
    ):
        raise PermissionError("classifier fitting must use the audited fresh-classifier workflow")
    if binding.manifest_sha256 != manifest.manifest_sha256:
        raise ValueError("training manifest binding is stale")
    if not isinstance(model, nn.Module) or not any(p.requires_grad for p in model.parameters()):
        raise ValueError("training model must have trainable parameters")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be positive")
    if resume and checkpoint_path is None:
        raise ValueError("resume requires an explicit checkpoint path")
    if checkpoint_path is not None:
        _checkpoint_paths(checkpoint_path)
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= epochs:
        raise ValueError("stop_after_epoch must fall within the configured run")

    frozen_hashes: list[str] = []
    for frozen in frozen_modules:
        if any(parameter.requires_grad for parameter in frozen.parameters()):
            raise ValueError("frozen modules must have requires_grad disabled")
        frozen.eval()
        frozen_hashes.append(module_state_sha256(frozen))

    def authorized(records: tuple[PrivateExamRecord, ...]) -> TrainingResult:
        if not records:
            raise PermissionError("authorized fitting role contains no records")
        expected_keys = {record.exam_key for record in records}
        if len(expected_keys) != len(records):
            raise ValueError("authorized fitting records contain duplicate identities")
        random.seed(binding.seed)
        np.random.seed(binding.seed % (2**32))
        torch.manual_seed(binding.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(binding.seed)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(optimizer_config.learning_rate),
            weight_decay=float(optimizer_config.weight_decay),
        )
        completed_epoch = 0
        update_count = 0
        if resume:
            assert checkpoint_path is not None
            completed_epoch, update_count = _load_resume(
                checkpoint_path, model=model, optimizer=optimizer, binding=binding
            )
            if completed_epoch >= epochs:
                raise ValueError("resume checkpoint has already reached the configured epoch budget")

        final_epoch = epochs if stop_after_epoch is None else stop_after_epoch
        model.train()
        for epoch in range(completed_epoch + 1, final_epoch + 1):
            seen: set[str] = set()
            for batch in batch_loader(records, epoch):
                if not isinstance(batch, RoleBoundBatch):
                    raise TypeError("batch_loader must yield RoleBoundBatch values")
                keys = tuple(batch.exam_keys)
                if (
                    not keys
                    or len(keys) > optimizer_config.batch_size
                    or len(set(keys)) != len(keys)
                    or any(key not in expected_keys or key in seen for key in keys)
                ):
                    raise ValueError("optimizer batch exposure is incomplete, duplicate, or unauthorized")
                seen.update(keys)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model, batch.payload)
                if not isinstance(loss, Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
                    raise ValueError("training loss must be one finite scalar")
                loss.backward()
                optimizer.step()
                update_count += 1
                for index, frozen in enumerate(frozen_modules):
                    if module_state_sha256(frozen) != frozen_hashes[index]:
                        raise RuntimeError("frozen classifier changed during confidence fitting")
            if seen != expected_keys:
                raise ValueError("optimizer epoch did not expose every authorized record exactly once")
            completed_epoch = epoch
            if checkpoint_path is not None:
                _save_resume(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    binding=binding,
                    completed_epoch=completed_epoch,
                    update_count=update_count,
                )
        scaling_by_method = {
            "same_input_mlp": "method_owned_confidence_fit_affine",
            "same_input_density": "method_owned_confidence_fit_affine",
        }
        scaling = scaling_by_method.get(binding.method, "declared_identity_no_fitted_scaler")
        return TrainingResult(
            method=binding.method,
            seed=binding.seed,
            completed_epoch=completed_epoch,
            update_count=update_count,
            exposed_record_count=len(records),
            parameter_count=count_trainable_parameters(model),
            model_state_sha256=module_state_sha256(model),
            manifest_sha256=manifest.manifest_sha256,
            scaling=scaling,
            selection_trial_budget=epochs,
            actual_exposure_verified=True,
        )

    return run_with_role_access(
        manifest,
        operation=operation,
        roles=(role,),
        loader=authorized,
    )


def fit_fresh_classifier_with_role_access(
    *,
    manifest: RoleManifest,
    model: nn.Module,
    classifier: ClassifierProvenance,
    readiness: ReadinessAudit,
    optimizer_config: OptimizerConfig,
    epochs: int,
    binding: TrainingBinding,
    batch_loader: Callable[
        [tuple[PrivateExamRecord, ...], int], Iterable[RoleBoundBatch]
    ],
    loss_fn: Callable[[nn.Module, object], Tensor],
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    stop_after_epoch: int | None = None,
) -> TrainingResult:
    """Fit only classifier-fit rows from a pinned public initialization.

    A synthetic readiness record permits software testing but is carried into
    the result and can never satisfy :meth:`ClassifierProvenance.require_pilot_eligible`
    on behalf of real patient data.
    """

    if classifier.kind != "public_pretrained_fresh" or classifier.workflow_complete:
        raise PermissionError("classifier fitting requires an unfitted pinned public initialization")
    if classifier.checkpoint_sha256 != module_state_sha256(model):
        raise ValueError("fresh classifier initialization identity is stale")
    if binding.classifier_checkpoint_sha256 != classifier.checkpoint_sha256:
        raise ValueError("classifier training binding disagrees with public initialization")
    if readiness.kind not in ("synthetic_software", "real_data"):
        raise ValueError("classifier fitting requires a recognized readiness audit")
    if manifest.manifest_sha256 not in readiness.manifest_sha256s:
        raise ValueError("classifier fitting manifest is absent from the readiness audit")
    result = fit_role_bound_module(
        manifest=manifest,
        operation=Operation.CLASSIFIER_FITTING,
        role=Role.CLASSIFIER_FIT,
        model=model,
        optimizer_config=optimizer_config,
        epochs=epochs,
        binding=binding,
        batch_loader=batch_loader,
        loss_fn=loss_fn,
        checkpoint_path=checkpoint_path,
        resume=resume,
        stop_after_epoch=stop_after_epoch,
        _classifier_fit_token=_FRESH_CLASSIFIER_FIT_TOKEN,
    )
    return replace(result, software_only=not readiness.patient_ready)


@dataclass(frozen=True)
class ConfidenceFitBatch:
    """A bounded set of homogeneous-mask cache bundles for one optimizer step."""

    bundles: tuple[CacheBundle, ...]
    corruption_targets: tuple[Tensor | None, ...] = ()

    def __post_init__(self) -> None:
        materialized = tuple(self.bundles)
        if not materialized or any(not isinstance(item, CacheBundle) for item in materialized):
            raise TypeError("confidence batches require one or more CacheBundle values")
        targets = tuple(self.corruption_targets)
        if not targets:
            targets = (None,) * len(materialized)
        if len(targets) != len(materialized):
            raise ValueError("corruption-target groups must align with cache bundles")
        object.__setattr__(self, "bundles", materialized)
        object.__setattr__(self, "corruption_targets", targets)


def _expected_perturbation(draw: TrainingSampleSpec, view: str) -> Mapping[str, object]:
    if view not in draw.stressed_views or draw.perturbation is None:
        return PerturbationSpec("clean").to_dict()
    return draw.perturbation.to_dict()


def _validate_scheduled_bundle(
    bundle: CacheBundle,
    *,
    manifest: RoleManifest,
    classifier_checkpoint_sha256: str,
    schedule: Mapping[str, ScheduledExample],
) -> None:
    provenance = bundle.provenance
    if (
        provenance.dataset_namespace != manifest.dataset_namespace
        or provenance.manifest_sha256 != manifest.manifest_sha256
        or provenance.role != Role.CONFIDENCE_FIT.value
        or provenance.operation != Operation.CONFIDENCE_FITTING.value
        or provenance.checkpoint_sha256 != classifier_checkpoint_sha256
    ):
        raise ValueError("confidence cache provenance binding is stale or mismatched")
    for exam_key in provenance.exam_keys:
        scheduled = schedule.get(exam_key)
        if scheduled is None:
            raise ValueError("confidence cache contains an unauthorized scheduled row")
        if provenance.observed_views != scheduled.draw.observed_views:
            raise ValueError("confidence cache observed mask disagrees with frozen schedule")
        for view in provenance.observed_views:
            record = provenance.perturbations[view]
            if record is None or not isinstance(record.get("spec"), Mapping):
                raise ValueError("confidence cache perturbation binding is incomplete")
            spec = record["spec"]
            actual = dict(spec)
            if actual != _expected_perturbation(scheduled.draw, view):
                raise ValueError("confidence cache perturbation disagrees with frozen schedule")


def fit_confidence_method_with_role_access(
    *,
    manifest: RoleManifest,
    model: nn.Module,
    method: str,
    classifier: ClassifierProvenance,
    config: ResearchRunConfig,
    seed: int,
    binding: TrainingBinding,
    batch_loader: Callable[
        [tuple[PrivateExamRecord, ...], tuple[ScheduledExample, ...], int, int],
        Iterable[ConfidenceFitBatch],
    ],
    frozen_encoder: VerifiedFrozenEncoder | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    stop_after_epoch: int | None = None,
) -> TrainingResult:
    """Fit a candidate/control by streaming bounded current cache bundles.

    All learned methods use the same method-independent schedule.  Each cache is
    checked against the role, manifest, classifier, mask, and realized stress
    before its regenerated target reaches an objective.
    """

    if method not in LEARNED_TORCH_METHODS:
        raise ValueError("requested confidence method has no torch fitting workflow")
    if classifier.kind != "public_pretrained_fresh" or not classifier.workflow_complete:
        raise PermissionError(
            "confidence fitting requires a fit-and-tune-selected fresh public classifier"
        )
    if method not in config.methods or seed not in config.seeds:
        raise ValueError("method/seed is outside the frozen configuration")
    if binding.method != method or binding.seed != seed:
        raise ValueError("training binding disagrees with method/seed")
    if binding.config_sha256 != config.sha256:
        raise ValueError("training configuration binding is stale")
    if binding.search_table_sha256 != config.search_table.sha256:
        raise ValueError("training search-table binding is stale")
    if binding.protocol_sha256 != config.protocol_sha256:
        raise ValueError("training protocol binding is stale")
    if binding.classifier_checkpoint_sha256 != classifier.checkpoint_sha256:
        raise ValueError("training classifier binding is stale")
    if frozen_encoder is not None:
        identity = frozen_encoder.identity
        if (
            identity.checkpoint_sha256 != classifier.checkpoint_sha256
            or identity.backbone != classifier.backbone
            or identity.hf_model != classifier.hf_model
            or identity.backbone_revision != classifier.revision
            or identity.image_size != classifier.image_size
            or identity.hidden_size != classifier.hidden_size
            or identity.preprocessing != classifier.preprocessing
            or identity.image_mean != classifier.image_mean
            or identity.image_std != classifier.image_std
            or identity.prompt_order != classifier.prompts
        ):
            raise ValueError("verified frozen encoder disagrees with classifier provenance")
        frozen = (frozen_encoder.classifier,)
    else:
        frozen = ()

    def batches(
        records: tuple[PrivateExamRecord, ...], epoch: int
    ) -> Iterable[RoleBoundBatch]:
        schedule = build_method_training_schedule(
            records, epoch=epoch, seed=seed, method=method
        )
        schedule_by_key = {item.exam_key: item for item in schedule}
        for batch in batch_loader(records, schedule, config.optimizer.batch_size, epoch):
            if not isinstance(batch, ConfidenceFitBatch):
                raise TypeError("confidence batch loader must yield ConfidenceFitBatch values")
            row_count = sum(bundle.features.batch_size for bundle in batch.bundles)
            if row_count > config.optimizer.batch_size:
                raise ValueError("confidence cache batch exceeds the frozen batch size")
            exam_keys: list[str] = []
            for bundle in batch.bundles:
                _validate_scheduled_bundle(
                    bundle,
                    manifest=manifest,
                    classifier_checkpoint_sha256=classifier.checkpoint_sha256,
                    schedule=schedule_by_key,
                )
                exam_keys.extend(bundle.provenance.exam_keys)
            yield RoleBoundBatch(tuple(exam_keys), batch)

    def loss(current_model: nn.Module, payload: object) -> Tensor:
        if not isinstance(payload, ConfidenceFitBatch):
            raise TypeError("confidence loss payload binding is invalid")
        weighted: list[Tensor] = []
        counts: list[int] = []
        for bundle, corruption in zip(payload.bundles, payload.corruption_targets):
            value = confidence_bundle_loss(
                current_model, method, bundle, corruption_targets=corruption
            )
            weighted.append(value * bundle.features.batch_size)
            counts.append(bundle.features.batch_size)
        return torch.stack(weighted).sum() / sum(counts)

    return fit_role_bound_module(
        manifest=manifest,
        operation=Operation.CONFIDENCE_FITTING,
        role=Role.CONFIDENCE_FIT,
        model=model,
        optimizer_config=config.optimizer,
        epochs=config.epochs,
        binding=binding,
        batch_loader=batches,
        loss_fn=loss,
        checkpoint_path=checkpoint_path,
        resume=resume,
        stop_after_epoch=stop_after_epoch,
        frozen_modules=frozen,
    )


__all__ = [
    "ABLATION_METHODS",
    "LEARNED_TORCH_METHODS",
    "MANDATORY_METHODS",
    "METRIC_VERSION",
    "ClassifierProvenance",
    "ConfidenceFitBatch",
    "FrozenSearchTable",
    "OptimizerConfig",
    "PublicCLIPConfiguration",
    "ReadinessAudit",
    "ResearchRunConfig",
    "RoleBoundBatch",
    "ScheduledExample",
    "TrainingBinding",
    "TrainingResult",
    "audit_real_data_readiness",
    "audit_software_fixture",
    "build_learned_method",
    "build_method_training_schedule",
    "build_training_schedule",
    "checked_intervention_targets",
    "confidence_bundle_loss",
    "fit_role_bound_module",
    "fit_confidence_method_with_role_access",
    "fit_fresh_classifier_with_role_access",
    "initialize_public_classifier",
    "load_research_run_config",
    "load_role_manifest_for_operation",
    "module_state_sha256",
    "pinned_public_clip_configuration",
]
