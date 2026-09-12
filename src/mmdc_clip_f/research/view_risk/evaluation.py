"""Guarded tune selection and immutable pilot-evaluation orchestration."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence, TypeVar

from .cache import _require_external_or_ignored_destination
from .inputs import CANONICAL_VIEWS
from .metrics import EvaluationPrediction, evaluate_aurc_panel
from .perturbations import FITTING_FAMILIES, FITTING_SEVERITIES
from .roles import Operation, PrivateExamRecord, Role, RoleManifest, run_with_role_access
from .training import (
    MANDATORY_METHODS,
    METRIC_VERSION,
    ClassifierProvenance,
    FrozenSearchTable,
    ReadinessAudit,
    ResearchRunConfig,
    TrainingResult,
)


PILOT_PLAN_VERSION = "view-risk-pilot-plan/v1"
PILOT_PLAN_ENVELOPE_VERSION = "view-risk-pilot-plan-envelope/v1"
TUNE_GUARDRAIL = 0.005


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
class TuneCellResult:
    family: str
    severity: str
    target_view: str
    aurc: float

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value for value in self.__dict__.values() if isinstance(value, str)):
            raise ValueError("tune cell names must be nonempty")
        if self.target_view not in CANONICAL_VIEWS:
            raise ValueError("tune target view must be canonical")
        if (
            isinstance(self.aurc, bool)
            or not isinstance(self.aurc, (int, float))
            or not math.isfinite(self.aurc)
        ):
            raise ValueError("tune AURC must be finite")
        if self.aurc < 0:
            raise ValueError("tune AURC must be nonnegative")


def _validate_tune_cells(cells: Sequence[TuneCellResult], *, require_complete: bool) -> None:
    materialized = tuple(cells)
    if not materialized:
        raise ValueError("tune evaluation requires at least one fitting-family cell")
    for cell in materialized:
        if not isinstance(cell, TuneCellResult):
            raise TypeError("tune cells must be TuneCellResult values")
        if cell.family not in FITTING_FAMILIES:
            raise ValueError("held-out families are forbidden during tune evaluation")
        if cell.severity not in FITTING_SEVERITIES:
            raise ValueError("strong or unknown severity is forbidden during tune evaluation")
    keys = {(cell.family, cell.severity, cell.target_view) for cell in materialized}
    if len(keys) != len(materialized):
        raise ValueError("tune evaluation contains duplicate cells")
    expected = {
        (family, severity, view)
        for family in FITTING_FAMILIES
        for severity in FITTING_SEVERITIES
        for view in CANONICAL_VIEWS
    }
    if require_complete and keys != expected:
        raise ValueError("checkpoint selection requires the balanced 16-cell tune panel")


T = TypeVar("T")


def evaluate_tune_cells_with_role_access(
    manifest: RoleManifest,
    cells: Sequence[TuneCellResult],
    reader: Callable[[tuple[PrivateExamRecord, ...], tuple[TuneCellResult, ...]], T],
) -> T:
    """Validate the tune panel before authorizing and invoking a row reader."""

    materialized = tuple(cells)
    _validate_tune_cells(materialized, require_complete=False)
    return run_with_role_access(
        manifest,
        operation=Operation.TUNE_SELECTION,
        roles=(Role.TUNE,),
        loader=lambda records: reader(records, materialized),
    )


@dataclass(frozen=True)
class ClassifierTuneCheckpoint:
    epoch: int
    artifact_sha256: str
    tune_nll: float

    def __post_init__(self) -> None:
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int) or self.epoch < 1:
            raise ValueError("classifier checkpoint epoch must be positive")
        if not _is_sha256(self.artifact_sha256):
            raise ValueError("classifier checkpoint artifact identity must be SHA-256")
        if (
            isinstance(self.tune_nll, bool)
            or not isinstance(self.tune_nll, (int, float))
            or not math.isfinite(self.tune_nll)
        ):
            raise ValueError("classifier tune NLL must be finite")


def select_fresh_classifier_on_tune(
    tune_manifest: RoleManifest,
    *,
    initialization: ClassifierProvenance,
    classifier_fit_result: TrainingResult,
    readiness: ReadinessAudit,
    checkpoint_evaluator: Callable[
        [tuple[PrivateExamRecord, ...]], Sequence[ClassifierTuneCheckpoint]
    ],
) -> ClassifierProvenance:
    """Evaluate/select a fresh classifier only after tune-role authorization."""

    if initialization.kind != "public_pretrained_fresh" or initialization.workflow_complete:
        raise PermissionError("classifier selection requires a pinned public initialization")
    if not classifier_fit_result.actual_exposure_verified:
        raise ValueError("classifier-fit exposure has not been verified")
    if classifier_fit_result.manifest_sha256 not in readiness.manifest_sha256s:
        raise ValueError("classifier-fit result is absent from the readiness audit")

    def authorized(records: tuple[PrivateExamRecord, ...]) -> ClassifierProvenance:
        checkpoints = tuple(checkpoint_evaluator(records))
        if not checkpoints:
            raise ValueError("classifier tune selection requires checkpoint results")
        if len({item.epoch for item in checkpoints}) != len(checkpoints):
            raise ValueError("classifier tune results contain duplicate epochs")
        selected = min(checkpoints, key=lambda item: (item.tune_nll, item.epoch))
        selection_sha256 = _sha256_json(
            {
                "metric": "multiclass_nll",
                "role": Role.TUNE.value,
                "manifest_sha256": tune_manifest.manifest_sha256,
                "candidates": [
                    {
                        "epoch": item.epoch,
                        "artifact_sha256": item.artifact_sha256,
                        "tune_nll": item.tune_nll,
                    }
                    for item in checkpoints
                ],
                "selected_artifact_sha256": selected.artifact_sha256,
            }
        )
        return ClassifierProvenance.fresh_selected(
            initialization,
            checkpoint_sha256=selected.artifact_sha256,
            classifier_fit_manifest_sha256=classifier_fit_result.manifest_sha256,
            classifier_fit_update_count=classifier_fit_result.update_count,
            tune_selection_sha256=selection_sha256,
            tune_manifest_sha256=tune_manifest.manifest_sha256,
            readiness=readiness,
        )

    return run_with_role_access(
        tune_manifest,
        operation=Operation.TUNE_SELECTION,
        roles=(Role.TUNE,),
        loader=authorized,
    )


@dataclass(frozen=True)
class CheckpointTuneResult:
    method: str
    seed: int
    classifier_checkpoint_sha256: str
    tune_manifest_sha256: str
    epoch: int
    clean_aurc: float
    stress_cells: tuple[TuneCellResult, ...]
    artifact_sha256: str
    role: str = Role.TUNE.value

    def __post_init__(self) -> None:
        if self.method not in MANDATORY_METHODS:
            raise ValueError("checkpoint method is outside the frozen method table")
        if self.seed not in (42, 43, 44) or isinstance(self.seed, bool):
            raise ValueError("checkpoint training seed must be one of 42, 43, 44")
        if not _is_sha256(self.classifier_checkpoint_sha256):
            raise ValueError("checkpoint classifier identity must be SHA-256")
        if self.role != Role.TUNE.value or not _is_sha256(self.tune_manifest_sha256):
            raise ValueError("checkpoint result must bind a tune-role manifest SHA-256")
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int) or self.epoch < 1:
            raise ValueError("checkpoint epoch must be positive")
        if (
            isinstance(self.clean_aurc, bool)
            or not isinstance(self.clean_aurc, (int, float))
            or not math.isfinite(self.clean_aurc)
        ):
            raise ValueError("clean tune AURC must be finite")
        if self.clean_aurc < 0:
            raise ValueError("clean tune AURC must be nonnegative")
        if not _is_sha256(self.artifact_sha256):
            raise ValueError("tune checkpoint artifact identity must be SHA-256")
        materialized = tuple(self.stress_cells)
        _validate_tune_cells(materialized, require_complete=True)
        object.__setattr__(self, "stress_cells", materialized)

    @property
    def stress_aurc(self) -> float:
        return sum(cell.aurc for cell in self.stress_cells) / len(self.stress_cells)


def select_clean_reference(
    checkpoints: Sequence[CheckpointTuneResult],
) -> CheckpointTuneResult:
    """Select correctness-MV-ACN first using only clean tune AURC."""

    materialized = tuple(checkpoints)
    if not materialized:
        raise ValueError("clean reference selection requires checkpoints")
    if {item.method for item in materialized} != {"correctness_mvacn"}:
        raise ValueError("clean reference must be correctness-MVACN")
    if len({item.seed for item in materialized}) != 1 or len(
        {item.classifier_checkpoint_sha256 for item in materialized}
    ) != 1 or len({item.tune_manifest_sha256 for item in materialized}) != 1:
        raise ValueError(
            "clean reference checkpoints must share one seed, classifier, and tune manifest"
        )
    if len({item.epoch for item in materialized}) != len(materialized):
        raise ValueError("clean reference checkpoints contain duplicate epochs")
    return min(materialized, key=lambda item: (item.clean_aurc, item.epoch))


def select_candidate_checkpoint(
    checkpoints: Sequence[CheckpointTuneResult],
    *,
    clean_reference: CheckpointTuneResult,
    guardrail: float = TUNE_GUARDRAIL,
) -> CheckpointTuneResult | None:
    """Apply the clean guardrail, then stress/clean/epoch tie ordering."""

    if not isinstance(clean_reference, CheckpointTuneResult):
        raise TypeError("clean_reference must be selected before candidate selection")
    if clean_reference.method != "correctness_mvacn":
        raise ValueError("clean reference must be a selected correctness-MVACN checkpoint")
    if guardrail != TUNE_GUARDRAIL:
        raise ValueError("clean selection guardrail is frozen at 0.005")
    materialized = tuple(checkpoints)
    if materialized:
        if len({item.method for item in materialized}) != 1:
            raise ValueError("candidate selection requires exactly one method")
        if {item.seed for item in materialized} != {clean_reference.seed}:
            raise ValueError("candidate and clean reference training seeds must match")
        if {item.classifier_checkpoint_sha256 for item in materialized} != {
            clean_reference.classifier_checkpoint_sha256
        }:
            raise ValueError("candidate and clean reference must share the frozen classifier")
        if {item.tune_manifest_sha256 for item in materialized} != {
            clean_reference.tune_manifest_sha256
        }:
            raise ValueError("candidate and clean reference must share the tune manifest")
    if len({item.epoch for item in materialized}) != len(materialized):
        raise ValueError("candidate checkpoints contain duplicate epochs")
    eligible = tuple(
        item
        for item in materialized
        if item.clean_aurc <= clean_reference.clean_aurc + guardrail
    )
    if not eligible:
        return None
    return min(eligible, key=lambda item: (item.stress_aurc, item.clean_aurc, item.epoch))


def validate_checkpoint_selection_budget(
    checkpoints: Sequence[CheckpointTuneResult], search_table: FrozenSearchTable
) -> tuple[CheckpointTuneResult, ...]:
    """Require every and only prespecified epoch before a production selection."""

    materialized = tuple(checkpoints)
    if tuple(sorted(item.epoch for item in materialized)) != search_table.checkpoint_epochs:
        raise ValueError("checkpoint results do not match the frozen selection budget")
    return materialized


@dataclass(frozen=True)
class ModelArtifactSelection:
    method: str
    seed: int
    artifact_sha256: str

    def __post_init__(self) -> None:
        if self.method not in MANDATORY_METHODS:
            raise ValueError("selection method is not in the frozen method table")
        if self.seed not in (42, 43, 44) or isinstance(self.seed, bool):
            raise ValueError("selection seed must be one of 42, 43, 44")
        if not _is_sha256(self.artifact_sha256):
            raise ValueError("selected model artifact identity must be SHA-256")


@dataclass(frozen=True)
class PilotPlan:
    config_sha256: str
    search_table_sha256: str
    protocol_sha256: str
    metric_version: str
    dataset: str
    backbone: str
    methods: tuple[str, ...]
    seeds: tuple[int, ...]
    stress_seed: int
    manifest_sha256_by_role: tuple[tuple[str, str], ...]
    classifier: ClassifierProvenance
    selection_map: tuple[tuple[str, str], ...]
    evaluation_kind: str
    patient_ready: bool
    sha256: str
    source_path: Path
    schema_version: str = PILOT_PLAN_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PILOT_PLAN_VERSION:
            raise ValueError("unsupported pilot plan schema")
        source = Path(self.source_path).resolve()
        if source.suffix != ".json":
            raise ValueError("pilot plan source must be a JSON artifact")
        object.__setattr__(self, "source_path", source)
        for value in (self.config_sha256, self.search_table_sha256, self.protocol_sha256):
            if not _is_sha256(value):
                raise ValueError("pilot plan binding must use SHA-256")
        if not _is_sha256(self.sha256):
            raise ValueError("pilot plan identity must use SHA-256")
        if tuple(self.methods) != MANDATORY_METHODS or tuple(self.seeds) != (42, 43, 44):
            raise ValueError("pilot plan has a missing or changed method/seed table")
        if self.metric_version != METRIC_VERSION:
            raise ValueError("pilot plan metric version is not the accepted P4A binding")
        if self.dataset not in ("RSNA", "DDSM") or self.backbone != self.classifier.backbone:
            raise ValueError("pilot plan dataset/backbone binding is invalid")
        if not isinstance(self.patient_ready, bool):
            raise ValueError("pilot plan patient readiness must be boolean")
        if self.stress_seed != 4242:
            raise ValueError("pilot plan stress table seed must remain 4242")
        if self.classifier.kind == "original_finetuned_diagnostic":
            raise PermissionError("diagnostic original classifier cannot qualify a pilot plan")
        if not self.classifier.workflow_complete:
            raise PermissionError("fresh classifier fitting/tune workflow is incomplete")
        expected_kind = "real_pilot" if self.classifier.patient_readiness_verified else "synthetic_software"
        if self.evaluation_kind != expected_kind or self.patient_ready != (
            self.evaluation_kind == "real_pilot"
        ):
            raise ValueError("pilot plan readiness kind disagrees with classifier audit")
        role_map = dict(self.manifest_sha256_by_role)
        if set(role_map) != {role.value for role in Role} or any(
            not _is_sha256(value) for value in role_map.values()
        ):
            raise ValueError("pilot plan must bind every role manifest/lock identity")
        if (
            role_map[Role.CLASSIFIER_FIT.value]
            != self.classifier.classifier_fit_manifest_sha256
            or role_map[Role.TUNE.value] != self.classifier.tune_manifest_sha256
        ):
            raise ValueError("pilot plan role manifests disagree with fresh classifier exposure")
        selections = dict(self.selection_map)
        expected = {f"{method}:{seed}" for method in self.methods for seed in self.seeds}
        if set(selections) != expected or any(not _is_sha256(value) for value in selections.values()):
            raise ValueError("pilot plan has a missing or invalid model selection")

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "config_sha256": self.config_sha256,
            "search_table_sha256": self.search_table_sha256,
            "protocol_sha256": self.protocol_sha256,
            "metric_version": self.metric_version,
            "dataset": self.dataset,
            "backbone": self.backbone,
            "methods": list(self.methods),
            "seeds": list(self.seeds),
            "stress_seed": self.stress_seed,
            "manifest_sha256_by_role": dict(self.manifest_sha256_by_role),
            "classifier": self.classifier.to_dict(),
            "selection_map": dict(self.selection_map),
            "evaluation_kind": self.evaluation_kind,
            "patient_ready": self.patient_ready,
        }


def _plan_from_payload(payload: object, digest: str, source_path: str | Path) -> PilotPlan:
    if not isinstance(payload, dict):
        raise ValueError("pilot plan payload must be an object")
    expected = {
        "schema_version",
        "config_sha256",
        "search_table_sha256",
        "protocol_sha256",
        "metric_version",
        "dataset",
        "backbone",
        "methods",
        "seeds",
        "stress_seed",
        "manifest_sha256_by_role",
        "classifier",
        "selection_map",
        "evaluation_kind",
        "patient_ready",
    }
    if set(payload) != expected:
        raise ValueError("pilot plan payload has missing or unknown fields")
    try:
        raw_classifier = payload["classifier"]
        classifier_fields = {
            "kind",
            "backbone",
            "hf_model",
            "revision",
            "image_size",
            "hidden_size",
            "prompts",
            "preprocessing",
            "image_mean",
            "image_std",
            "checkpoint_sha256",
            "diagnostic_reason",
            "initialization_sha256",
            "classifier_fit_manifest_sha256",
            "classifier_fit_update_count",
            "tune_selection_sha256",
            "tune_manifest_sha256",
            "patient_readiness_verified",
            "readiness_audit_sha256",
        }
        if not isinstance(raw_classifier, dict) or set(raw_classifier) != classifier_fields:
            raise TypeError
        classifier = ClassifierProvenance(
            kind=raw_classifier["kind"],
            backbone=raw_classifier["backbone"],
            hf_model=raw_classifier["hf_model"],
            revision=raw_classifier["revision"],
            image_size=raw_classifier["image_size"],
            hidden_size=raw_classifier["hidden_size"],
            prompts=tuple(raw_classifier["prompts"]),
            preprocessing=raw_classifier["preprocessing"],
            image_mean=tuple(raw_classifier["image_mean"]),
            image_std=tuple(raw_classifier["image_std"]),
            checkpoint_sha256=raw_classifier["checkpoint_sha256"],
            diagnostic_reason=raw_classifier["diagnostic_reason"],
            initialization_sha256=raw_classifier["initialization_sha256"],
            classifier_fit_manifest_sha256=raw_classifier[
                "classifier_fit_manifest_sha256"
            ],
            classifier_fit_update_count=raw_classifier["classifier_fit_update_count"],
            tune_selection_sha256=raw_classifier["tune_selection_sha256"],
            tune_manifest_sha256=raw_classifier["tune_manifest_sha256"],
            patient_readiness_verified=raw_classifier["patient_readiness_verified"],
            readiness_audit_sha256=raw_classifier["readiness_audit_sha256"],
        )
        manifests = payload["manifest_sha256_by_role"]
        selections = payload["selection_map"]
        if not isinstance(manifests, dict) or not isinstance(selections, dict):
            raise TypeError
        return PilotPlan(
            config_sha256=payload["config_sha256"],
            search_table_sha256=payload["search_table_sha256"],
            protocol_sha256=payload["protocol_sha256"],
            metric_version=payload["metric_version"],
            dataset=payload["dataset"],
            backbone=payload["backbone"],
            methods=tuple(payload["methods"]),
            seeds=tuple(payload["seeds"]),
            stress_seed=payload["stress_seed"],
            manifest_sha256_by_role=tuple(sorted(manifests.items())),
            classifier=classifier,
            selection_map=tuple(sorted(selections.items())),
            evaluation_kind=payload["evaluation_kind"],
            patient_ready=payload["patient_ready"],
            sha256=digest,
            source_path=Path(source_path).resolve(),
            schema_version=payload["schema_version"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("pilot plan payload schema is invalid") from exc


def freeze_pilot_plan(
    path: str | Path,
    *,
    config: ResearchRunConfig,
    manifest_sha256_by_role: Mapping[str, str],
    classifier: ClassifierProvenance,
    selections: Sequence[ModelArtifactSelection],
) -> PilotPlan:
    """Create, validate, and exclusively publish an immutable pilot plan."""

    if not isinstance(config, ResearchRunConfig):
        raise TypeError("config must be a validated ResearchRunConfig")
    if classifier.kind == "original_finetuned_diagnostic":
        raise PermissionError("diagnostic original classifier cannot qualify a pilot plan")
    if not classifier.workflow_complete:
        raise PermissionError("fresh classifier fitting/tune workflow is incomplete")
    materialized = tuple(selections)
    selection_map = tuple(
        sorted((f"{selection.method}:{selection.seed}", selection.artifact_sha256) for selection in materialized)
    )
    if len(dict(selection_map)) != len(selection_map):
        raise ValueError("pilot plan contains duplicate model selections")
    target = Path(path).resolve()
    if target.suffix != ".json" or not target.parent.exists():
        raise ValueError("pilot plan must be a JSON file in an existing directory")
    provisional = PilotPlan(
        config_sha256=config.sha256,
        search_table_sha256=config.search_table.sha256,
        protocol_sha256=config.protocol_sha256,
        metric_version=config.metric_version,
        dataset=config.dataset,
        backbone=config.backbone,
        methods=config.methods,
        seeds=config.seeds,
        stress_seed=config.stress_seed,
        manifest_sha256_by_role=tuple(sorted(manifest_sha256_by_role.items())),
        classifier=classifier,
        selection_map=selection_map,
        evaluation_kind=(
            "real_pilot" if classifier.patient_readiness_verified else "synthetic_software"
        ),
        patient_ready=classifier.patient_readiness_verified,
        sha256="0" * 64,
        source_path=target,
    )
    payload = provisional.payload()
    digest = _sha256_json(payload)
    plan = _plan_from_payload(payload, digest, target)
    staging = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    _require_external_or_ignored_destination(target)
    _require_external_or_ignored_destination(staging)
    document = {
        "envelope_version": PILOT_PLAN_ENVELOPE_VERSION,
        "plan_sha256": digest,
        "plan": payload,
    }
    try:
        with staging.open("xb") as handle:
            handle.write(_canonical_json(document) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(staging, target)
        except FileExistsError:
            raise
        target.chmod(0o600)
    finally:
        staging.unlink(missing_ok=True)
    return plan


def load_pilot_plan(
    path: str | Path, *, expected_config_sha256: str | None = None
) -> PilotPlan:
    """Verify the immutable plan envelope and all internal bindings."""

    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("pilot plan is unavailable or invalid") from exc
    if not isinstance(document, dict) or set(document) != {
        "envelope_version",
        "plan_sha256",
        "plan",
    }:
        raise ValueError("pilot plan envelope is invalid")
    if document["envelope_version"] != PILOT_PLAN_ENVELOPE_VERSION:
        raise ValueError("pilot plan envelope version is invalid")
    digest = _sha256_json(document["plan"])
    if document["plan_sha256"] != digest:
        raise ValueError("pilot plan integrity check failed")
    plan = _plan_from_payload(document["plan"], digest, path)
    if expected_config_sha256 is not None and plan.config_sha256 != expected_config_sha256:
        raise ValueError("pilot plan configuration binding is stale")
    return plan


@dataclass(frozen=True)
class PilotEvaluationSummary:
    method: str
    seed: int
    panel: str
    mean_aurc: float
    patient_count: int
    exam_count: int
    patient_ready: bool = False

    def __post_init__(self) -> None:
        if self.method not in MANDATORY_METHODS:
            raise ValueError("pilot summary method is not frozen in the plan")
        if self.panel not in (
            "primary",
            "clean_four_view",
            "clean_masks",
            "strong_seen",
            "common_mode",
        ):
            raise ValueError("pilot summary panel is unknown")
        if not math.isfinite(self.mean_aurc) or self.mean_aurc < 0:
            raise ValueError("pilot summary AURC must be finite and nonnegative")
        if self.patient_count < 1 or self.exam_count < self.patient_count:
            raise ValueError("pilot summary aggregate counts are invalid")


def evaluate_pilot_with_role_access(
    plan: PilotPlan,
    *,
    manifest: RoleManifest,
    role: Role,
    method: str,
    seed: int,
    model_sha256: str,
    outcome_reader: Callable[[tuple[PrivateExamRecord, ...]], PilotEvaluationSummary],
) -> PilotEvaluationSummary:
    """Verify plan/model bindings, then authorize pilot rows before reading outcomes."""

    if not isinstance(plan, PilotPlan):
        raise TypeError("plan must be a verified PilotPlan")
    current_plan = load_pilot_plan(
        plan.source_path, expected_config_sha256=plan.config_sha256
    )
    if current_plan.sha256 != plan.sha256:
        raise ValueError("pilot plan changed after it was loaded")
    if role is not Role.PILOT:
        raise PermissionError("only the pilot role is available; locked outcomes remain inaccessible")
    expected_model = dict(plan.selection_map).get(f"{method}:{seed}")
    if expected_model is None or model_sha256 != expected_model:
        raise ValueError("pilot model binding is missing or stale")
    if dict(plan.manifest_sha256_by_role)[Role.PILOT.value] != manifest.manifest_sha256:
        raise ValueError("pilot manifest binding is stale")

    def authorized(records: tuple[PrivateExamRecord, ...]) -> PilotEvaluationSummary:
        result = outcome_reader(records)
        if not isinstance(result, PilotEvaluationSummary):
            raise TypeError("outcome reader must return an aggregate PilotEvaluationSummary")
        if result.method != method or result.seed != seed:
            raise ValueError("pilot aggregate disagrees with the frozen method/seed binding")
        if result.patient_ready != plan.patient_ready:
            raise ValueError("pilot aggregate readiness disagrees with the frozen plan")
        return result

    return run_with_role_access(
        manifest,
        operation=Operation.PILOT_EVALUATION,
        roles=(Role.PILOT,),
        loader=authorized,
    )


def evaluate_prediction_panel_with_role_access(
    plan: PilotPlan,
    *,
    manifest: RoleManifest,
    role: Role,
    method: str,
    seed: int,
    model_sha256: str,
    prediction_reader: Callable[
        [tuple[PrivateExamRecord, ...]], Sequence[EvaluationPrediction]
    ],
) -> PilotEvaluationSummary:
    """Authorize rows, validate frozen identities, and aggregate through P4A."""

    def read_and_aggregate(
        records: tuple[PrivateExamRecord, ...],
    ) -> PilotEvaluationSummary:
        predictions = tuple(prediction_reader(records))
        authorized = {record.exam_key: record for record in records}
        if not predictions:
            raise ValueError("pilot prediction panel is empty")
        for prediction in predictions:
            record = authorized.get(prediction.exam_id)
            if record is None:
                raise ValueError("pilot prediction contains an unauthorized exam")
            if (
                prediction.patient_id != record.patient_key
                or prediction.target != record.density
                or prediction.role != Role.PILOT.value
                or prediction.cohort != manifest.manifest_sha256
                or prediction.method != method
                or prediction.training_seed != seed
            ):
                raise ValueError("pilot prediction row disagrees with the frozen plan/manifest")
        aggregate = evaluate_aurc_panel(predictions)
        return PilotEvaluationSummary(
            method=method,
            seed=seed,
            panel=aggregate.panel,
            mean_aurc=aggregate.mean_aurc,
            patient_count=aggregate.n_patients,
            exam_count=aggregate.n_exams,
            patient_ready=plan.patient_ready,
        )

    return evaluate_pilot_with_role_access(
        plan,
        manifest=manifest,
        role=role,
        method=method,
        seed=seed,
        model_sha256=model_sha256,
        outcome_reader=read_and_aggregate,
    )


__all__ = [
    "CheckpointTuneResult",
    "ClassifierTuneCheckpoint",
    "ModelArtifactSelection",
    "PilotEvaluationSummary",
    "PilotPlan",
    "TuneCellResult",
    "evaluate_pilot_with_role_access",
    "evaluate_prediction_panel_with_role_access",
    "evaluate_tune_cells_with_role_access",
    "freeze_pilot_plan",
    "load_pilot_plan",
    "select_candidate_checkpoint",
    "select_clean_reference",
    "select_fresh_classifier_on_tune",
    "validate_checkpoint_selection_budget",
]
