"""Guarded tune selection and immutable pilot-evaluation orchestration."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence, TypeVar

from mmdc_clip_f.provenance import sha256_file

from .artifacts import ControlArtifact, load_control_artifact
from .cache import _require_external_or_ignored_destination
from .inputs import CANONICAL_VIEWS
from .metrics import EvaluationPrediction, confidence_panel_metrics, evaluate_aurc_panel
from .perturbations import FITTING_FAMILIES, FITTING_SEVERITIES
from .roles import Operation, PrivateExamRecord, Role, RoleManifest, run_with_role_access
from .training import (
    LEARNED_TORCH_METHODS,
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
PREDICTION_SET_VERSION = "view-risk-authoritative-predictions/v1"
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


@dataclass(frozen=True)
class TunePrediction:
    exam_id: str
    target: int
    prediction: int
    confidence: float
    family: str | None = None
    severity: str | None = None
    target_view: str | None = None

    @property
    def cell_key(self) -> tuple[str | None, str | None, str | None]:
        return self.family, self.severity, self.target_view


def generate_checkpoint_tune_result_with_role_access(
    manifest: RoleManifest,
    *,
    method: str,
    seed: int,
    classifier_checkpoint_sha256: str,
    epoch: int,
    artifact_path: str | Path,
    prediction_reader: Callable[[tuple[PrivateExamRecord, ...]], Sequence[TunePrediction]],
) -> "CheckpointTuneResult":
    """Generate clean and balanced stress AURCs from authorized tune predictions."""

    def authorized(records: tuple[PrivateExamRecord, ...]) -> CheckpointTuneResult:
        rows = tuple(prediction_reader(records))
        expected_exams = {record.exam_key: record for record in records}
        grouped: dict[tuple[str | None, str | None, str | None], list[TunePrediction]] = {}
        for row in rows:
            if not isinstance(row, TunePrediction) or row.exam_id not in expected_exams:
                raise ValueError("tune prediction contains an unauthorized exam")
            record = expected_exams[row.exam_id]
            if row.target != record.density or row.prediction not in range(4):
                raise ValueError("tune prediction target/classifier binding is stale")
            grouped.setdefault(row.cell_key, []).append(row)
        clean_key = (None, None, None)
        expected_cells = {
            (family, severity, view)
            for family in FITTING_FAMILIES
            for severity in FITTING_SEVERITIES
            for view in CANONICAL_VIEWS
        }
        if set(grouped) != {clean_key, *expected_cells}:
            raise ValueError("tune predictions must contain clean plus the exact balanced 16 cells")
        for cell_rows in grouped.values():
            if {row.exam_id for row in cell_rows} != set(expected_exams) or len(cell_rows) != len(
                expected_exams
            ):
                raise ValueError("every tune cell must contain the exact authorized exam cohort")

        def aurc(cell_rows: Sequence[TunePrediction]) -> float:
            return confidence_panel_metrics(
                [int(row.prediction == row.target) for row in cell_rows],
                [row.confidence for row in cell_rows],
                confidence_kind="ranking",
            ).aurc

        stress = tuple(
            TuneCellResult(family, severity, view, aurc(grouped[(family, severity, view)]))
            for family, severity, view in sorted(expected_cells)
        )
        artifact = Path(artifact_path).resolve()
        _require_external_or_ignored_destination(artifact)
        return CheckpointTuneResult(
            method=method,
            seed=seed,
            classifier_checkpoint_sha256=classifier_checkpoint_sha256,
            tune_manifest_sha256=manifest.manifest_sha256,
            epoch=epoch,
            clean_aurc=aurc(grouped[clean_key]),
            stress_cells=stress,
            artifact_sha256=sha256_file(artifact),
        )

    return run_with_role_access(
        manifest,
        operation=Operation.TUNE_SELECTION,
        roles=(Role.TUNE,),
        loader=authorized,
    )


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

    if initialization.kind not in (
        "public_pretrained_fresh",
        "synthetic_injected",
    ) or initialization.workflow_complete:
        raise PermissionError("classifier selection requires a fresh production or synthetic initialization")
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


_SELECTION_TOKEN = object()


@dataclass(frozen=True, init=False)
class ModelArtifactSelection:
    method: str
    seed: int
    artifact_sha256: str
    artifact_path: Path
    evidence_kind: str
    config_sha256: str
    classifier_checkpoint_sha256: str
    tune_manifest_sha256: str
    search_table_sha256: str
    selection_trials: int
    eligible: bool
    reference_artifact_sha256: str
    workflow_evidence_sha256: str
    workflow_evidence_path: Path
    confidence_fit_manifest_sha256: str | None

    def __init__(
        self,
        *,
        method: str,
        seed: int,
        artifact_path: str | Path,
        evidence_kind: str,
        config_sha256: str,
        classifier_checkpoint_sha256: str,
        tune_manifest_sha256: str,
        search_table_sha256: str,
        selection_trials: int,
        eligible: bool,
        reference_artifact_sha256: str,
        workflow_evidence_sha256: str,
        workflow_evidence_path: str | Path,
        confidence_fit_manifest_sha256: str | None,
        _factory_token: object,
    ) -> None:
        if _factory_token is not _SELECTION_TOKEN:
            raise RuntimeError("model selections must come from a verified artifact workflow")
        if method not in MANDATORY_METHODS:
            raise ValueError("selection method is not in the frozen method table")
        if seed not in (42, 43, 44) or isinstance(seed, bool):
            raise ValueError("selection seed must be one of 42, 43, 44")
        path = Path(artifact_path).resolve()
        evidence_path = Path(workflow_evidence_path).resolve()
        _require_external_or_ignored_destination(path)
        _require_external_or_ignored_destination(evidence_path)
        digest = sha256_file(path)
        for value in (
            config_sha256,
            classifier_checkpoint_sha256,
            tune_manifest_sha256,
            search_table_sha256,
            reference_artifact_sha256,
            workflow_evidence_sha256,
        ):
            if not _is_sha256(value):
                raise ValueError("selection workflow bindings must be SHA-256")
        if isinstance(selection_trials, bool) or not isinstance(selection_trials, int) or selection_trials < 0:
            raise ValueError("selection trial count must be a nonnegative integer")
        if not isinstance(eligible, bool) or not eligible:
            raise ValueError("incomplete or no-eligible selections cannot be frozen")
        values = {
            "method": method,
            "seed": seed,
            "artifact_sha256": digest,
            "artifact_path": path,
            "evidence_kind": evidence_kind,
            "config_sha256": config_sha256,
            "classifier_checkpoint_sha256": classifier_checkpoint_sha256,
            "tune_manifest_sha256": tune_manifest_sha256,
            "search_table_sha256": search_table_sha256,
            "selection_trials": selection_trials,
            "eligible": eligible,
            "reference_artifact_sha256": reference_artifact_sha256,
            "workflow_evidence_sha256": workflow_evidence_sha256,
            "workflow_evidence_path": evidence_path,
            "confidence_fit_manifest_sha256": confidence_fit_manifest_sha256,
        }
        if confidence_fit_manifest_sha256 is not None and not _is_sha256(
            confidence_fit_manifest_sha256
        ):
            raise ValueError("confidence-fit selection binding must be SHA-256")
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @classmethod
    def synthetic_from_artifact(
        cls,
        artifact_path: str | Path,
        *,
        method: str,
        seed: int,
        config: ResearchRunConfig,
        classifier: ClassifierProvenance,
    ) -> "ModelArtifactSelection":
        if classifier.kind != "synthetic_injected":
            raise PermissionError("synthetic selection evidence requires a synthetic classifier")
        digest = sha256_file(Path(artifact_path).resolve())
        return cls(
            method=method,
            seed=seed,
            artifact_path=artifact_path,
            evidence_kind="synthetic_software",
            config_sha256=config.sha256,
            classifier_checkpoint_sha256=classifier.checkpoint_sha256,
            tune_manifest_sha256=classifier.tune_manifest_sha256,
            search_table_sha256=config.search_table.sha256,
            selection_trials=(
                config.epochs
                if method in LEARNED_TORCH_METHODS
                else 1
                if method == "temperature_scaled_msp"
                else len(config.search_table.ds_regularizations)
                if method == "ds_logistic"
                else 0
            ),
            eligible=True,
            reference_artifact_sha256=digest,
            workflow_evidence_sha256=digest,
            workflow_evidence_path=artifact_path,
            confidence_fit_manifest_sha256=None,
            _factory_token=_SELECTION_TOKEN,
        )

    @classmethod
    def verified_from_artifact(
        cls,
        artifact_path: str | Path,
        *,
        method: str,
        seed: int,
        config: ResearchRunConfig,
        classifier: ClassifierProvenance,
        tune_manifest_sha256: str,
        selection_trials: int,
        reference_artifact_sha256: str,
        workflow_evidence_path: str | Path,
        evidence_kind: str,
        eligible: bool,
    ) -> "ModelArtifactSelection":
        raise PermissionError(
            "production selections must be created from typed training/control workflow evidence"
        )

    @classmethod
    def from_control_artifact(
        cls,
        control: ControlArtifact,
        *,
        config: ResearchRunConfig,
        classifier: ClassifierProvenance,
        clean_reference: "ModelArtifactSelection",
    ) -> "ModelArtifactSelection":
        """Create an analytic selection only from a verified fitted/raw control artifact."""

        if not classifier.pilot_eligible:
            raise PermissionError("real selection evidence requires a pilot-eligible fresh classifier")
        current = load_control_artifact(control.path)
        if current.sha256 != control.sha256:
            raise ValueError("control artifact content changed before selection")
        if (
            current.config_sha256 != config.sha256
            or current.classifier_checkpoint_sha256 != classifier.checkpoint_sha256
            or current.tune_manifest_sha256 != classifier.tune_manifest_sha256
        ):
            raise ValueError("control artifact workflow binding is stale")
        if (
            clean_reference.method != "correctness_mvacn"
            or clean_reference.seed != current.seed
            or clean_reference.classifier_checkpoint_sha256 != classifier.checkpoint_sha256
        ):
            raise ValueError("analytic control requires the same-seed clean reference")
        expected_kind = {
            "temperature_scaled_msp": "temperature_tune_nll",
            "ds_logistic": "ds_confidence_fit_tune_regularization",
            "msp": "analytic_raw_ranking",
            "margin": "analytic_raw_ranking",
            "negative_entropy": "analytic_raw_ranking",
            "energy": "analytic_raw_ranking",
            "absolute_omission_sensitivity": "analytic_raw_ranking",
        }
        if expected_kind.get(current.method) != current.evidence_kind:
            raise ValueError("control artifact fitting evidence disagrees with its method")
        return cls(
            method=current.method,
            seed=current.seed,
            artifact_path=current.path,
            evidence_kind=current.evidence_kind,
            config_sha256=config.sha256,
            classifier_checkpoint_sha256=classifier.checkpoint_sha256,
            tune_manifest_sha256=current.tune_manifest_sha256,
            search_table_sha256=config.search_table.sha256,
            selection_trials=current.selection_trials,
            eligible=True,
            reference_artifact_sha256=clean_reference.artifact_sha256,
            workflow_evidence_sha256=sha256_file(current.path),
            workflow_evidence_path=current.path,
            confidence_fit_manifest_sha256=current.confidence_manifest_sha256,
            _factory_token=_SELECTION_TOKEN,
        )

    @classmethod
    def from_learned_workflow(
        cls,
        *,
        artifact_path: str | Path,
        evidence_path: str | Path,
        config: ResearchRunConfig,
        classifier: ClassifierProvenance,
        training_result: TrainingResult,
        checkpoints: Sequence[CheckpointTuneResult],
        clean_reference: CheckpointTuneResult,
    ) -> "ModelArtifactSelection":
        """Replay the frozen budget/guardrail and bind its real checkpoint bytes."""

        if not classifier.workflow_complete:
            raise PermissionError("learned selection requires a completed fresh classifier")
        materialized = validate_checkpoint_selection_budget(checkpoints, config.search_table)
        if not materialized:
            raise ValueError("learned selection has no checkpoint evidence")
        method = materialized[0].method
        seed = materialized[0].seed
        if any(item.method != method or item.seed != seed for item in materialized):
            raise ValueError("learned checkpoint evidence mixes methods or seeds")
        if method not in LEARNED_TORCH_METHODS:
            raise ValueError("learned workflow evidence requires a learned method")
        if (
            training_result.method != method
            or training_result.seed != seed
            or training_result.completed_epoch != config.epochs
            or training_result.selection_trial_budget != config.epochs
            or not training_result.actual_exposure_verified
            or training_result.update_count < config.epochs
            or training_result.exposed_record_count < 1
            or training_result.parameter_count < 1
        ):
            raise ValueError("learned training result is incomplete or budget-mismatched")
        if classifier.kind == "public_pretrained_fresh":
            classifier.require_pilot_eligible()
            if training_result.software_only:
                raise ValueError("production selection cannot use software-only training evidence")
        elif not training_result.software_only:
            raise ValueError("synthetic training evidence must remain software-only")
        if (
            {item.classifier_checkpoint_sha256 for item in materialized}
            != {classifier.checkpoint_sha256}
            or {item.tune_manifest_sha256 for item in materialized}
            != {classifier.tune_manifest_sha256}
        ):
            raise ValueError("learned tune evidence has stale classifier/manifest bindings")
        if method == "correctness_mvacn":
            selected = select_clean_reference(materialized)
            if selected != clean_reference:
                raise ValueError("clean reference was not selected from this exact workflow")
        else:
            selected = select_candidate_checkpoint(
                materialized, clean_reference=clean_reference
            )
            if selected is None:
                raise ValueError("learned workflow has no guardrail-eligible checkpoint")
        artifact = Path(artifact_path).resolve()
        _require_external_or_ignored_destination(artifact)
        if sha256_file(artifact) != selected.artifact_sha256:
            raise ValueError("selected checkpoint bytes disagree with tune evidence")
        evidence = Path(evidence_path).resolve()
        if evidence.suffix != ".json" or not evidence.parent.exists():
            raise ValueError("learned workflow evidence must be JSON in an existing directory")
        staging = evidence.with_name(f".{evidence.name}.{os.getpid()}.tmp")
        for candidate in (evidence, staging):
            _require_external_or_ignored_destination(candidate)
        payload = {
            "schema_version": "view-risk-learned-selection-evidence/v1",
            "method": method,
            "seed": seed,
            "config_sha256": config.sha256,
            "search_table_sha256": config.search_table.sha256,
            "classifier_checkpoint_sha256": classifier.checkpoint_sha256,
            "confidence_fit_manifest_sha256": training_result.manifest_sha256,
            "tune_manifest_sha256": classifier.tune_manifest_sha256,
            "training_result": training_result.to_dict(),
            "reference": {
                "epoch": clean_reference.epoch,
                "artifact_sha256": clean_reference.artifact_sha256,
                "clean_aurc": clean_reference.clean_aurc,
            },
            "selected": {
                "epoch": selected.epoch,
                "artifact_sha256": selected.artifact_sha256,
                "clean_aurc": selected.clean_aurc,
                "stress_aurc": selected.stress_aurc,
            },
            "checkpoint_results_sha256": _sha256_json(
                [
                    {
                        "epoch": item.epoch,
                        "artifact_sha256": item.artifact_sha256,
                        "clean_aurc": item.clean_aurc,
                        "stress_cells": [cell.__dict__ for cell in item.stress_cells],
                    }
                    for item in materialized
                ]
            ),
            "selection_trials": len(materialized),
            "eligible": True,
        }
        try:
            with staging.open("xb") as handle:
                handle.write(_canonical_json(payload) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.link(staging, evidence)
            evidence.chmod(0o600)
        finally:
            staging.unlink(missing_ok=True)
        return cls(
            method=method,
            seed=seed,
            artifact_path=artifact,
            evidence_kind=(
                "learned_epoch_selection"
                if classifier.kind == "public_pretrained_fresh"
                else "synthetic_software"
            ),
            config_sha256=config.sha256,
            classifier_checkpoint_sha256=classifier.checkpoint_sha256,
            tune_manifest_sha256=classifier.tune_manifest_sha256,
            search_table_sha256=config.search_table.sha256,
            selection_trials=len(materialized),
            eligible=True,
            reference_artifact_sha256=clean_reference.artifact_sha256,
            workflow_evidence_sha256=sha256_file(evidence),
            workflow_evidence_path=evidence,
            confidence_fit_manifest_sha256=training_result.manifest_sha256,
            _factory_token=_SELECTION_TOKEN,
        )

    def verify_current_artifact(self) -> None:
        if sha256_file(self.artifact_path) != self.artifact_sha256:
            raise ValueError("selected model artifact content changed after verification")
        if sha256_file(self.workflow_evidence_path) != self.workflow_evidence_sha256:
            raise ValueError("selection workflow evidence changed after verification")


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
            "public_weight_content_sha256",
        }
        if not isinstance(raw_classifier, dict) or set(raw_classifier) != classifier_fields:
            raise TypeError
        classifier = ClassifierProvenance._restore_frozen_record(
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
            public_weight_content_sha256=raw_classifier["public_weight_content_sha256"],
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
    for selection in materialized:
        if not isinstance(selection, ModelArtifactSelection):
            raise TypeError("pilot selections must be verified artifact records")
        selection.verify_current_artifact()
        if (
            selection.config_sha256 != config.sha256
            or selection.search_table_sha256 != config.search_table.sha256
            or selection.classifier_checkpoint_sha256 != classifier.checkpoint_sha256
            or selection.tune_manifest_sha256 != classifier.tune_manifest_sha256
        ):
            raise ValueError("selection evidence has a stale workflow binding")
        if (
            selection.confidence_fit_manifest_sha256 is not None
            and selection.confidence_fit_manifest_sha256
            != manifest_sha256_by_role.get(Role.CONFIDENCE_FIT.value)
        ):
            raise ValueError("selection confidence-fit exposure binding is stale")
        if classifier.kind == "synthetic_injected":
            if selection.evidence_kind != "synthetic_software":
                raise ValueError("synthetic plans require explicit synthetic selection evidence")
        elif selection.evidence_kind == "synthetic_software":
            raise PermissionError("synthetic selection evidence cannot qualify a real pilot")
        expected_trials = (
            config.epochs
            if selection.method in LEARNED_TORCH_METHODS
            else 1
            if selection.method == "temperature_scaled_msp"
            else len(config.search_table.ds_regularizations)
            if selection.method == "ds_logistic"
            else 0
        )
        if selection.selection_trials != expected_trials:
            raise ValueError("selection evidence disagrees with the frozen method budget")
    if classifier.kind == "public_pretrained_fresh":
        by_key = {(item.method, item.seed): item for item in materialized}
        for seed in config.seeds:
            reference = by_key.get(("correctness_mvacn", seed))
            if reference is None:
                raise ValueError("pilot plan is missing the clean reference selection")
            for method in config.methods:
                item = by_key.get((method, seed))
                if item is not None and item.reference_artifact_sha256 != reference.artifact_sha256:
                    raise ValueError("selection does not bind the same-seed clean reference")
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


@dataclass(frozen=True)
class FrozenClassifierPrediction:
    """One label-free classifier decision for a frozen realized pilot input."""

    exam_id: str
    panel: str
    prediction: int
    family: str | None = None
    severity: str | None = None
    target_view: str | None = None
    variant: str | None = None
    mask: tuple[str, ...] = tuple(CANONICAL_VIEWS)
    realization_id: str = "frozen"

    def __post_init__(self) -> None:
        if not isinstance(self.exam_id, str) or not self.exam_id:
            raise ValueError("classifier prediction exam identity is invalid")
        if self.panel not in (
            "primary",
            "clean_four_view",
            "clean_masks",
            "strong_seen",
            "common_mode",
        ):
            raise ValueError("classifier prediction panel is unknown")
        if isinstance(self.prediction, bool) or self.prediction not in range(4):
            raise ValueError("classifier prediction must be a four-class index")
        mask = tuple(self.mask)
        if not mask or mask != tuple(view for view in CANONICAL_VIEWS if view in mask):
            raise ValueError("classifier prediction mask must be canonical and nonempty")
        if not isinstance(self.realization_id, str) or not self.realization_id:
            raise ValueError("classifier prediction realization identity is invalid")
        object.__setattr__(self, "mask", mask)

    @property
    def cell_key(self) -> tuple[object, ...]:
        return (
            self.panel,
            self.family,
            self.severity,
            self.target_view,
            self.variant,
            self.mask,
            self.realization_id,
        )

    @property
    def row_key(self) -> tuple[object, ...]:
        return (self.exam_id, *self.cell_key)

    def to_dict(self) -> dict[str, object]:
        return {
            "exam_id": self.exam_id,
            "panel": self.panel,
            "prediction": self.prediction,
            "family": self.family,
            "severity": self.severity,
            "target_view": self.target_view,
            "variant": self.variant,
            "mask": list(self.mask),
            "realization_id": self.realization_id,
        }


_AUTHORITATIVE_PREDICTION_TOKEN = object()


@dataclass(frozen=True, init=False)
class AuthoritativePredictionSet:
    plan_sha256: str
    classifier_checkpoint_sha256: str
    manifest_sha256: str
    rows: tuple[FrozenClassifierPrediction, ...]
    sha256: str
    source_path: Path | None

    def __init__(
        self,
        *,
        plan_sha256: str,
        classifier_checkpoint_sha256: str,
        manifest_sha256: str,
        rows: Sequence[FrozenClassifierPrediction],
        source_path: str | Path | None,
        _factory_token: object,
    ) -> None:
        if _factory_token is not _AUTHORITATIVE_PREDICTION_TOKEN:
            raise RuntimeError("authoritative predictions require role-bound validation")
        for value in (plan_sha256, classifier_checkpoint_sha256, manifest_sha256):
            if not _is_sha256(value):
                raise ValueError("authoritative prediction bindings must be SHA-256")
        materialized = tuple(rows)
        if not materialized or len({row.row_key for row in materialized}) != len(materialized):
            raise ValueError("authoritative classifier predictions are empty or duplicated")
        payload = {
            "schema_version": PREDICTION_SET_VERSION,
            "plan_sha256": plan_sha256,
            "classifier_checkpoint_sha256": classifier_checkpoint_sha256,
            "manifest_sha256": manifest_sha256,
            "rows": [row.to_dict() for row in materialized],
        }
        object.__setattr__(self, "plan_sha256", plan_sha256)
        object.__setattr__(self, "classifier_checkpoint_sha256", classifier_checkpoint_sha256)
        object.__setattr__(self, "manifest_sha256", manifest_sha256)
        object.__setattr__(self, "rows", materialized)
        object.__setattr__(self, "sha256", _sha256_json(payload))
        object.__setattr__(
            self, "source_path", None if source_path is None else Path(source_path).resolve()
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": PREDICTION_SET_VERSION,
            "plan_sha256": self.plan_sha256,
            "classifier_checkpoint_sha256": self.classifier_checkpoint_sha256,
            "manifest_sha256": self.manifest_sha256,
            "rows": [row.to_dict() for row in self.rows],
        }


def _validate_complete_prediction_cohort(
    rows: Sequence[FrozenClassifierPrediction], records: Sequence[PrivateExamRecord]
) -> None:
    expected = {record.exam_key for record in records}
    by_cell: dict[tuple[object, ...], set[str]] = {}
    for row in rows:
        if row.exam_id not in expected:
            raise ValueError("classifier prediction contains an unauthorized exam")
        by_cell.setdefault(row.cell_key, set()).add(row.exam_id)
    if not by_cell or any(exams != expected for exams in by_cell.values()):
        raise ValueError("every evaluation cell must contain the exact authorized exam cohort")


def bind_authoritative_predictions_with_role_access(
    plan: PilotPlan,
    *,
    manifest: RoleManifest,
    role: Role,
    prediction_reader: Callable[
        [tuple[PrivateExamRecord, ...]], Sequence[FrozenClassifierPrediction]
    ],
) -> AuthoritativePredictionSet:
    """Bind post-freeze, label-free classifier predictions to every pilot cell."""

    current = load_pilot_plan(plan.source_path, expected_config_sha256=plan.config_sha256)
    if current.sha256 != plan.sha256:
        raise ValueError("pilot plan changed before classifier prediction binding")
    if role is not Role.PILOT:
        raise PermissionError("only pilot label-free predictions may be bound")
    if dict(plan.manifest_sha256_by_role)[Role.PILOT.value] != manifest.manifest_sha256:
        raise ValueError("pilot manifest binding is stale")

    def authorized(records: tuple[PrivateExamRecord, ...]) -> AuthoritativePredictionSet:
        rows = tuple(prediction_reader(records))
        if any(not isinstance(row, FrozenClassifierPrediction) for row in rows):
            raise TypeError("classifier prediction reader returned an invalid row")
        _validate_complete_prediction_cohort(rows, records)
        return AuthoritativePredictionSet(
            plan_sha256=plan.sha256,
            classifier_checkpoint_sha256=plan.classifier.checkpoint_sha256,
            manifest_sha256=manifest.manifest_sha256,
            rows=rows,
            source_path=None,
            _factory_token=_AUTHORITATIVE_PREDICTION_TOKEN,
        )

    return run_with_role_access(
        manifest,
        operation=Operation.PILOT_EVALUATION,
        roles=(Role.PILOT,),
        loader=authorized,
    )


def freeze_authoritative_predictions(
    path: str | Path,
    plan: PilotPlan,
    *,
    manifest: RoleManifest,
    role: Role,
    prediction_reader: Callable[
        [tuple[PrivateExamRecord, ...]], Sequence[FrozenClassifierPrediction]
    ],
) -> AuthoritativePredictionSet:
    """Exclusively persist the one reusable, private classifier-prediction set."""

    bound = bind_authoritative_predictions_with_role_access(
        plan, manifest=manifest, role=role, prediction_reader=prediction_reader
    )
    target = Path(path).resolve()
    if target.suffix != ".json" or not target.parent.exists():
        raise ValueError("classifier prediction artifact must be JSON in an existing directory")
    staging = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    _require_external_or_ignored_destination(target)
    _require_external_or_ignored_destination(staging)
    document = {"sha256": bound.sha256, "predictions": bound.payload()}
    try:
        with staging.open("xb") as handle:
            handle.write(_canonical_json(document) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(staging, target)
        target.chmod(0o600)
    finally:
        staging.unlink(missing_ok=True)
    return AuthoritativePredictionSet(
        plan_sha256=bound.plan_sha256,
        classifier_checkpoint_sha256=bound.classifier_checkpoint_sha256,
        manifest_sha256=bound.manifest_sha256,
        rows=bound.rows,
        source_path=target,
        _factory_token=_AUTHORITATIVE_PREDICTION_TOKEN,
    )


def load_authoritative_predictions_with_role_access(
    path: str | Path,
    plan: PilotPlan,
    *,
    manifest: RoleManifest,
    role: Role,
) -> AuthoritativePredictionSet:
    """Role-authorize, verify, and cohort-check a private prediction artifact."""

    if role is not Role.PILOT:
        raise PermissionError("only pilot classifier predictions may be loaded")

    def authorized(records: tuple[PrivateExamRecord, ...]) -> AuthoritativePredictionSet:
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
            payload = document["predictions"]
            if set(document) != {"sha256", "predictions"} or set(payload) != {
                "schema_version",
                "plan_sha256",
                "classifier_checkpoint_sha256",
                "manifest_sha256",
                "rows",
            }:
                raise ValueError
            rows = tuple(
                FrozenClassifierPrediction(**{**row, "mask": tuple(row["mask"])})
                for row in payload["rows"]
            )
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError("classifier prediction artifact is unavailable or invalid") from exc
        result = AuthoritativePredictionSet(
            plan_sha256=payload["plan_sha256"],
            classifier_checkpoint_sha256=payload["classifier_checkpoint_sha256"],
            manifest_sha256=payload["manifest_sha256"],
            rows=rows,
            source_path=path,
            _factory_token=_AUTHORITATIVE_PREDICTION_TOKEN,
        )
        if document["sha256"] != result.sha256:
            raise ValueError("classifier prediction artifact integrity check failed")
        if (
            result.plan_sha256 != plan.sha256
            or result.classifier_checkpoint_sha256 != plan.classifier.checkpoint_sha256
            or result.manifest_sha256 != manifest.manifest_sha256
        ):
            raise ValueError("classifier prediction artifact binding is stale")
        _validate_complete_prediction_cohort(result.rows, records)
        return result

    return run_with_role_access(
        manifest,
        operation=Operation.PILOT_EVALUATION,
        roles=(Role.PILOT,),
        loader=authorized,
    )


@dataclass(frozen=True)
class ConfidenceScore:
    row_key: tuple[object, ...]
    confidence: float
    confidence_kind: str


def generate_confidence_predictions_with_role_access(
    plan: PilotPlan,
    *,
    authoritative_predictions: AuthoritativePredictionSet,
    manifest: RoleManifest,
    role: Role,
    method: str,
    seed: int,
    score_reader: Callable[
        [tuple[PrivateExamRecord, ...], tuple[FrozenClassifierPrediction, ...]],
        Sequence[ConfidenceScore],
    ],
) -> tuple[EvaluationPrediction, ...]:
    """Generate target-bound P4A rows from label-free method confidence scores."""

    if (
        authoritative_predictions.plan_sha256 != plan.sha256
        or authoritative_predictions.classifier_checkpoint_sha256
        != plan.classifier.checkpoint_sha256
        or authoritative_predictions.manifest_sha256 != manifest.manifest_sha256
    ):
        raise ValueError("authoritative prediction binding is stale")
    if dict(plan.selection_map).get(f"{method}:{seed}") is None:
        raise ValueError("confidence method/seed is absent from the frozen plan")

    def authorized(records: tuple[PrivateExamRecord, ...]) -> tuple[EvaluationPrediction, ...]:
        values = tuple(score_reader(records, authoritative_predictions.rows))
        if any(not isinstance(value, ConfidenceScore) for value in values):
            raise TypeError("confidence scorer returned an invalid row")
        scores = {value.row_key: value for value in values}
        expected = {row.row_key for row in authoritative_predictions.rows}
        if len(scores) != len(values) or set(scores) != expected:
            raise ValueError("confidence scores must cover every authoritative row exactly once")
        record_map = {record.exam_key: record for record in records}
        result = []
        for frozen in authoritative_predictions.rows:
            record = record_map[frozen.exam_id]
            value = scores[frozen.row_key]
            result.append(
                EvaluationPrediction(
                    dataset=plan.dataset,
                    role=Role.PILOT.value,
                    cohort=manifest.manifest_sha256,
                    method=method,
                    training_seed=seed,
                    patient_id=record.patient_key,
                    exam_id=record.exam_key,
                    panel=frozen.panel,  # type: ignore[arg-type]
                    target=record.density,
                    prediction=frozen.prediction,
                    confidence=value.confidence,
                    confidence_kind=value.confidence_kind,  # type: ignore[arg-type]
                    family=frozen.family,
                    severity=frozen.severity,
                    target_view=frozen.target_view,
                    variant=frozen.variant,
                    mask=frozen.mask,
                    realization_id=frozen.realization_id,
                )
            )
        return tuple(result)

    return run_with_role_access(
        manifest,
        operation=Operation.PILOT_EVALUATION,
        roles=(role,),
        loader=authorized,
    )


def evaluate_pilot_with_role_access(
    plan: PilotPlan,
    *,
    manifest: RoleManifest,
    role: Role,
    method: str,
    seed: int,
    model_sha256: str,
    model_artifact_path: str | Path | None = None,
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
    if plan.patient_ready:
        if model_artifact_path is None:
            raise ValueError("real pilot evaluation requires the selected model artifact")
        artifact = Path(model_artifact_path).resolve()
        _require_external_or_ignored_destination(artifact)
        if sha256_file(artifact) != expected_model:
            raise ValueError("selected model artifact content changed after plan freeze")
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
    authoritative_predictions: AuthoritativePredictionSet,
    manifest: RoleManifest,
    role: Role,
    method: str,
    seed: int,
    model_sha256: str,
    model_artifact_path: str | Path | None = None,
    prediction_reader: Callable[
        [tuple[PrivateExamRecord, ...]], Sequence[EvaluationPrediction]
    ],
) -> PilotEvaluationSummary:
    """Authorize rows, validate frozen identities, and aggregate through P4A."""

    if (
        not isinstance(authoritative_predictions, AuthoritativePredictionSet)
        or authoritative_predictions.plan_sha256 != plan.sha256
        or authoritative_predictions.classifier_checkpoint_sha256
        != plan.classifier.checkpoint_sha256
        or authoritative_predictions.manifest_sha256 != manifest.manifest_sha256
    ):
        raise ValueError("authoritative classifier prediction binding is stale or missing")

    def read_and_aggregate(
        records: tuple[PrivateExamRecord, ...],
    ) -> PilotEvaluationSummary:
        predictions = tuple(prediction_reader(records))
        authorized = {record.exam_key: record for record in records}
        authoritative = {row.row_key: row.prediction for row in authoritative_predictions.rows}
        if not predictions:
            raise ValueError("pilot prediction panel is empty")
        supplied_keys: set[tuple[object, ...]] = set()
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
            key = (
                prediction.exam_id,
                prediction.panel,
                prediction.family,
                prediction.severity,
                prediction.target_view,
                prediction.variant,
                prediction.mask,
                prediction.realization_id,
            )
            if key in supplied_keys or authoritative.get(key) != prediction.prediction:
                raise ValueError("confidence row disagrees with authoritative classifier predictions")
            supplied_keys.add(key)
        if supplied_keys != set(authoritative):
            raise ValueError("confidence predictions omit or add frozen evaluation rows")
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
        model_artifact_path=model_artifact_path,
        outcome_reader=read_and_aggregate,
    )


__all__ = [
    "AuthoritativePredictionSet",
    "CheckpointTuneResult",
    "ClassifierTuneCheckpoint",
    "ConfidenceScore",
    "FrozenClassifierPrediction",
    "ModelArtifactSelection",
    "PilotEvaluationSummary",
    "PilotPlan",
    "TuneCellResult",
    "TunePrediction",
    "bind_authoritative_predictions_with_role_access",
    "evaluate_pilot_with_role_access",
    "evaluate_prediction_panel_with_role_access",
    "evaluate_tune_cells_with_role_access",
    "freeze_authoritative_predictions",
    "freeze_pilot_plan",
    "load_pilot_plan",
    "load_authoritative_predictions_with_role_access",
    "generate_checkpoint_tune_result_with_role_access",
    "generate_confidence_predictions_with_role_access",
    "select_candidate_checkpoint",
    "select_clean_reference",
    "select_fresh_classifier_on_tune",
    "validate_checkpoint_selection_budget",
]
