"""Content-bound artifacts for the protocol's analytic confidence controls."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor

from mmdc_clip_f.provenance import sha256_file

from .baselines import (
    DSBaselineFeatures,
    DSLogisticErrorControl,
    MonotoneLogisticProbabilityAdapter,
    ScoreOrientation,
    TemperatureScaler,
    absolute_omission_sensitivity,
    scalar_baseline_scores,
)
from .cache import _require_external_or_ignored_destination
from .roles import Operation, PrivateExamRecord, Role, RoleManifest, run_with_role_access
from .training import ClassifierProvenance, ResearchRunConfig


CONTROL_ARTIFACT_VERSION = "view-risk-control-artifact/v2"
RAW_CONTROL_METHODS = frozenset(
    {"msp", "margin", "negative_entropy", "energy", "absolute_omission_sensitivity"}
)


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


def _finite_tensor(value: Tensor, *, name: str, ndim: int | None = None) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating tensor")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} has an invalid rank")
    if value.requires_grad or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite and detached")
    return value


@dataclass(frozen=True)
class ScalarControlRows:
    exam_keys: tuple[str, ...]
    scores: Tensor

    def __post_init__(self) -> None:
        object.__setattr__(self, "exam_keys", tuple(self.exam_keys))
        _finite_tensor(self.scores, name="classifier scores", ndim=2)
        if self.scores.shape != (len(self.exam_keys), 4):
            raise ValueError("classifier scores must have one four-class row per exam")


@dataclass(frozen=True)
class DSControlRows:
    exam_keys: tuple[str, ...]
    features: DSBaselineFeatures
    classifier_prediction: Tensor

    def __post_init__(self) -> None:
        object.__setattr__(self, "exam_keys", tuple(self.exam_keys))
        if not isinstance(self.features, DSBaselineFeatures):
            raise TypeError("DS rows require accepted DSBaselineFeatures")
        matrix = self.features.as_logistic_input()
        if matrix.shape[0] != len(self.exam_keys):
            raise ValueError("DS features must have one row per exam")
        if (
            not isinstance(self.classifier_prediction, Tensor)
            or self.classifier_prediction.shape != (len(self.exam_keys),)
            or self.classifier_prediction.dtype != torch.long
        ):
            raise ValueError("DS classifier predictions must be long with one value per exam")


@dataclass(frozen=True)
class ScalarCalibrationRows:
    exam_keys: tuple[str, ...]
    score: Tensor
    classifier_prediction: Tensor

    def __post_init__(self) -> None:
        object.__setattr__(self, "exam_keys", tuple(self.exam_keys))
        _finite_tensor(self.score, name="scalar control score", ndim=1)
        if self.score.shape != (len(self.exam_keys),):
            raise ValueError("scalar calibration score must have one value per exam")
        if (
            self.classifier_prediction.shape != (len(self.exam_keys),)
            or self.classifier_prediction.dtype != torch.long
        ):
            raise ValueError("scalar classifier predictions must be long with one value per exam")


@dataclass(frozen=True)
class ControlArtifact:
    method: str
    seed: int
    config_sha256: str
    classifier_checkpoint_sha256: str
    confidence_manifest_sha256: str | None
    tune_manifest_sha256: str
    evidence_kind: str
    parameter_count: int
    update_count: int
    selection_trials: int
    exposure_by_role: tuple[tuple[str, int], ...]
    scaling: str
    output_kind: str
    state: Mapping[str, object]
    evidence_files: tuple[tuple[str, str, str], ...]
    sha256: str
    path: Path

    def __post_init__(self) -> None:
        if self.method not in RAW_CONTROL_METHODS | {"temperature_scaled_msp", "ds_logistic"}:
            raise ValueError("control artifact method is unsupported")
        if self.seed not in (42, 43, 44) or isinstance(self.seed, bool):
            raise ValueError("control artifact seed is outside the frozen table")
        for value in (
            self.config_sha256,
            self.classifier_checkpoint_sha256,
            self.tune_manifest_sha256,
        ):
            if not _is_sha256(value):
                raise ValueError("control artifact binding must be SHA-256")
        if self.confidence_manifest_sha256 is not None and not _is_sha256(
            self.confidence_manifest_sha256
        ):
            raise ValueError("control confidence-fit binding must be SHA-256")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.parameter_count, self.update_count, self.selection_trials)
        ):
            raise ValueError("control counts must be nonnegative integers")
        if self.output_kind not in ("ranking", "probability"):
            raise ValueError("control output kind is invalid")
        if not _is_sha256(self.sha256):
            raise ValueError("control artifact identity must be SHA-256")
        object.__setattr__(self, "path", Path(self.path).resolve())
        object.__setattr__(self, "exposure_by_role", tuple(self.exposure_by_role))
        evidence = tuple(self.evidence_files)
        if len({purpose for purpose, _path, _digest in evidence}) != len(evidence) or any(
            not purpose
            or not Path(path).is_absolute()
            or not _is_sha256(digest)
            for purpose, path, digest in evidence
        ):
            raise ValueError("control fitting evidence file bindings are invalid")
        object.__setattr__(self, "evidence_files", evidence)

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": CONTROL_ARTIFACT_VERSION,
            "method": self.method,
            "seed": self.seed,
            "config_sha256": self.config_sha256,
            "classifier_checkpoint_sha256": self.classifier_checkpoint_sha256,
            "confidence_manifest_sha256": self.confidence_manifest_sha256,
            "tune_manifest_sha256": self.tune_manifest_sha256,
            "evidence_kind": self.evidence_kind,
            "parameter_count": self.parameter_count,
            "update_count": self.update_count,
            "selection_trials": self.selection_trials,
            "exposure_by_role": dict(self.exposure_by_role),
            "scaling": self.scaling,
            "output_kind": self.output_kind,
            "state": dict(self.state),
            "evidence_files": {
                purpose: {"path": path, "file_sha256": digest}
                for purpose, path, digest in self.evidence_files
            },
        }


def _validate_classifier(config: ResearchRunConfig, classifier: ClassifierProvenance) -> None:
    if not isinstance(config, ResearchRunConfig) or not isinstance(classifier, ClassifierProvenance):
        raise TypeError("controls require validated config and classifier provenance")
    if not classifier.workflow_complete or classifier.backbone != config.backbone:
        raise ValueError("control classifier/config workflow binding is incomplete or stale")


def _validate_rows(
    records: Sequence[PrivateExamRecord], exam_keys: Sequence[str]
) -> tuple[int, ...]:
    expected = tuple(record.exam_key for record in records)
    if tuple(exam_keys) != expected or len(set(exam_keys)) != len(exam_keys):
        raise ValueError("control rows must exactly follow the authorized manifest cohort")
    return tuple(record.density for record in records)


def _require_exact_manifest_role(manifest: RoleManifest, role: Role) -> None:
    if not isinstance(manifest, RoleManifest) or not manifest.records:
        raise PermissionError("control fitting requires a nonempty role manifest")
    if {record.role for record in manifest.records} != {role}:
        raise PermissionError("control manifest contains records outside its authorized role")


def _write_artifact(path: str | Path, payload: Mapping[str, object]) -> ControlArtifact:
    target = Path(path).resolve()
    if target.suffix != ".json" or not target.parent.exists():
        raise ValueError("control artifact must be JSON in an existing directory")
    staging = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    for candidate in (target, staging):
        _require_external_or_ignored_destination(candidate)
    digest = _sha256_json(payload)
    document = {"artifact_sha256": digest, "artifact": dict(payload)}
    try:
        with staging.open("xb") as handle:
            handle.write(_canonical_json(document) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(staging, target)
        target.chmod(0o600)
    finally:
        staging.unlink(missing_ok=True)
    return _artifact_from_payload(payload, digest=digest, path=target)


def _artifact_from_payload(
    payload: Mapping[str, object], *, digest: str, path: str | Path
) -> ControlArtifact:
    expected = {
        "schema_version",
        "method",
        "seed",
        "config_sha256",
        "classifier_checkpoint_sha256",
        "confidence_manifest_sha256",
        "tune_manifest_sha256",
        "evidence_kind",
        "parameter_count",
        "update_count",
        "selection_trials",
        "exposure_by_role",
        "scaling",
        "output_kind",
        "state",
        "evidence_files",
    }
    if set(payload) != expected or payload["schema_version"] != CONTROL_ARTIFACT_VERSION:
        raise ValueError("control artifact has a missing, unknown, or stale field")
    exposure = payload["exposure_by_role"]
    state = payload["state"]
    evidence = payload["evidence_files"]
    if (
        not isinstance(exposure, Mapping)
        or not isinstance(state, Mapping)
        or not isinstance(evidence, Mapping)
    ):
        raise ValueError("control artifact exposure/state schema is invalid")
    normalized_evidence = []
    for purpose, record in evidence.items():
        if (
            not isinstance(purpose, str)
            or not isinstance(record, Mapping)
            or set(record) != {"path", "file_sha256"}
        ):
            raise ValueError("control fitting evidence schema is invalid")
        source = Path(str(record["path"])).resolve()
        _require_external_or_ignored_destination(source)
        if sha256_file(source) != record["file_sha256"]:
            raise ValueError("control fitting evidence content changed")
        normalized_evidence.append((purpose, str(source), record["file_sha256"]))
    return ControlArtifact(
        method=payload["method"],  # type: ignore[arg-type]
        seed=payload["seed"],  # type: ignore[arg-type]
        config_sha256=payload["config_sha256"],  # type: ignore[arg-type]
        classifier_checkpoint_sha256=payload["classifier_checkpoint_sha256"],  # type: ignore[arg-type]
        confidence_manifest_sha256=payload["confidence_manifest_sha256"],  # type: ignore[arg-type]
        tune_manifest_sha256=payload["tune_manifest_sha256"],  # type: ignore[arg-type]
        evidence_kind=payload["evidence_kind"],  # type: ignore[arg-type]
        parameter_count=payload["parameter_count"],  # type: ignore[arg-type]
        update_count=payload["update_count"],  # type: ignore[arg-type]
        selection_trials=payload["selection_trials"],  # type: ignore[arg-type]
        exposure_by_role=tuple(sorted((str(k), int(v)) for k, v in exposure.items())),
        scaling=payload["scaling"],  # type: ignore[arg-type]
        output_kind=payload["output_kind"],  # type: ignore[arg-type]
        state=dict(state),
        evidence_files=tuple(sorted(normalized_evidence)),
        sha256=digest,
        path=Path(path),
    )


def load_control_artifact(path: str | Path) -> ControlArtifact:
    """Load and content-verify a private/git-ignored control artifact."""

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
        raise ValueError("control artifact is unavailable or invalid") from exc
    if document["artifact_sha256"] != digest:
        raise ValueError("control artifact content digest is stale")
    return _artifact_from_payload(document["artifact"], digest=digest, path=source)


def _base_payload(
    *,
    method: str,
    config: ResearchRunConfig,
    classifier: ClassifierProvenance,
    seed: int,
    confidence_manifest_sha256: str | None,
    tune_manifest_sha256: str,
    evidence_kind: str,
    parameter_count: int,
    update_count: int,
    selection_trials: int,
    exposure_by_role: Mapping[str, int],
    scaling: str,
    output_kind: str,
    state: Mapping[str, object],
    evidence_files: Mapping[str, str | Path] | None = None,
) -> dict[str, object]:
    _validate_classifier(config, classifier)
    bound_evidence = {}
    for purpose, raw_path in (evidence_files or {}).items():
        source = Path(raw_path).resolve()
        _require_external_or_ignored_destination(source)
        bound_evidence[purpose] = {
            "path": str(source),
            "file_sha256": sha256_file(source),
        }
    return {
        "schema_version": CONTROL_ARTIFACT_VERSION,
        "method": method,
        "seed": seed,
        "config_sha256": config.sha256,
        "classifier_checkpoint_sha256": classifier.checkpoint_sha256,
        "confidence_manifest_sha256": confidence_manifest_sha256,
        "tune_manifest_sha256": tune_manifest_sha256,
        "evidence_kind": evidence_kind,
        "parameter_count": parameter_count,
        "update_count": update_count,
        "selection_trials": selection_trials,
        "exposure_by_role": dict(exposure_by_role),
        "scaling": scaling,
        "output_kind": output_kind,
        "state": dict(state),
        "evidence_files": bound_evidence,
    }


def create_raw_control_artifact(
    path: str | Path,
    *,
    method: str,
    config: ResearchRunConfig,
    classifier: ClassifierProvenance,
    seed: int,
) -> ControlArtifact:
    """Record an unfitted raw-ranking control without calling identity a fitted scaler."""

    if method not in RAW_CONTROL_METHODS:
        raise ValueError("raw control method is unsupported")
    assert classifier.tune_manifest_sha256 is not None
    return _write_artifact(
        path,
        _base_payload(
            method=method,
            config=config,
            classifier=classifier,
            seed=seed,
            confidence_manifest_sha256=None,
            tune_manifest_sha256=classifier.tune_manifest_sha256,
            evidence_kind="analytic_raw_ranking",
            parameter_count=0,
            update_count=0,
            selection_trials=0,
            exposure_by_role={},
            scaling="none_declared_raw_ranking",
            output_kind="ranking",
            state={},
        ),
    )


def fit_temperature_control_artifact_with_role_access(
    path: str | Path,
    *,
    config: ResearchRunConfig,
    classifier: ClassifierProvenance,
    seed: int,
    tune_manifest: RoleManifest,
    row_reader: Callable[[tuple[PrivateExamRecord, ...]], ScalarControlRows],
    tune_manifest_path: str | Path | None = None,
    tune_input_path: str | Path | None = None,
) -> ControlArtifact:
    """Tune one positive temperature on authorized tune NLL and persist its state."""

    _require_exact_manifest_role(tune_manifest, Role.TUNE)
    run_with_role_access(
        tune_manifest,
        operation=Operation.TUNE_SELECTION,
        roles=(Role.TUNE,),
        loader=lambda _records: None,
    )
    _validate_classifier(config, classifier)
    if tune_manifest.manifest_sha256 != classifier.tune_manifest_sha256:
        raise ValueError("temperature tune manifest disagrees with classifier selection")
    evidence_files = {}
    if classifier.kind == "public_pretrained_fresh":
        if tune_manifest_path is None or tune_input_path is None:
            raise ValueError("production temperature fitting requires persisted tune evidence")
        evidence_files = {
            "tune_manifest": tune_manifest_path,
            "tune_rows": tune_input_path,
        }

    def authorized(records: tuple[PrivateExamRecord, ...]) -> ControlArtifact:
        rows = row_reader(records)
        if not isinstance(rows, ScalarControlRows):
            raise TypeError("temperature reader returned invalid rows")
        labels = torch.tensor(
            _validate_rows(records, rows.exam_keys), dtype=torch.long, device=rows.scores.device
        )
        scaler = TemperatureScaler().fit(
            rows.scores,
            labels,
            manifest=tune_manifest,
            exam_keys=rows.exam_keys,
        )
        return _write_artifact(
            path,
            _base_payload(
                method="temperature_scaled_msp",
                config=config,
                classifier=classifier,
                seed=seed,
                confidence_manifest_sha256=None,
                tune_manifest_sha256=tune_manifest.manifest_sha256,
                evidence_kind="temperature_tune_nll",
                parameter_count=1,
                update_count=1,
                selection_trials=1,
                exposure_by_role={Role.TUNE.value: len(records)},
                scaling="tune_fitted_temperature",
                output_kind="ranking",
                state={"temperature": float(scaler.temperature.cpu())},
                evidence_files=evidence_files,
            ),
        )

    return run_with_role_access(
        tune_manifest,
        operation=Operation.TUNE_SELECTION,
        roles=(Role.TUNE,),
        loader=authorized,
    )


def fit_ds_control_artifact_with_role_access(
    path: str | Path,
    *,
    config: ResearchRunConfig,
    classifier: ClassifierProvenance,
    seed: int,
    confidence_manifest: RoleManifest,
    tune_manifest: RoleManifest,
    confidence_reader: Callable[[tuple[PrivateExamRecord, ...]], DSControlRows],
    tune_reader: Callable[[tuple[PrivateExamRecord, ...]], DSControlRows],
    confidence_manifest_path: str | Path | None = None,
    tune_manifest_path: str | Path | None = None,
    confidence_input_path: str | Path | None = None,
    tune_input_path: str | Path | None = None,
) -> ControlArtifact:
    """Fit the DS scaler/weights on confidence_fit and select regularization on tune."""

    _validate_classifier(config, classifier)
    if tune_manifest.manifest_sha256 != classifier.tune_manifest_sha256:
        raise ValueError("DS tune manifest disagrees with classifier selection")
    evidence_files = {}
    if classifier.kind == "public_pretrained_fresh":
        raw_evidence = {
            "confidence_manifest": confidence_manifest_path,
            "confidence_rows": confidence_input_path,
            "tune_manifest": tune_manifest_path,
            "tune_rows": tune_input_path,
        }
        if any(path is None for path in raw_evidence.values()):
            raise ValueError("production DS fitting requires persisted role/input evidence")
        evidence_files = {
            purpose: path for purpose, path in raw_evidence.items() if path is not None
        }
    # Validate both role contracts before either private tensor reader is opened.
    _require_exact_manifest_role(confidence_manifest, Role.CONFIDENCE_FIT)
    _require_exact_manifest_role(tune_manifest, Role.TUNE)
    run_with_role_access(
        confidence_manifest,
        operation=Operation.CONFIDENCE_FITTING,
        roles=(Role.CONFIDENCE_FIT,),
        loader=lambda _records: None,
    )
    run_with_role_access(
        tune_manifest,
        operation=Operation.TUNE_SELECTION,
        roles=(Role.TUNE,),
        loader=lambda _records: None,
    )
    confidence_records = run_with_role_access(
        confidence_manifest,
        operation=Operation.CONFIDENCE_FITTING,
        roles=(Role.CONFIDENCE_FIT,),
        loader=lambda records: tuple(records),
    )
    tune_records = run_with_role_access(
        tune_manifest,
        operation=Operation.TUNE_SELECTION,
        roles=(Role.TUNE,),
        loader=lambda records: tuple(records),
    )
    confidence_rows = confidence_reader(confidence_records)
    tune_rows = tune_reader(tune_records)
    if not isinstance(confidence_rows, DSControlRows) or not isinstance(tune_rows, DSControlRows):
        raise TypeError("DS reader returned invalid rows")
    confidence_labels = _validate_rows(confidence_records, confidence_rows.exam_keys)
    tune_labels = _validate_rows(tune_records, tune_rows.exam_keys)
    confidence_targets = torch.tensor(
        confidence_labels, dtype=torch.long, device=confidence_rows.classifier_prediction.device
    )
    tune_targets = torch.tensor(
        tune_labels, dtype=torch.long, device=tune_rows.classifier_prediction.device
    )
    confidence_errors = (confidence_rows.classifier_prediction != confidence_targets).long()
    tune_errors = (tune_rows.classifier_prediction != tune_targets).long()
    control = DSLogisticErrorControl.fit_with_tune_selection(
        confidence_rows.features,
        confidence_errors,
        tune_rows.features,
        tune_errors,
        confidence_manifest=confidence_manifest,
        confidence_exam_keys=confidence_rows.exam_keys,
        tune_manifest=tune_manifest,
        tune_exam_keys=tune_rows.exam_keys,
        regularizations=config.search_table.ds_regularizations,
    )
    state = {
        "mean": control.mean.cpu().tolist(),
        "scale": control.scale.cpu().tolist(),
        "weight": control.weight.cpu().tolist(),
        "bias": float(control.bias.cpu()),
        "selected_regularization": control.selected_regularization,
    }
    return _write_artifact(
        path,
        _base_payload(
            method="ds_logistic",
            config=config,
            classifier=classifier,
            seed=seed,
            confidence_manifest_sha256=confidence_manifest.manifest_sha256,
            tune_manifest_sha256=tune_manifest.manifest_sha256,
            evidence_kind="ds_confidence_fit_tune_regularization",
            parameter_count=int(control.weight.numel()) + 1,
            update_count=control.selection_trials,
            selection_trials=control.selection_trials,
            exposure_by_role={
                Role.CONFIDENCE_FIT.value: len(confidence_records),
                Role.TUNE.value: len(tune_records),
            },
            scaling="confidence_fit_fitted_feature_scaler",
            output_kind="probability",
            state=state,
            evidence_files=evidence_files,
        ),
    )


def fit_scalar_calibration_artifact_with_role_access(
    path: str | Path,
    *,
    method: str,
    orientation: ScoreOrientation,
    config: ResearchRunConfig,
    classifier: ClassifierProvenance,
    seed: int,
    tune_manifest: RoleManifest,
    row_reader: Callable[[tuple[PrivateExamRecord, ...]], ScalarCalibrationRows],
    tune_manifest_path: str | Path | None = None,
    tune_input_path: str | Path | None = None,
) -> ControlArtifact:
    """Create the separately labelled tune-monotone probability output for a scalar."""

    if method not in RAW_CONTROL_METHODS:
        raise ValueError("scalar calibration method is unsupported")
    _require_exact_manifest_role(tune_manifest, Role.TUNE)
    _validate_classifier(config, classifier)
    if tune_manifest.manifest_sha256 != classifier.tune_manifest_sha256:
        raise ValueError("scalar calibration tune manifest binding is stale")
    evidence_files = {}
    if classifier.kind == "public_pretrained_fresh":
        if tune_manifest_path is None or tune_input_path is None:
            raise ValueError("production scalar calibration requires persisted tune evidence")
        evidence_files = {
            "tune_manifest": tune_manifest_path,
            "tune_rows": tune_input_path,
        }

    def authorized(records: tuple[PrivateExamRecord, ...]) -> ControlArtifact:
        rows = row_reader(records)
        if not isinstance(rows, ScalarCalibrationRows):
            raise TypeError("scalar calibration reader returned invalid rows")
        labels = torch.tensor(
            _validate_rows(records, rows.exam_keys),
            dtype=torch.long,
            device=rows.classifier_prediction.device,
        )
        errors = (rows.classifier_prediction != labels).long()
        adapter = MonotoneLogisticProbabilityAdapter(orientation).fit(
            rows.score,
            errors,
            manifest=tune_manifest,
            exam_keys=rows.exam_keys,
        )
        assert adapter._slope is not None and adapter._intercept is not None
        return _write_artifact(
            path,
            _base_payload(
                method=method,
                config=config,
                classifier=classifier,
                seed=seed,
                confidence_manifest_sha256=None,
                tune_manifest_sha256=tune_manifest.manifest_sha256,
                evidence_kind="scalar_tune_monotone_calibration",
                parameter_count=2,
                update_count=1,
                selection_trials=1,
                exposure_by_role={Role.TUNE.value: len(records)},
                scaling="tune_fitted_monotone_probability",
                output_kind="probability",
                state={
                    "orientation": orientation.value,
                    "slope": float(adapter._slope.cpu()),
                    "intercept": float(adapter._intercept.cpu()),
                    "output_label": "tune_calibrated_error_probability",
                },
                evidence_files=evidence_files,
            ),
        )

    return run_with_role_access(
        tune_manifest,
        operation=Operation.TUNE_SELECTION,
        roles=(Role.TUNE,),
        loader=authorized,
    )


def analytic_control_confidence(
    artifact: ControlArtifact,
    *,
    scores: Tensor | None = None,
    ds_features: DSBaselineFeatures | None = None,
    omission_score_differences: Tensor | None = None,
    removal_valid_mask: Tensor | None = None,
) -> tuple[Tensor, str]:
    """Generate confidence without labels or patient/corruption metadata."""

    current = load_control_artifact(artifact.path)
    if current.sha256 != artifact.sha256:
        raise ValueError("control artifact changed before inference")
    if artifact.method == "ds_logistic":
        if ds_features is None:
            raise ValueError("DS confidence requires DS features")
        state = artifact.state
        control = DSLogisticErrorControl(
            mean=torch.tensor(state["mean"], dtype=ds_features.values.dtype),
            scale=torch.tensor(state["scale"], dtype=ds_features.values.dtype),
            weight=torch.tensor(state["weight"], dtype=ds_features.values.dtype),
            bias=torch.tensor(state["bias"], dtype=ds_features.values.dtype),
            selected_regularization=float(state["selected_regularization"]),
            selection_trials=artifact.selection_trials,
        )
        return 1 - control.predict_error_probability(ds_features), artifact.output_kind
    if artifact.method == "absolute_omission_sensitivity":
        if omission_score_differences is None or removal_valid_mask is None:
            raise ValueError("omission confidence requires differences and validity mask")
        scalar = absolute_omission_sensitivity(
            omission_score_differences, removal_valid_mask
        ).values
        if artifact.evidence_kind != "scalar_tune_monotone_calibration":
            return -scalar, "ranking"
        raw = scalar
    else:
        if scores is None:
            raise ValueError("scalar confidence requires frozen classifier scores")
        if artifact.method == "temperature_scaled_msp":
            temperature = float(artifact.state["temperature"])
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("temperature artifact state is invalid")
            return (scores / temperature).softmax(1).max(1).values, "ranking"
        result = scalar_baseline_scores(scores)[artifact.method]
        raw = result.values
        if artifact.evidence_kind != "scalar_tune_monotone_calibration":
            if result.orientation is ScoreOrientation.HIGHER_ERROR:
                raw = -raw
            return raw, "ranking"
    orientation = ScoreOrientation(artifact.state["orientation"])
    error_oriented = raw if orientation is ScoreOrientation.HIGHER_ERROR else -raw
    slope = float(artifact.state["slope"])
    intercept = float(artifact.state["intercept"])
    error_probability = torch.sigmoid(slope * error_oriented + intercept)
    return 1 - error_probability, "probability"


__all__ = [
    "CONTROL_ARTIFACT_VERSION",
    "ControlArtifact",
    "DSControlRows",
    "RAW_CONTROL_METHODS",
    "ScalarCalibrationRows",
    "ScalarControlRows",
    "analytic_control_confidence",
    "create_raw_control_artifact",
    "fit_ds_control_artifact_with_role_access",
    "fit_scalar_calibration_artifact_with_role_access",
    "fit_temperature_control_artifact_with_role_access",
    "load_control_artifact",
]
