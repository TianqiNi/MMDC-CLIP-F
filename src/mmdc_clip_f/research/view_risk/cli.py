"""Working CLI dispatch for guarded view-risk research workflows."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import torch
from mmdc_clip_f.provenance import sha256_file

from .artifacts import (
    DSControlRows,
    RAW_CONTROL_METHODS,
    ScalarCalibrationRows,
    ScalarControlRows,
    create_raw_control_artifact,
    fit_ds_control_artifact_with_role_access,
    fit_scalar_calibration_artifact_with_role_access,
    fit_temperature_control_artifact_with_role_access,
)
from .baselines import (
    DSBaselineFeatures,
    absolute_omission_sensitivity,
    build_ds_features,
    scalar_baseline_scores,
)
from .head_inputs import prepare_raw_head_inputs
from .fusion import DDSM_FUSION_PAIRS, RSNA_FUSION_PAIRS

from .cache import (
    bind_cache_files,
    CacheProvenance,
    _require_external_or_ignored_destination,
    load_cache_bundle,
    load_cache_provenance,
)
from .evaluation import (
    ConfidenceScore,
    CheckpointTuneResult,
    ModelArtifactSelection,
    TuneCellResult,
    TunePrediction,
    evaluate_prediction_panel_with_role_access,
    validate_evaluation_cache_bundle,
    validate_evaluation_cache_provenance,
    generate_confidence_predictions_with_role_access,
    generate_checkpoint_tune_result_with_role_access,
    freeze_authoritative_predictions,
    freeze_pilot_plan,
    load_pilot_plan,
    load_authoritative_predictions_with_role_access,
    load_model_artifact_selection,
    load_clean_reference_checkpoint,
    select_candidate_checkpoint,
    select_clean_reference,
    validate_checkpoint_selection_budget,
)
from .inputs import CANONICAL_VIEWS
from .metrics import EvaluationPrediction
from .perturbations import FITTING_FAMILIES, FITTING_SEVERITIES
from .roles import (
    ManifestBinding,
    Operation,
    Role,
    run_with_role_access,
)
from .training import (
    LEARNED_TORCH_METHODS,
    ClassifierProvenance,
    ConfidenceFitBatch,
    TrainingBinding,
    build_learned_method,
    build_method_training_schedule,
    build_training_schedule,
    confidence_bundle_scores,
    fit_confidence_method_with_role_access,
    load_research_run_config,
    load_role_manifest_for_operation,
    load_pinned_public_clip_classifier,
    tune_selection_cells,
    validate_scheduled_cache_provenance,
    validate_tune_cache_provenance,
)
from .production import (
    fit_classifier_images_with_role_access,
    load_confidence_fit_artifact,
    save_confidence_fit_artifact,
    save_tensor_checkpoint,
    load_public_initialization_artifact,
    load_selected_classifier_artifact,
    reload_verified_public_classifier,
    save_public_initialization_artifact,
    select_classifier_from_images_with_role_access,
)
from .inference import (
    load_learned_confidence_checkpoint,
    load_learned_confidence_model,
    score_confidence_cache_bundle,
)


COMMANDS = frozenset(
    {
        "view-risk-validate-config",
        "view-risk-select",
        "view-risk-freeze-pilot",
        "view-risk-evaluate",
        "view-risk-freeze-classifier-predictions",
        "view-risk-train-confidence",
        "view-risk-fit-classifier",
        "view-risk-select-classifier",
        "view-risk-fit-control",
        "view-risk-score-confidence",
        "view-risk-select-confidence",
    }
)


def add_view_risk_subparsers(subparsers: argparse._SubParsersAction) -> None:
    validate = subparsers.add_parser(
        "view-risk-validate-config",
        help="Validate and hash a strict P4B JSON/YAML run configuration",
    )
    validate.add_argument("--config", required=True)

    select = subparsers.add_parser(
        "view-risk-select", help="Select a checkpoint from aggregate tune-only AURCs"
    )
    select.add_argument("--config", required=True)
    select.add_argument("--reference-results", required=True)
    select.add_argument("--candidate-results", required=True)

    freeze = subparsers.add_parser(
        "view-risk-freeze-pilot", help="Exclusively freeze a complete immutable pilot plan"
    )
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--bindings", required=True)
    freeze.add_argument("--selections", required=True)
    freeze.add_argument("--output", required=True)

    evaluate = subparsers.add_parser(
        "view-risk-evaluate", help="Role-authorize and aggregate one frozen pilot panel via P4A"
    )
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--plan", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--manifest-binding", required=True)
    evaluate.add_argument("--private-root", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--method", required=True)
    evaluate.add_argument("--seed", required=True, type=int)
    evaluate.add_argument("--model-sha256", required=True)
    evaluate.add_argument("--model-artifact")

    predictions = subparsers.add_parser(
        "view-risk-freeze-classifier-predictions",
        help="Bind one complete label-free classifier prediction artifact after plan freeze",
    )
    predictions.add_argument("--config", required=True)
    predictions.add_argument("--plan", required=True)
    predictions.add_argument("--manifest", required=True)
    predictions.add_argument("--manifest-binding", required=True)
    predictions.add_argument("--private-root", required=True)
    predictions.add_argument("--cache-index", required=True)
    predictions.add_argument("--output", required=True)

    train = subparsers.add_parser(
        "view-risk-train-confidence",
        help="Fit one frozen-table learned method from role-bound streamed caches",
    )
    train.add_argument("--config", required=True)
    train.add_argument("--manifest", required=True)
    train.add_argument("--manifest-binding", required=True)
    train.add_argument("--private-root", required=True)
    train.add_argument("--cache-index", required=True)
    train.add_argument("--method", required=True)
    train.add_argument("--seed", required=True, type=int)
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--stop-after-epoch", type=int)
    train.add_argument("--device", default="cpu")
    train.add_argument("--classifier-artifact")
    train.add_argument("--epoch-checkpoint-directory", required=True)
    train.add_argument("--training-artifact", required=True)

    classifier_fit = subparsers.add_parser(
        "view-risk-fit-classifier",
        help="Fit a fresh pinned public classifier on classifier_fit only",
    )
    classifier_fit.add_argument("--config", required=True)
    classifier_fit.add_argument("--manifest", required=True)
    classifier_fit.add_argument("--manifest-binding", required=True)
    classifier_fit.add_argument("--private-root", required=True)
    classifier_fit.add_argument("--image-root", required=True)
    classifier_fit.add_argument("--readiness", required=True)
    classifier_fit.add_argument("--initialization", required=True)
    classifier_fit.add_argument("--checkpoint-directory", required=True)
    classifier_fit.add_argument("--resume-checkpoint", required=True)
    classifier_fit.add_argument("--output", required=True)
    classifier_fit.add_argument("--resume", action="store_true")
    classifier_fit.add_argument("--stop-after-epoch", type=int)
    classifier_fit.add_argument("--device", default="cpu")

    classifier_select = subparsers.add_parser(
        "view-risk-select-classifier",
        help="Compute tune NLL for every classifier epoch and persist the selected model",
    )
    classifier_select.add_argument("--config", required=True)
    classifier_select.add_argument("--fit-artifact", required=True)
    classifier_select.add_argument("--manifest", required=True)
    classifier_select.add_argument("--manifest-binding", required=True)
    classifier_select.add_argument("--private-root", required=True)
    classifier_select.add_argument("--image-root", required=True)
    classifier_select.add_argument("--output", required=True)
    classifier_select.add_argument("--device", default="cpu")

    control = subparsers.add_parser(
        "view-risk-fit-control",
        help="Persist one of the seven accepted analytic confidence controls",
    )
    control.add_argument("--config", required=True)
    control.add_argument("--classifier-artifact", required=True)
    control.add_argument("--method", required=True)
    control.add_argument("--calibrate-scalar", action="store_true")
    control.add_argument("--seed", required=True, type=int)
    control.add_argument("--output", required=True)
    control.add_argument("--tune-manifest")
    control.add_argument("--tune-manifest-binding")
    control.add_argument("--tune-private-root")
    control.add_argument("--tune-cache-index")
    control.add_argument("--confidence-manifest")
    control.add_argument("--confidence-manifest-binding")
    control.add_argument("--confidence-private-root")
    control.add_argument("--confidence-cache-index")

    score = subparsers.add_parser(
        "view-risk-score-confidence",
        help="Reload a frozen learned/control artifact and emit P4A prediction rows",
    )
    score.add_argument("--config", required=True)
    score.add_argument("--plan", required=True)
    score.add_argument("--manifest", required=True)
    score.add_argument("--manifest-binding", required=True)
    score.add_argument("--private-root", required=True)
    score.add_argument("--cache-index", required=True)
    score.add_argument("--panel", required=True)
    score.add_argument("--method", required=True)
    score.add_argument("--seed", required=True, type=int)
    score.add_argument("--output", required=True)
    score.add_argument("--device", default="cpu")

    confidence_select = subparsers.add_parser(
        "view-risk-select-confidence",
        help="Infer all tune cells for every learned epoch and persist guarded selection",
    )
    confidence_select.add_argument("--config", required=True)
    confidence_select.add_argument("--training-artifact", required=True)
    confidence_select.add_argument("--manifest", required=True)
    confidence_select.add_argument("--manifest-binding", required=True)
    confidence_select.add_argument("--private-root", required=True)
    confidence_select.add_argument("--tune-cache-index", required=True)
    confidence_select.add_argument("--clean-reference-record")
    confidence_select.add_argument("--output", required=True)
    confidence_select.add_argument("--device", default="cpu")


def _read_json(path: str | Path, *, description: str) -> object:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} could not be read as JSON") from exc


def _checkpoint_results(path: str | Path) -> tuple[CheckpointTuneResult, ...]:
    document = _read_json(path, description="tune result file")
    if not isinstance(document, dict) or set(document) != {"checkpoints"}:
        raise ValueError("tune result file has missing or unknown fields")
    checkpoints = document["checkpoints"]
    if not isinstance(checkpoints, list):
        raise ValueError("tune checkpoints must be a list")
    result = []
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict) or set(checkpoint) != {
            "method",
            "seed",
            "classifier_checkpoint_sha256",
            "tune_manifest_sha256",
            "epoch",
            "clean_aurc",
            "stress_cells",
            "artifact_sha256",
            "role",
        }:
            raise ValueError("tune checkpoint has missing or unknown fields")
        cells = checkpoint["stress_cells"]
        if not isinstance(cells, list):
            raise ValueError("tune stress cells must be a list")
        try:
            result.append(
                CheckpointTuneResult(
                    method=checkpoint["method"],
                    seed=checkpoint["seed"],
                    classifier_checkpoint_sha256=checkpoint[
                        "classifier_checkpoint_sha256"
                    ],
                    tune_manifest_sha256=checkpoint["tune_manifest_sha256"],
                    epoch=checkpoint["epoch"],
                    clean_aurc=checkpoint["clean_aurc"],
                    stress_cells=tuple(TuneCellResult(**cell) for cell in cells),
                    artifact_sha256=checkpoint["artifact_sha256"],
                    role=checkpoint["role"],
                )
            )
        except TypeError as exc:
            raise ValueError("tune result entry has an invalid schema") from exc
    return tuple(result)


def _classifier_from_dict(value: object) -> ClassifierProvenance:
    if not isinstance(value, dict) or set(value) != {
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
    }:
        raise ValueError("classifier binding has missing or unknown fields")
    try:
        return ClassifierProvenance(
            kind=value["kind"],
            backbone=value["backbone"],
            hf_model=value["hf_model"],
            revision=value["revision"],
            image_size=value["image_size"],
            hidden_size=value["hidden_size"],
            prompts=tuple(value["prompts"]),
            preprocessing=value["preprocessing"],
            image_mean=tuple(value["image_mean"]),
            image_std=tuple(value["image_std"]),
            checkpoint_sha256=value["checkpoint_sha256"],
            diagnostic_reason=value["diagnostic_reason"],
            initialization_sha256=value["initialization_sha256"],
            classifier_fit_manifest_sha256=value["classifier_fit_manifest_sha256"],
            classifier_fit_update_count=value["classifier_fit_update_count"],
            tune_selection_sha256=value["tune_selection_sha256"],
            tune_manifest_sha256=value["tune_manifest_sha256"],
            patient_readiness_verified=value["patient_readiness_verified"],
            readiness_audit_sha256=value["readiness_audit_sha256"],
            public_weight_content_sha256=value["public_weight_content_sha256"],
        )
    except TypeError as exc:
        raise ValueError("classifier binding has an invalid schema") from exc


def _manifest_binding(path: str | Path) -> ManifestBinding:
    value = _read_json(path, description="manifest binding")
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "dataset_namespace",
        "source_hashes",
        "manifest_sha256",
    }:
        raise ValueError("manifest binding has missing or unknown fields")
    try:
        return ManifestBinding(**value)
    except TypeError as exc:
        raise ValueError("manifest binding has an invalid schema") from exc


def _prediction_rows(path: str | Path) -> tuple[EvaluationPrediction, ...]:
    _require_external_or_ignored_destination(Path(path).resolve())
    value = _read_json(path, description="private prediction artifact")
    if not isinstance(value, dict) or set(value) != {"predictions"}:
        raise ValueError("private prediction artifact has missing or unknown fields")
    rows = value["predictions"]
    if not isinstance(rows, list):
        raise ValueError("private predictions must be a list")
    try:
        return tuple(
            EvaluationPrediction(
                **{
                    **row,
                    "mask": tuple(row.get("mask", ("L_CC", "L_MLO", "R_CC", "R_MLO"))),
                }
            )
            for row in rows
        )
    except (AttributeError, TypeError) as exc:
        raise ValueError("private prediction row has an invalid schema") from exc



def _cache_index(
    path: str | Path,
    *,
    expected_classifier: ClassifierProvenance | None = None,
) -> tuple[ClassifierProvenance, tuple[dict[str, object], ...]]:
    _require_external_or_ignored_destination(Path(path).resolve())
    value = _read_json(path, description="private cache index")
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "classifier",
        "entries",
    }:
        raise ValueError("private cache index has missing or unknown fields")
    if value["schema_version"] != "view-risk-training-cache-index/v1":
        raise ValueError("private cache index version is unsupported")
    entries = value["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("private cache index entries must be a nonempty list")
    normalized = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "epoch",
            "metadata_path",
            "provenance",
        }:
            raise ValueError("private cache index entry has missing or unknown fields")
        epoch = entry["epoch"]
        path_value = entry["metadata_path"]
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
            raise ValueError("private cache index epoch must be positive")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError("private cache index path must be nonempty")
        if not Path(path_value).is_absolute():
            raise ValueError("private cache index paths must be absolute")
        normalized.append(
            {
                "epoch": epoch,
                "metadata_path": path_value,
                "provenance": CacheProvenance.from_dict(entry["provenance"]),
            }
        )
    if expected_classifier is None:
        classifier = _classifier_from_dict(value["classifier"])
        if classifier.kind != "synthetic_injected":
            raise PermissionError(
                "production cache indexes require selected-classifier artifact evidence"
            )
    else:
        if value["classifier"] != expected_classifier.to_dict():
            raise ValueError("cache-index classifier disagrees with verified classifier evidence")
        classifier = expected_classifier
    return classifier, tuple(normalized)




def _control_cache_index(
    path: str | Path,
    *,
    classifier: ClassifierProvenance,
    semantics: str,
) -> tuple[dict[str, object], ...]:
    """Parse an explicit method-specific selected-classifier cache schedule."""

    allowed = {"clean_full_tune", "ds_confidence_fit", "ds_tune_regularization"}
    if semantics not in allowed:
        raise ValueError("control cache semantics are unsupported")
    source = Path(path).resolve()
    _require_external_or_ignored_destination(source)
    value = _read_json(source, description="private control cache index")
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "classifier", "semantics", "entries"}
        or value["schema_version"] != "view-risk-control-cache-index/v2"
        or value["classifier"] != classifier.to_dict()
        or value["semantics"] != semantics
        or not isinstance(value["entries"], list)
        or not value["entries"]
    ):
        raise ValueError("control cache schedule/classifier binding is invalid")
    result = []
    paths = set()
    for entry in value["entries"]:
        common = {"metadata_path", "provenance"}
        expected = (
            common
            if semantics == "clean_full_tune"
            else common | {"epoch"}
            if semantics == "ds_confidence_fit"
            else common | {"family", "severity", "target_view"}
        )
        if not isinstance(entry, dict) or set(entry) != expected:
            raise ValueError("control cache schedule entry has an invalid schema")
        path_value = entry["metadata_path"]
        if not isinstance(path_value, str) or not Path(path_value).is_absolute():
            raise ValueError("control cache paths must be absolute")
        metadata = Path(path_value).resolve()
        if metadata in paths:
            raise ValueError("control cache schedule contains duplicate cache paths")
        paths.add(metadata)
        record = {
            "metadata_path": metadata,
            "provenance": CacheProvenance.from_dict(entry["provenance"]),
        }
        if semantics == "ds_confidence_fit":
            epoch = entry["epoch"]
            if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
                raise ValueError("DS confidence-fit cache epoch is invalid")
            record["epoch"] = epoch
        elif semantics == "ds_tune_regularization":
            cell = (entry["family"], entry["severity"], entry["target_view"])
            if cell not in tune_selection_cells():
                raise ValueError("DS tune cache cell is outside the frozen panel")
            record["cell"] = cell
        result.append(record)
    return tuple(result)


def _control_cache_evidence(
    path: str | Path,
    entries: tuple[dict[str, object], ...],
    *,
    prefix: str,
) -> dict[str, Path]:
    evidence = {f"{prefix}_cache_index": Path(path).resolve()}
    for index, entry in enumerate(entries):
        binding = bind_cache_files(entry["metadata_path"])
        evidence[f"{prefix}_cache_metadata:{index}"] = Path(binding["metadata_path"])
        evidence[f"{prefix}_cache_tensors:{index}"] = Path(binding["tensors_path"])
    return evidence


def _preflight_control_cache_entries(
    entries,
    records,
    *,
    semantics,
    manifest,
    config,
    seed,
    encoder_identity,
) -> tuple[str, ...]:
    if encoder_identity is None:
        raise ValueError("production controls require persisted encoder identity")
    fusion_pairs = (
        RSNA_FUSION_PAIRS if config.dataset == "RSNA" else DDSM_FUSION_PAIRS
    )
    expected_keys: list[str] = []
    actual_keys: list[str] = []
    for entry in entries:
        operation = (
            Operation.CONFIDENCE_FITTING
            if semantics == "ds_confidence_fit"
            else Operation.TUNE_SELECTION
        )
        role = (
            Role.CONFIDENCE_FIT
            if semantics == "ds_confidence_fit"
            else Role.TUNE
        )
        provenance = load_cache_provenance(
            entry["metadata_path"],
            expected_provenance=entry["provenance"],
            manifest=manifest,
            operation=operation,
            role=role,
        )
        if provenance.batch_size != 1:
            raise ValueError("control schedule caches must contain one exam")
        actual_keys.append(provenance.exam_keys[0])
    if semantics == "clean_full_tune":
        expected_keys = [record.exam_key for record in records]
        if actual_keys != expected_keys:
            raise ValueError("clean control caches must follow the exact tune cohort")
        by_exam = {record.exam_key: record for record in records}
        for entry in entries:
            provenance = entry["provenance"]
            validate_tune_cache_provenance(
                provenance,
                record=by_exam[provenance.exam_keys[0]],
                cell=(None, None, None),
                stress_seed=config.stress_seed,
                encoder_identity=encoder_identity,
                fusion_pairs=fusion_pairs,
            )
    elif semantics == "ds_confidence_fit":
        schedules = {
            epoch: {item.exam_key: item for item in build_training_schedule(
                records, epoch=epoch, seed=seed
            )}
            for epoch in range(1, config.epochs + 1)
        }
        expected_pairs = [
            (epoch, record.exam_key)
            for epoch in range(1, config.epochs + 1)
            for record in records
        ]
        actual_pairs = [
            (entry["epoch"], entry["provenance"].exam_keys[0]) for entry in entries
        ]
        if actual_pairs != expected_pairs:
            raise ValueError("DS confidence caches do not follow the finite draw schedule")
        expected_keys = [exam_key for _epoch, exam_key in expected_pairs]
        for entry in entries:
            validate_scheduled_cache_provenance(
                entry["provenance"],
                manifest=manifest,
                classifier_checkpoint_sha256=encoder_identity.checkpoint_sha256,
                schedule=schedules[entry["epoch"]],
                encoder_identity=encoder_identity,
                fusion_pairs=fusion_pairs,
            )
    else:
        expected_pairs = [
            (cell, record.exam_key)
            for cell in tune_selection_cells()
            for record in records
        ]
        actual_pairs = [
            (entry["cell"], entry["provenance"].exam_keys[0]) for entry in entries
        ]
        if actual_pairs != expected_pairs:
            raise ValueError("DS tune caches do not follow the exact eligible tune schedule")
        expected_keys = [exam_key for _cell, exam_key in expected_pairs]
        by_exam = {record.exam_key: record for record in records}
        for entry in entries:
            provenance = entry["provenance"]
            validate_tune_cache_provenance(
                provenance,
                record=by_exam[provenance.exam_keys[0]],
                cell=entry["cell"],
                stress_seed=config.stress_seed,
                encoder_identity=encoder_identity,
                fusion_pairs=fusion_pairs,
            )
    return tuple(expected_keys)


def _load_control_cache_bundles(
    entries,
    *,
    manifest,
    operation: Operation,
    role: Role,
):
    return tuple(
        load_cache_bundle(
            entry["metadata_path"],
            expected_provenance=entry["provenance"],
            manifest=manifest,
            operation=operation,
            role=role,
        )
        for entry in entries
    )


def _scalar_rows_from_control_caches(
    entries, expected_exam_keys, *, manifest, operation: Operation, role: Role
) -> ScalarControlRows:
    bundles = _load_control_cache_bundles(
        entries, manifest=manifest, operation=operation, role=role
    )
    return ScalarControlRows(
        tuple(expected_exam_keys),
        torch.cat(tuple(bundle.features.scores for bundle in bundles), dim=0),
    )


def _ds_rows_from_control_caches(
    entries, expected_exam_keys, *, manifest, operation: Operation, role: Role
) -> DSControlRows:
    bundles = _load_control_cache_bundles(
        entries, manifest=manifest, operation=operation, role=role
    )
    features = tuple(
        build_ds_features(
            bundle.features.logits_by_view,
            bundle.features.observed_views,
            fusion_pairs=bundle.features.fusion_pairs,
        )
        for bundle in bundles
    )
    pairs = features[0].fusion_pairs
    if any(current.fusion_pairs != pairs for current in features):
        raise ValueError("control caches mix classifier fusion trees")
    combined = DSBaselineFeatures(
        torch.cat(tuple(item.values for item in features), dim=0),
        torch.cat(tuple(item.valid_mask for item in features), dim=0),
        torch.cat(tuple(item.observed_mask for item in features), dim=0),
        torch.cat(tuple(item.conflict_valid_mask for item in features), dim=0),
        pairs,
    )
    return DSControlRows(
        tuple(expected_exam_keys),
        combined,
        torch.cat(tuple(bundle.features.prediction for bundle in bundles), dim=0).long(),
    )


def _calibration_rows_from_control_caches(
    method: str,
    entries: tuple[dict[str, object], ...],
    expected_exam_keys,
    *,
    manifest,
    classifier: ClassifierProvenance,
) -> tuple[ScalarCalibrationRows, object]:
    bundles = _load_control_cache_bundles(
        entries,
        manifest=manifest,
        operation=Operation.TUNE_SELECTION,
        role=Role.TUNE,
    )
    values = []
    orientation = None
    for bundle in bundles:
        if method == "absolute_omission_sensitivity":
            raw = prepare_raw_head_inputs(
                bundle.features, backbone=bundle.provenance.backbone
            )
            result = absolute_omission_sensitivity(
                raw.omission_score_differences, raw.removal_valid_mask
            )
        else:
            result = scalar_baseline_scores(bundle.features.scores)[method]
        if orientation is not None and result.orientation != orientation:
            raise RuntimeError("scalar control orientation changed across cache batches")
        orientation = result.orientation
        values.append(result.values)
    assert orientation is not None
    return (
        ScalarCalibrationRows(
            tuple(expected_exam_keys),
            torch.cat(tuple(values), dim=0),
            torch.cat(tuple(bundle.features.prediction for bundle in bundles), dim=0).long(),
        ),
        orientation,
    )

def _evaluation_cache_index(path: str | Path) -> tuple[dict[str, object], ...]:
    _require_external_or_ignored_destination(Path(path).resolve())
    value = _read_json(path, description="private evaluation cache index")
    if not isinstance(value, dict) or set(value) != {"schema_version", "entries"} or value[
        "schema_version"
    ] != "view-risk-evaluation-cache-index/v1":
        raise ValueError("evaluation cache index has missing or unknown fields")
    entries = value["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("evaluation cache index entries must be nonempty")
    result = []
    cell_fields = {
        "panel", "family", "severity", "target_view", "variant", "mask",
        "realization_id",
    }
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "cell", "metadata_path", "provenance"
        } or not isinstance(entry["cell"], dict) or set(entry["cell"]) != cell_fields:
            raise ValueError("evaluation cache entry has an invalid schema")
        metadata_path = Path(entry["metadata_path"])
        if not metadata_path.is_absolute():
            raise ValueError("evaluation cache paths must be absolute")
        cell = dict(entry["cell"])
        cell["mask"] = tuple(cell["mask"])
        result.append(
            {
                "cell": cell,
                "metadata_path": metadata_path.resolve(),
                "provenance": CacheProvenance.from_dict(entry["provenance"]),
            }
        )
    return tuple(result)



def _evaluation_cell_key(cell: Mapping[str, object]) -> tuple[object, ...]:
    return (
        cell["panel"],
        cell["family"],
        cell["severity"],
        cell["target_view"],
        cell["variant"],
        cell["mask"],
        cell["realization_id"],
    )


def _prevalidate_evaluation_cache_entries(
    plan,
    entries: tuple[dict[str, object], ...],
    records,
    *,
    manifest,
    expected_by_cell: Mapping[tuple[object, ...], set[str]] | None = None,
) -> None:
    """Validate all private metadata and exact cohorts before tensor access."""

    record_by_exam = {record.exam_key: record for record in records}
    expected_exams = set(record_by_exam)
    seen: dict[tuple[object, ...], set[str]] = {}
    for entry in entries:
        cell = entry["cell"]
        cell_key = _evaluation_cell_key(cell)
        if expected_by_cell is not None and cell_key not in expected_by_cell:
            raise ValueError("evaluation cache index contains an unrequested panel/cell")
        provenance = load_cache_provenance(
            entry["metadata_path"],
            expected_provenance=entry["provenance"],
            manifest=manifest,
            operation=Operation.PILOT_EVALUATION,
            role=Role.PILOT,
        )
        if (
            provenance.batch_size != 1
            or provenance.exam_keys[0] not in record_by_exam
        ):
            raise ValueError("evaluation cache must contain one authorized exam")
        exam_key = provenance.exam_keys[0]
        validate_evaluation_cache_provenance(
            plan, provenance, cell, record_by_exam[exam_key]
        )
        group = seen.setdefault(cell_key, set())
        if exam_key in group:
            raise ValueError("evaluation cache index duplicates an authorized exam")
        group.add(exam_key)
    expected_cells = (
        {cell: set(exams) for cell, exams in expected_by_cell.items()}
        if expected_by_cell is not None
        else {cell: expected_exams for cell in seen}
    )
    if set(seen) != set(expected_cells) or any(
        seen[cell] != expected_cells[cell] for cell in expected_cells
    ):
        raise ValueError("every evaluation cell must contain the exact authorized exam cohort")


def _tune_cache_index(path: str | Path) -> tuple[dict[str, object], ...]:
    _require_external_or_ignored_destination(Path(path).resolve())
    value = _read_json(path, description="private tune cache index")
    if not isinstance(value, dict) or set(value) != {"schema_version", "entries"} or value[
        "schema_version"
    ] != "view-risk-tune-cache-index/v1":
        raise ValueError("tune cache index has missing or unknown fields")
    entries = value["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("tune cache index entries must be nonempty")
    result = []
    represented = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "family", "severity", "target_view", "metadata_path", "provenance"
        }:
            raise ValueError("tune cache entry has an invalid schema")
        cell = (entry["family"], entry["severity"], entry["target_view"])
        if cell != (None, None, None) and (
            cell[0] not in FITTING_FAMILIES
            or cell[1] not in FITTING_SEVERITIES
            or cell[2] not in CANONICAL_VIEWS
        ):
            raise ValueError("tune cache contains a held-out family/severity/view")
        metadata_path = Path(entry["metadata_path"])
        if not metadata_path.is_absolute():
            raise ValueError("tune cache paths must be absolute")
        represented.add(cell)
        result.append(
            {
                "cell": cell,
                "metadata_path": metadata_path.resolve(),
                "provenance": CacheProvenance.from_dict(entry["provenance"]),
            }
        )
    expected = {
        (None, None, None),
        *{
            (family, severity, view)
            for family in FITTING_FAMILIES
            for severity in FITTING_SEVERITIES
            for view in CANONICAL_VIEWS
        },
    }
    if represented != expected:
        raise ValueError("tune cache index must cover clean plus the balanced 16 cells")
    return tuple(result)


def _preflight_confidence_training_entries(
    entries,
    records,
    *,
    manifest,
    config,
    seed,
    method,
    encoder_identity,
) -> None:
    """Validate every learned fit exposure from metadata before tensor loading."""

    if encoder_identity is None:
        raise ValueError("production confidence caches require persisted encoder identity")
    fusion_pairs = (
        RSNA_FUSION_PAIRS if config.dataset == "RSNA" else DDSM_FUSION_PAIRS
    )
    schedules = {
        epoch: {item.exam_key: item for item in build_method_training_schedule(
            records, epoch=epoch, seed=seed, method=method
        )}
        for epoch in range(1, config.epochs + 1)
    }
    seen = set()
    actual_order = []
    for entry in entries:
        provenance = load_cache_provenance(
            entry["metadata_path"],
            expected_provenance=entry["provenance"],
            manifest=manifest,
            operation=Operation.CONFIDENCE_FITTING,
            role=Role.CONFIDENCE_FIT,
        )
        epoch = entry["epoch"]
        if provenance.batch_size != 1:
            raise ValueError("confidence schedule caches must contain one exam")
        validate_scheduled_cache_provenance(
            provenance,
            manifest=manifest,
            classifier_checkpoint_sha256=encoder_identity.checkpoint_sha256,
            schedule=schedules.get(epoch, {}),
            encoder_identity=encoder_identity,
            fusion_pairs=fusion_pairs,
        )
        key = (epoch, provenance.exam_keys[0])
        if key in seen:
            raise ValueError("confidence cache index duplicates an epoch/exam exposure")
        seen.add(key)
        actual_order.append(key)
    expected = [
        (epoch, record.exam_key)
        for epoch in range(1, config.epochs + 1)
        for record in records
    ]
    if actual_order != expected:
        raise ValueError(
            "confidence cache index does not follow the exact ordered exposure budget"
        )



def _preflight_tune_cache_entries(
    entries,
    records,
    *,
    manifest,
    config,
    encoder_identity,
) -> None:
    """Reject any incomplete/wrong tune realization before target tensors open."""

    if encoder_identity is None:
        raise ValueError("tune cache validation requires persisted encoder identity")
    record_by_exam = {record.exam_key: record for record in records}
    fusion_pairs = (
        RSNA_FUSION_PAIRS if config.dataset == "RSNA" else DDSM_FUSION_PAIRS
    )
    seen = set()
    for entry in entries:
        provenance = load_cache_provenance(
            entry["metadata_path"],
            expected_provenance=entry["provenance"],
            manifest=manifest,
            operation=Operation.TUNE_SELECTION,
            role=Role.TUNE,
        )
        if provenance.batch_size != 1:
            raise ValueError("tune realization caches must contain one authorized exam")
        exam_key = provenance.exam_keys[0]
        record = record_by_exam.get(exam_key)
        if record is None:
            raise ValueError("tune cache contains an unauthorized exam")
        validate_tune_cache_provenance(
            provenance,
            record=record,
            cell=entry["cell"],
            stress_seed=config.stress_seed,
            encoder_identity=encoder_identity,
            fusion_pairs=fusion_pairs,
        )
        key = (entry["cell"], exam_key)
        if key in seen:
            raise ValueError("tune cache index duplicates a cell/exam realization")
        seen.add(key)
    expected = {
        (cell, record.exam_key)
        for cell in tune_selection_cells()
        for record in records
    }
    if seen != expected:
        raise ValueError("tune cache index must cover every cell and authorized exam exactly")


def _write_private_predictions(
    path: str | Path, predictions: tuple[EvaluationPrediction, ...]
) -> None:
    target = Path(path).resolve()
    if target.suffix != ".json" or not target.parent.exists():
        raise ValueError("private prediction output must be JSON in an existing directory")
    staging = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    for candidate in (target, staging):
        _require_external_or_ignored_destination(candidate)
    document = {"predictions": [asdict(item) for item in predictions]}
    try:
        with staging.open("x", encoding="utf-8") as handle:
            json.dump(document, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(staging, target)
        target.chmod(0o600)
    finally:
        staging.unlink(missing_ok=True)


def _run_confidence_training(args: argparse.Namespace) -> dict[str, object]:
    config = load_research_run_config(args.config)
    if args.method not in LEARNED_TORCH_METHODS:
        raise ValueError("method does not have a learned confidence-fitting workflow")
    if args.seed not in config.seeds:
        raise ValueError("seed is outside the frozen configuration")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    # Operation and role are fixed before the private manifest or cache index is opened.
    manifest = load_role_manifest_for_operation(
        args.manifest,
        expected=_manifest_binding(args.manifest_binding),
        private_root=args.private_root,
        operation=Operation.CONFIDENCE_FITTING,
        role=Role.CONFIDENCE_FIT,
    )

    def authorized(_records):
        selected_classifier = (
            None
            if args.classifier_artifact is None
            else load_selected_classifier_artifact(
                args.classifier_artifact, expected_config=config
            )
        )
        verified_classifier = (
            None if selected_classifier is None else selected_classifier.classifier
        )
        classifier, entries = _cache_index(
            args.cache_index, expected_classifier=verified_classifier
        )
        if classifier.kind not in (
            "public_pretrained_fresh",
            "synthetic_injected",
        ) or not classifier.workflow_complete:
            raise PermissionError(
                "confidence fitting requires a fit-and-tune-selected fresh public classifier"
            )
        if classifier.backbone != config.backbone:
            raise ValueError("cache-index classifier backbone disagrees with configuration")
        if {entry["epoch"] for entry in entries} != set(config.search_table.checkpoint_epochs):
            raise ValueError("cache index epochs disagree with the frozen search table")
        _preflight_confidence_training_entries(
            entries,
            _records,
            manifest=manifest,
            config=config,
            seed=args.seed,
            method=args.method,
            encoder_identity=(
                None
                if selected_classifier is None
                else selected_classifier.encoder_identity
            ),
        )
        first = entries[0]
        first_path = Path(first["metadata_path"]).resolve()
        _require_external_or_ignored_destination(first_path)
        first_bundle = load_cache_bundle(
            first_path,
            expected_provenance=first["provenance"],
            manifest=manifest,
            operation=Operation.CONFIDENCE_FITTING,
            role=Role.CONFIDENCE_FIT,
        )
        torch.manual_seed(args.seed)
        model = build_learned_method(
            args.method,
            config.backbone,
            first_bundle.features.fusion_pairs,
            visual_width=int(
                first_bundle.features.projected_by_view[
                    first_bundle.features.observed_views[0]
                ].shape[1]
            ),
            text_embeddings=first_bundle.features.normalized_text_embeddings,
        ).to(device)
        binding = TrainingBinding(
            protocol_sha256=config.protocol_sha256,
            config_sha256=config.sha256,
            search_table_sha256=config.search_table.sha256,
            manifest_sha256=manifest.manifest_sha256,
            classifier_checkpoint_sha256=classifier.checkpoint_sha256,
            method=args.method,
            seed=args.seed,
        )
        checkpoint_directory = Path(args.epoch_checkpoint_directory).resolve()
        if not checkpoint_directory.is_dir():
            raise ValueError("confidence epoch checkpoint directory must already exist")
        checkpoint_paths: dict[int, Path] = {}
        for epoch in range(1, (args.stop_after_epoch or config.epochs) + 1):
            candidate = checkpoint_directory / f"{args.method}-seed-{args.seed}-epoch-{epoch}.safetensors"
            _require_external_or_ignored_destination(candidate)
            if candidate.exists():
                checkpoint_paths[epoch] = candidate

        def batches(_authorized, schedule, _batch_size, epoch):
            selected = tuple(entry for entry in entries if entry["epoch"] == epoch)
            if not selected:
                raise ValueError("cache index is missing a configured training epoch")
            schedule_by_exam = {item.exam_key: item for item in schedule}
            for entry in selected:
                metadata_path = Path(entry["metadata_path"]).resolve()
                _require_external_or_ignored_destination(metadata_path)
                bundle = load_cache_bundle(
                    metadata_path,
                    expected_provenance=entry["provenance"],
                    manifest=manifest,
                    operation=Operation.CONFIDENCE_FITTING,
                    role=Role.CONFIDENCE_FIT,
                )
                corruption = None
                if args.method == "corruption_auxiliary":
                    rows = []
                    for exam_key in bundle.provenance.exam_keys:
                        draw = schedule_by_exam[exam_key].draw
                        rows.append(
                            [
                                (1 if view in draw.stressed_views else 0)
                                if view in draw.observed_views
                                else -1
                                for view in CANONICAL_VIEWS
                            ]
                        )
                    corruption = torch.tensor(rows, dtype=torch.long)
                yield ConfidenceFitBatch((bundle,), (corruption,))

        def save_epoch(epoch, current):
            destination = checkpoint_directory / (
                f"{args.method}-seed-{args.seed}-epoch-{epoch}.safetensors"
            )
            if destination.exists():
                raise FileExistsError("confidence epoch checkpoint already exists")
            checkpoint_paths[epoch] = save_tensor_checkpoint(current, destination)

        result = fit_confidence_method_with_role_access(
            manifest=manifest,
            model=model,
            method=args.method,
            classifier=classifier,
            config=config,
            seed=args.seed,
            binding=binding,
            batch_loader=batches,
            encoder_identity=(
                None
                if selected_classifier is None
                else selected_classifier.encoder_identity
            ),
            checkpoint_path=args.checkpoint,
            resume=args.resume,
            stop_after_epoch=args.stop_after_epoch,
            epoch_callback=save_epoch,
        )
        fit_artifact = save_confidence_fit_artifact(
            args.training_artifact,
            config=config,
            classifier=classifier,
            classifier_artifact_path=args.classifier_artifact,
            manifest=manifest,
            manifest_path=args.manifest,
            input_evidence_path=args.cache_index,
            binding=binding,
            result=result,
            checkpoint_paths=checkpoint_paths,
            input_cache_file_bindings=tuple(
                bind_cache_files(entry["metadata_path"]) for entry in entries
            ),
        )
        return {
            "status": "trained" if result.completed_epoch == config.epochs else "partial",
            **result.to_dict(),
            "training_artifact_sha256": fit_artifact.sha256,
            "classifier_kind": classifier.kind,
            "classifier_workflow_complete": classifier.workflow_complete,
            "patient_readiness": (
                "verified" if classifier.patient_readiness_verified else "not_verified"
            ),
        }

    return run_with_role_access(
        manifest,
        operation=Operation.CONFIDENCE_FITTING,
        roles=(Role.CONFIDENCE_FIT,),
        loader=authorized,
    )


def _run_confidence_selection(args: argparse.Namespace) -> dict[str, object]:
    config = load_research_run_config(args.config)
    manifest = load_role_manifest_for_operation(
        args.manifest,
        expected=_manifest_binding(args.manifest_binding),
        private_root=args.private_root,
        operation=Operation.TUNE_SELECTION,
        role=Role.TUNE,
    )
    fit = load_confidence_fit_artifact(args.training_artifact, expected_config=config)
    if fit.result.completed_epoch != config.epochs:
        raise ValueError("confidence fit has not completed the frozen epoch budget")
    checkpoint_paths = dict(fit.checkpoint_paths)
    results = []
    for epoch in config.search_table.checkpoint_epochs:

        def predictions(records, *, current_epoch=epoch):
            entries = _tune_cache_index(args.tune_cache_index)
            _preflight_tune_cache_entries(
                entries,
                records,
                manifest=manifest,
                config=config,
                encoder_identity=fit.encoder_identity,
            )
            labels = {record.exam_key: record.density for record in records}
            model = None
            rows = []
            for entry in entries:
                bundle = load_cache_bundle(
                    entry["metadata_path"],
                    expected_provenance=entry["provenance"],
                    manifest=manifest,
                    operation=Operation.TUNE_SELECTION,
                    role=Role.TUNE,
                )
                if bundle.provenance.checkpoint_sha256 != fit.classifier.checkpoint_sha256:
                    raise ValueError("tune cache classifier binding is stale")
                if model is None:
                    model = load_learned_confidence_checkpoint(
                        method=fit.result.method,
                        seed=fit.result.seed,
                        checkpoint_path=checkpoint_paths[current_epoch],
                        config=config,
                        example=bundle,
                        device=args.device,
                    )
                with torch.no_grad():
                    confidence = confidence_bundle_scores(
                        model, fit.result.method, bundle
                    ).cpu()
                prediction = bundle.features.prediction.cpu()
                family, severity, target_view = entry["cell"]
                for index, exam_key in enumerate(bundle.provenance.exam_keys):
                    if exam_key not in labels:
                        raise ValueError("tune cache contains an unauthorized exam")
                    rows.append(
                        TunePrediction(
                            exam_id=exam_key,
                            target=labels[exam_key],
                            prediction=int(prediction[index]),
                            confidence=float(confidence[index]),
                            family=family,
                            severity=severity,
                            target_view=target_view,
                        )
                    )
            return tuple(rows)

        results.append(
            generate_checkpoint_tune_result_with_role_access(
                manifest,
                method=fit.result.method,
                seed=fit.result.seed,
                classifier_checkpoint_sha256=fit.classifier.checkpoint_sha256,
                epoch=epoch,
                artifact_path=checkpoint_paths[epoch],
                prediction_reader=predictions,
            )
        )
    materialized = tuple(results)
    if fit.result.method == "correctness_mvacn":
        if args.clean_reference_record is not None:
            raise ValueError("correctness-MVACN creates, rather than consumes, the clean reference")
        reference = select_clean_reference(materialized)
        chosen = reference
    else:
        if args.clean_reference_record is None:
            raise ValueError("candidate/matched selection requires the persisted clean reference")
        reference = load_clean_reference_checkpoint(args.clean_reference_record)
        if (
            reference.seed != fit.result.seed
            or reference.classifier_checkpoint_sha256 != fit.classifier.checkpoint_sha256
            or reference.tune_manifest_sha256 != manifest.manifest_sha256
        ):
            raise ValueError("clean-reference selection binding is stale")
        chosen = select_candidate_checkpoint(materialized, clean_reference=reference)
        if chosen is None:
            return {
                "status": "no_eligible_selection",
                "method": fit.result.method,
                "seed": fit.result.seed,
                "selection_trials": len(materialized),
                "patient_readiness": (
                    "verified"
                    if fit.classifier.patient_readiness_verified
                    else "synthetic_only"
                ),
            }
    selected = ModelArtifactSelection.from_learned_workflow(
        artifact_path=next(
            path
            for epoch, path in fit.checkpoint_paths
            if epoch == chosen.epoch and sha256_file(path) == chosen.artifact_sha256
        ),
        evidence_path=args.output,
        config=config,
        classifier=fit.classifier,
        training_binding=fit.binding,
        training_result=fit.result,
        checkpoints=materialized,
        clean_reference=reference,
        training_artifact_path=fit.source_path,
        tune_evidence_path=args.tune_cache_index,
    )
    return {
        "status": "selected",
        "method": selected.method,
        "seed": selected.seed,
        "selected_artifact_sha256": selected.artifact_sha256,
        "selection_record_sha256": selected.workflow_evidence_sha256,
        "selection_trials": selected.selection_trials,
        "patient_readiness": (
            "verified" if fit.classifier.patient_readiness_verified else "synthetic_only"
        ),
    }


def run_view_risk_command(args: argparse.Namespace) -> dict[str, object] | None:
    if args.command not in COMMANDS:
        return None
    if args.command == "view-risk-validate-config":
        config = load_research_run_config(args.config)
        return {
            "status": "valid",
            "config_sha256": config.sha256,
            "search_table_sha256": config.search_table.sha256,
            "dataset": config.dataset,
            "backbone": config.backbone,
            "epochs": config.epochs,
            "method_count": len(config.methods),
            "patient_readiness": "not_assessed",
        }
    if args.command == "view-risk-fit-classifier":
        config = load_research_run_config(args.config)
        manifest = load_role_manifest_for_operation(
            args.manifest,
            expected=_manifest_binding(args.manifest_binding),
            private_root=args.private_root,
            operation=Operation.CLASSIFIER_FITTING,
            role=Role.CLASSIFIER_FIT,
        )
        initialization_path = Path(args.initialization).resolve()
        if initialization_path.exists():
            initialization = load_public_initialization_artifact(initialization_path)
            model, input_ids = reload_verified_public_classifier(
                initialization, device=args.device
            )
        else:
            if args.resume:
                raise ValueError("classifier resume requires the existing initialization artifact")
            model, input_ids, provenance = load_pinned_public_clip_classifier(
                config.backbone, config.dataset, device=args.device
            )
            initialization = save_public_initialization_artifact(
                initialization_path,
                model=model,
                input_ids=input_ids,
                provenance=provenance,
                dataset=config.dataset,
            )
        fit = fit_classifier_images_with_role_access(
            artifact_path=args.output,
            manifest=manifest,
            manifest_path=args.manifest,
            model=model,
            input_ids=input_ids,
            initialization=initialization,
            readiness_artifact_path=args.readiness,
            config=config,
            image_root=args.image_root,
            checkpoint_directory=args.checkpoint_directory,
            resume_checkpoint_path=args.resume_checkpoint,
            resume=args.resume,
            stop_after_epoch=args.stop_after_epoch,
        )
        return {
            "status": "classifier_fit_complete"
            if fit.result.completed_epoch == config.epochs
            else "classifier_fit_partial",
            "fit_artifact_sha256": fit.sha256,
            "initialization_sha256": fit.initialization.sha256,
            "completed_epoch": fit.result.completed_epoch,
            "update_count": fit.result.update_count,
            "exposed_record_count": fit.result.exposed_record_count,
            "patient_readiness": (
                "verified" if fit.readiness.patient_ready else "not_verified"
            ),
        }
    if args.command == "view-risk-select-classifier":
        config = load_research_run_config(args.config)
        manifest = load_role_manifest_for_operation(
            args.manifest,
            expected=_manifest_binding(args.manifest_binding),
            private_root=args.private_root,
            operation=Operation.TUNE_SELECTION,
            role=Role.TUNE,
        )
        selected = select_classifier_from_images_with_role_access(
            output_path=args.output,
            fit_artifact_path=args.fit_artifact,
            tune_manifest=manifest,
            tune_manifest_path=args.manifest,
            image_root=args.image_root,
            config=config,
            device=args.device,
        )
        return {
            "status": "classifier_selected",
            "selected_classifier_artifact_sha256": selected.sha256,
            "classifier_checkpoint_sha256": selected.classifier.checkpoint_sha256,
            "classifier_fit_update_count": selected.classifier.classifier_fit_update_count,
            "patient_readiness": (
                "verified"
                if selected.classifier.patient_readiness_verified
                else "not_verified"
            ),
        }
    if args.command == "view-risk-fit-control":
        config = load_research_run_config(args.config)
        analytic_methods = RAW_CONTROL_METHODS | {"temperature_scaled_msp", "ds_logistic"}
        if args.method not in analytic_methods or args.seed not in config.seeds:
            raise ValueError("analytic control method/seed is outside the frozen table")
        if args.calibrate_scalar and args.method not in RAW_CONTROL_METHODS:
            raise ValueError("scalar calibration is available only for raw scalar controls")
        needs_tune = args.method in {"temperature_scaled_msp", "ds_logistic"} or (
            args.calibrate_scalar
        )
        selected_classifier = load_selected_classifier_artifact(
            args.classifier_artifact, expected_config=config
        )
        classifier = selected_classifier.classifier
        encoder_identity = selected_classifier.encoder_identity
        tune_manifest = None
        tune_entries = None
        tune_expected_exam_keys = None
        tune_evidence = None
        if needs_tune:
            if not all((
                args.tune_manifest, args.tune_manifest_binding,
                args.tune_private_root, args.tune_cache_index,
            )):
                raise ValueError("fitted analytic control requires complete tune cache inputs")
            tune_manifest = load_role_manifest_for_operation(
                args.tune_manifest,
                expected=_manifest_binding(args.tune_manifest_binding),
                private_root=args.tune_private_root,
                operation=Operation.TUNE_SELECTION,
                role=Role.TUNE,
            )
            tune_semantics = (
                "ds_tune_regularization"
                if args.method == "ds_logistic"
                else "clean_full_tune"
            )
            tune_entries = _control_cache_index(
                args.tune_cache_index,
                classifier=classifier,
                semantics=tune_semantics,
            )
            tune_expected_exam_keys = _preflight_control_cache_entries(
                tune_entries,
                tune_manifest.records,
                semantics=tune_semantics,
                manifest=tune_manifest,
                config=config,
                seed=args.seed,
                encoder_identity=encoder_identity,
            )
            tune_evidence = _control_cache_evidence(
                args.tune_cache_index, tune_entries, prefix="tune"
            )
        confidence_manifest = None
        confidence_entries = None
        confidence_expected_exam_keys = None
        confidence_evidence = None
        if args.method == "ds_logistic":
            if not all((
                args.confidence_manifest, args.confidence_manifest_binding,
                args.confidence_private_root, args.confidence_cache_index,
            )):
                raise ValueError("DS control requires complete confidence-fit cache inputs")
            confidence_manifest = load_role_manifest_for_operation(
                args.confidence_manifest,
                expected=_manifest_binding(args.confidence_manifest_binding),
                private_root=args.confidence_private_root,
                operation=Operation.CONFIDENCE_FITTING,
                role=Role.CONFIDENCE_FIT,
            )
            confidence_entries = _control_cache_index(
                args.confidence_cache_index,
                classifier=classifier,
                semantics="ds_confidence_fit",
            )
            confidence_expected_exam_keys = _preflight_control_cache_entries(
                confidence_entries,
                confidence_manifest.records,
                semantics="ds_confidence_fit",
                manifest=confidence_manifest,
                config=config,
                seed=args.seed,
                encoder_identity=encoder_identity,
            )
            confidence_evidence = _control_cache_evidence(
                args.confidence_cache_index, confidence_entries, prefix="confidence"
            )
        if args.method in RAW_CONTROL_METHODS and not args.calibrate_scalar:
            if any(value is not None for value in (
                args.tune_manifest, args.tune_manifest_binding, args.tune_private_root,
                args.tune_cache_index, args.confidence_manifest,
                args.confidence_manifest_binding, args.confidence_private_root,
                args.confidence_cache_index,
            )):
                raise ValueError("raw ranking controls do not consume fitting caches")
            artifact = create_raw_control_artifact(
                args.output,
                method=args.method,
                config=config,
                classifier=classifier,
                encoder_identity=encoder_identity,
                seed=args.seed,
            )
        elif args.calibrate_scalar:
            assert (
                tune_manifest is not None
                and tune_entries is not None
                and tune_expected_exam_keys is not None
            )
            rows, orientation = _calibration_rows_from_control_caches(
                args.method,
                tune_entries,
                tune_expected_exam_keys,
                manifest=tune_manifest,
                classifier=classifier,
            )
            artifact = fit_scalar_calibration_artifact_with_role_access(
                args.output,
                method=args.method,
                orientation=orientation,
                config=config,
                classifier=classifier,
                encoder_identity=encoder_identity,
                seed=args.seed,
                tune_manifest=tune_manifest,
                row_reader=lambda _records: rows,
                tune_manifest_path=args.tune_manifest,
                tune_evidence_files=tune_evidence,
            )
        elif args.method == "temperature_scaled_msp":
            assert (
                tune_manifest is not None
                and tune_entries is not None
                and tune_expected_exam_keys is not None
            )
            artifact = fit_temperature_control_artifact_with_role_access(
                args.output,
                config=config,
                classifier=classifier,
                encoder_identity=encoder_identity,
                seed=args.seed,
                tune_manifest=tune_manifest,
                row_reader=lambda _records: _scalar_rows_from_control_caches(
                    tune_entries,
                    tune_expected_exam_keys,
                    manifest=tune_manifest,
                    operation=Operation.TUNE_SELECTION,
                    role=Role.TUNE,
                ),
                tune_manifest_path=args.tune_manifest,
                tune_evidence_files=tune_evidence,
            )
        else:
            assert (
                tune_manifest is not None
                and tune_entries is not None
                and tune_expected_exam_keys is not None
                and confidence_manifest is not None
                and confidence_entries is not None
                and confidence_expected_exam_keys is not None
            )
            artifact = fit_ds_control_artifact_with_role_access(
                args.output,
                config=config,
                classifier=classifier,
                encoder_identity=encoder_identity,
                seed=args.seed,
                confidence_manifest=confidence_manifest,
                tune_manifest=tune_manifest,
                confidence_reader=lambda _records: _ds_rows_from_control_caches(
                    confidence_entries,
                    confidence_expected_exam_keys,
                    manifest=confidence_manifest,
                    operation=Operation.CONFIDENCE_FITTING,
                    role=Role.CONFIDENCE_FIT,
                ),
                tune_reader=lambda _records: _ds_rows_from_control_caches(
                    tune_entries,
                    tune_expected_exam_keys,
                    manifest=tune_manifest,
                    operation=Operation.TUNE_SELECTION,
                    role=Role.TUNE,
                ),
                confidence_manifest_path=args.confidence_manifest,
                tune_manifest_path=args.tune_manifest,
                confidence_evidence_files=confidence_evidence,
                tune_evidence_files=tune_evidence,
                confidence_expected_exam_keys=confidence_expected_exam_keys,
                tune_expected_exam_keys=tune_expected_exam_keys,
            )
        return {
            "status": "control_fitted" if artifact.parameter_count else "control_recorded",
            "method": artifact.method,
            "seed": artifact.seed,
            "artifact_sha256": artifact.sha256,
            "parameter_count": artifact.parameter_count,
            "update_count": artifact.update_count,
            "selection_trials": artifact.selection_trials,
            "output_kind": artifact.output_kind,
            "patient_readiness": (
                "verified" if classifier.patient_readiness_verified else "not_verified"
            ),
        }
    if args.command == "view-risk-select":
        config = load_research_run_config(args.config)
        reference_results = validate_checkpoint_selection_budget(
            _checkpoint_results(args.reference_results), config.search_table
        )
        candidate_results = validate_checkpoint_selection_budget(
            _checkpoint_results(args.candidate_results), config.search_table
        )
        reference = select_clean_reference(reference_results)
        selected = select_candidate_checkpoint(
            candidate_results, clean_reference=reference
        )
        return {
            "status": "selected" if selected is not None else "no_eligible_selection",
            "reference_epoch": reference.epoch,
            "reference_clean_aurc": reference.clean_aurc,
            "selected_epoch": None if selected is None else selected.epoch,
            "selected_clean_aurc": None if selected is None else selected.clean_aurc,
            "selected_stress_aurc": None if selected is None else selected.stress_aurc,
            "selected_artifact_sha256": (
                None if selected is None else selected.artifact_sha256
            ),
        }
    if args.command == "view-risk-freeze-pilot":
        config = load_research_run_config(args.config)
        bindings = _read_json(args.bindings, description="pilot binding file")
        if not isinstance(bindings, dict) or set(bindings) not in (
            {"manifest_sha256_by_role", "classifier"},
            {"manifest_sha256_by_role", "classifier_artifact_path"},
        ):
            raise ValueError("pilot binding file has missing or unknown fields")
        selections_value = _read_json(args.selections, description="selection file")
        if not isinstance(selections_value, dict) or set(selections_value) not in (
            {"selections"}, {"selection_record_paths"}
        ):
            raise ValueError("selection file has missing or unknown fields")
        manifests = bindings["manifest_sha256_by_role"]
        if not isinstance(manifests, dict):
            raise ValueError("manifest bindings must be a mapping")
        classifier_artifact_path = bindings.get("classifier_artifact_path")
        if classifier_artifact_path is not None:
            classifier = load_selected_classifier_artifact(
                classifier_artifact_path, expected_config=config
            ).classifier
        else:
            classifier = _classifier_from_dict(bindings["classifier"])
            if classifier.kind != "synthetic_injected":
                raise PermissionError(
                    "production pilot bindings require a verified selected-classifier artifact"
                )
        selections = []
        if "selection_record_paths" in selections_value:
            paths = selections_value["selection_record_paths"]
            if classifier.kind != "public_pretrained_fresh" or not isinstance(paths, list):
                raise ValueError("production selection records require a production classifier")
            indexed: dict[tuple[str, int], Path] = {}
            for raw_path in paths:
                record_path = Path(raw_path).resolve()
                record = _read_json(record_path, description="selection record")
                payload = record.get("artifact", record) if isinstance(record, dict) else None
                if not isinstance(payload, dict):
                    raise ValueError("selection record schema is invalid")
                key = (payload.get("method"), payload.get("seed"))
                if not isinstance(key[0], str) or not isinstance(key[1], int) or key in indexed:
                    raise ValueError("selection records contain invalid or duplicate identities")
                indexed[key] = record_path
            expected_keys = {(method, seed) for method in config.methods for seed in config.seeds}
            if set(indexed) != expected_keys:
                raise ValueError("selection records do not cover the frozen method/seed table")
            for seed in config.seeds:
                reference_path = indexed[("correctness_mvacn", seed)]
                for method in config.methods:
                    selections.append(
                        load_model_artifact_selection(
                            indexed[(method, seed)],
                            config=config,
                            classifier=classifier,
                            clean_reference_record_path=(
                                None if method == "correctness_mvacn" else reference_path
                            ),
                        )
                    )
        else:
            raw_selections = selections_value["selections"]
            if classifier.kind != "synthetic_injected" or not isinstance(raw_selections, list):
                raise ValueError("inline selections are restricted to synthetic software plans")
            for value in raw_selections:
                if not isinstance(value, dict) or set(value) != {
                    "method", "seed", "artifact_path", "evidence_kind"
                } or value["evidence_kind"] != "synthetic_software":
                    raise ValueError("synthetic selection entry has an invalid schema")
                selections.append(
                    ModelArtifactSelection.synthetic_from_artifact(
                        value["artifact_path"],
                        method=value["method"],
                        seed=value["seed"],
                        config=config,
                        classifier=classifier,
                    )
                )
        plan = freeze_pilot_plan(
            args.output,
            config=config,
            manifest_sha256_by_role=manifests,
            classifier=classifier,
            selections=selections,
            classifier_artifact_path=classifier_artifact_path,
        )
        return {"status": "frozen", "pilot_plan_sha256": plan.sha256}

    if args.command == "view-risk-train-confidence":
        return _run_confidence_training(args)
    if args.command == "view-risk-select-confidence":
        return _run_confidence_selection(args)

    config = load_research_run_config(args.config)
    # The label-free frozen plan is verified before any target-bearing pilot
    # manifest is opened; the operation/role is fixed before that manifest read.
    plan = load_pilot_plan(args.plan, expected_config_sha256=config.sha256)
    manifest = load_role_manifest_for_operation(
        args.manifest,
        expected=_manifest_binding(args.manifest_binding),
        private_root=args.private_root,
        operation=Operation.PILOT_EVALUATION,
        role=Role.PILOT,
    )
    if args.command == "view-risk-freeze-classifier-predictions":
        if Path(args.output).resolve() != plan.authoritative_prediction_path:
            raise ValueError("output must be the plan-owned authoritative registration path")
        entries = _evaluation_cache_index(args.cache_index)
        _prevalidate_evaluation_cache_entries(
            plan,
            entries,
            manifest.records,
            manifest=manifest,
        )
        cache_files = tuple(bind_cache_files(entry["metadata_path"]) for entry in entries)

        def prediction_reader(records):
            record_by_exam = {record.exam_key: record for record in records}
            rows = []
            for entry in entries:
                bundle = load_cache_bundle(
                    entry["metadata_path"],
                    expected_provenance=entry["provenance"],
                    manifest=manifest,
                    operation=Operation.PILOT_EVALUATION,
                    role=Role.PILOT,
                )
                if (
                    bundle.provenance.batch_size != 1
                    or bundle.provenance.exam_keys[0] not in record_by_exam
                ):
                    raise ValueError("evaluation cache must contain one authorized exam")
                rows.append(
                    validate_evaluation_cache_bundle(
                        plan,
                        bundle,
                        entry["cell"],
                        record_by_exam[bundle.provenance.exam_keys[0]],
                    )
                )
            return tuple(rows)

        predictions = freeze_authoritative_predictions(
            plan.authoritative_prediction_path,
            plan,
            manifest=manifest,
            role=Role.PILOT,
            prediction_reader=prediction_reader,
            source_cache_files=cache_files,
        )
        return {
            "status": "classifier_predictions_frozen",
            "prediction_sha256": predictions.sha256,
            "row_count": len(predictions.rows),
            "patient_readiness": "verified" if plan.patient_ready else "synthetic_only",
        }
    authoritative = load_authoritative_predictions_with_role_access(
        plan,
        manifest=manifest,
        role=Role.PILOT,
    )
    if args.command == "view-risk-score-confidence":
        selection_records = {
            key: Path(record_path)
            for key, record_path, _record_sha in plan.selection_record_map
        }
        selection_key = f"{args.method}:{args.seed}"
        record_path = selection_records.get(selection_key)
        reference_path = selection_records.get(f"correctness_mvacn:{args.seed}")
        if record_path is None or reference_path is None:
            raise ValueError("confidence method/seed selection is absent from the frozen plan")
        selection = load_model_artifact_selection(
            record_path,
            config=config,
            classifier=plan.classifier,
            clean_reference_record_path=(
                None if args.method == "correctness_mvacn" else reference_path
            ),
        )

        def score_reader(records, panel_rows):
            record_by_exam = {record.exam_key: record for record in records}
            entries = _evaluation_cache_index(args.cache_index)
            expected_by_cell: dict[tuple[object, ...], set[str]] = {}
            prediction_by_key = {}
            for row in panel_rows:
                expected_by_cell.setdefault(row.cell_key, set()).add(row.exam_id)
                prediction_by_key[row.row_key] = row.prediction
            _prevalidate_evaluation_cache_entries(
                plan,
                entries,
                records,
                manifest=manifest,
                expected_by_cell=expected_by_cell,
            )
            seen: dict[tuple[object, ...], set[str]] = {
                cell: set() for cell in expected_by_cell
            }
            output = []
            learned_model = None
            for entry in entries:
                cell_value = entry["cell"]
                cell = (
                    cell_value["panel"],
                    cell_value["family"],
                    cell_value["severity"],
                    cell_value["target_view"],
                    cell_value["variant"],
                    cell_value["mask"],
                    cell_value["realization_id"],
                )
                if cell not in expected_by_cell:
                    raise ValueError("evaluation cache index contains an unrequested panel/cell")
                bundle = load_cache_bundle(
                    entry["metadata_path"],
                    expected_provenance=entry["provenance"],
                    manifest=manifest,
                    operation=Operation.PILOT_EVALUATION,
                    role=Role.PILOT,
                )
                if (
                    bundle.provenance.batch_size != 1
                    or bundle.provenance.exam_keys[0] not in record_by_exam
                ):
                    raise ValueError("evaluation cache must contain one authorized exam")
                validated = validate_evaluation_cache_bundle(
                    plan,
                    bundle,
                    cell_value,
                    record_by_exam[bundle.provenance.exam_keys[0]],
                )
                if validated.row_key not in prediction_by_key:
                    raise ValueError("evaluation cache realization is not authoritative")
                if (
                    bundle.provenance.checkpoint_sha256
                    != plan.classifier.checkpoint_sha256
                    or any(
                        exam not in expected_by_cell[cell] or exam in seen[cell]
                        for exam in bundle.provenance.exam_keys
                    )
                ):
                    raise ValueError("evaluation cache cohort/classifier binding is stale")
                if selection.method in LEARNED_TORCH_METHODS and learned_model is None:
                    learned_model = load_learned_confidence_model(
                        selection,
                        config=config,
                        classifier=plan.classifier,
                        example=bundle,
                        device=args.device,
                    )
                scored = score_confidence_cache_bundle(
                    selection, bundle, learned_model=learned_model
                )
                for index, exam_key in enumerate(bundle.provenance.exam_keys):
                    row_key = (exam_key, *cell)
                    if int(scored.classifier_prediction[index]) != prediction_by_key[row_key]:
                        raise ValueError(
                            "evaluation cache prediction disagrees with authoritative classifier"
                        )
                    seen[cell].add(exam_key)
                    output.append(
                        ConfidenceScore(
                            row_key,
                            float(scored.confidence[index]),
                            scored.confidence_kind,
                        )
                    )
            if any(seen[cell] != exams for cell, exams in expected_by_cell.items()):
                raise ValueError("evaluation caches omit rows from the authorized panel cohort")
            return tuple(output)

        predictions = generate_confidence_predictions_with_role_access(
            plan,
            authoritative_predictions=authoritative,
            manifest=manifest,
            role=Role.PILOT,
            method=args.method,
            seed=args.seed,
            panel=args.panel,
            score_reader=score_reader,
        )
        _write_private_predictions(args.output, predictions)
        return {
            "status": "confidence_predictions_written",
            "method": args.method,
            "seed": args.seed,
            "panel": args.panel,
            "row_count": len(predictions),
            "model_sha256": selection.artifact_sha256,
            "authoritative_prediction_sha256": authoritative.sha256,
            "patient_readiness": "verified" if plan.patient_ready else "synthetic_only",
        }
    summary = evaluate_prediction_panel_with_role_access(
        plan,
        authoritative_predictions=authoritative,
        manifest=manifest,
        role=Role.PILOT,
        method=args.method,
        seed=args.seed,
        model_sha256=args.model_sha256,
        model_artifact_path=args.model_artifact,
        prediction_reader=lambda _records: _prediction_rows(args.predictions),
    )
    return {
        "status": "evaluated",
        "method": summary.method,
        "seed": summary.seed,
        "panel": summary.panel,
        "mean_aurc": summary.mean_aurc,
        "patient_count": summary.patient_count,
        "exam_count": summary.exam_count,
        "pilot_plan_sha256": plan.sha256,
        "patient_readiness": "verified" if summary.patient_ready else "synthetic_only",
    }


__all__ = ["COMMANDS", "add_view_risk_subparsers", "run_view_risk_command"]
