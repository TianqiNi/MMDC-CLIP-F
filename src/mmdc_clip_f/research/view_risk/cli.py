"""Working CLI dispatch for guarded view-risk research workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .cache import (
    CacheProvenance,
    _require_external_or_ignored_destination,
    load_cache_bundle,
)
from .evaluation import (
    CheckpointTuneResult,
    ModelArtifactSelection,
    TuneCellResult,
    evaluate_prediction_panel_with_role_access,
    freeze_pilot_plan,
    load_pilot_plan,
    select_candidate_checkpoint,
    select_clean_reference,
    validate_checkpoint_selection_budget,
)
from .inputs import CANONICAL_VIEWS
from .metrics import EvaluationPrediction
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
    fit_confidence_method_with_role_access,
    load_research_run_config,
    load_role_manifest_for_operation,
)


COMMANDS = frozenset(
    {
        "view-risk-validate-config",
        "view-risk-select",
        "view-risk-freeze-pilot",
        "view-risk-evaluate",
        "view-risk-train-confidence",
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


def _cache_index(path: str | Path) -> tuple[ClassifierProvenance, tuple[dict[str, object], ...]]:
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
    return _classifier_from_dict(value["classifier"]), tuple(normalized)


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
        classifier, entries = _cache_index(args.cache_index)
        if classifier.kind != "public_pretrained_fresh" or not classifier.workflow_complete:
            raise PermissionError(
                "confidence fitting requires a fit-and-tune-selected fresh public classifier"
            )
        if classifier.backbone != config.backbone:
            raise ValueError("cache-index classifier backbone disagrees with configuration")
        if {entry["epoch"] for entry in entries} != set(config.search_table.checkpoint_epochs):
            raise ValueError("cache index epochs disagree with the frozen search table")
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

        result = fit_confidence_method_with_role_access(
            manifest=manifest,
            model=model,
            method=args.method,
            classifier=classifier,
            config=config,
            seed=args.seed,
            binding=binding,
            batch_loader=batches,
            checkpoint_path=args.checkpoint,
            resume=args.resume,
            stop_after_epoch=args.stop_after_epoch,
        )
        return {
            "status": "trained",
            **result.to_dict(),
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
        if not isinstance(bindings, dict) or set(bindings) != {
            "manifest_sha256_by_role",
            "classifier",
        }:
            raise ValueError("pilot binding file has missing or unknown fields")
        selections_value = _read_json(args.selections, description="selection file")
        if not isinstance(selections_value, dict) or set(selections_value) != {"selections"}:
            raise ValueError("selection file has missing or unknown fields")
        raw_selections = selections_value["selections"]
        if not isinstance(raw_selections, list):
            raise ValueError("selections must be a list")
        try:
            selections = tuple(ModelArtifactSelection(**value) for value in raw_selections)
        except TypeError as exc:
            raise ValueError("selection entry has an invalid schema") from exc
        manifests = bindings["manifest_sha256_by_role"]
        if not isinstance(manifests, dict):
            raise ValueError("manifest bindings must be a mapping")
        plan = freeze_pilot_plan(
            args.output,
            config=config,
            manifest_sha256_by_role=manifests,
            classifier=_classifier_from_dict(bindings["classifier"]),
            selections=selections,
        )
        return {"status": "frozen", "pilot_plan_sha256": plan.sha256}

    if args.command == "view-risk-train-confidence":
        return _run_confidence_training(args)

    config = load_research_run_config(args.config)
    plan = load_pilot_plan(args.plan, expected_config_sha256=config.sha256)
    # The requested operation/role is fixed before either private file is read.
    manifest = load_role_manifest_for_operation(
        args.manifest,
        expected=_manifest_binding(args.manifest_binding),
        private_root=args.private_root,
        operation=Operation.PILOT_EVALUATION,
        role=Role.PILOT,
    )
    summary = evaluate_prediction_panel_with_role_access(
        plan,
        manifest=manifest,
        role=Role.PILOT,
        method=args.method,
        seed=args.seed,
        model_sha256=args.model_sha256,
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
