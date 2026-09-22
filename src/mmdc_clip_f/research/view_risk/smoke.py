"""Download-free P4C software smoke, resource preflight, and bounded timings.

This module intentionally uses tiny injected tensors while composing the accepted
view-risk feature, cache, target, fitting, artifact, inference, control, and metric
interfaces.  Its reports are software evidence only: they cannot establish real
patient readiness, a pilot decision, empirical benefit, or deployed CLIP cost.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import shutil
import statistics
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmdc_clip_f.backbones import BACKBONES, PROMPTS
from mmdc_clip_f.model import MultiViewCLIPClassifier
from mmdc_clip_f.provenance import sha256_file

from .artifacts import (
    DSControlRows,
    analytic_control_confidence,
    create_raw_control_artifact,
    fit_ds_control_artifact_with_role_access,
    load_control_artifact,
)
from .baselines import DSBaselineFeatures, build_ds_features
from .cache import (
    CacheBundle,
    _require_external_or_ignored_destination,
    build_exam_cache_with_role_access,
    load_cache_bundle,
    save_cache_bundle,
)
from .evaluation import EvaluationPrediction
from .features import (
    LEGACY_IMAGE_MEAN,
    LEGACY_IMAGE_STD,
    VerifiedFrozenEncoder,
    load_verified_frozen_encoder,
)
from .fusion import DDSM_FUSION_PAIRS, RSNA_FUSION_PAIRS, FusionPairs
from .head import count_trainable_parameters
from .head_inputs import prepare_raw_head_inputs
from .inference import load_learned_confidence_checkpoint
from .inputs import CANONICAL_VIEWS
from .metrics import confidence_panel_metrics, effect_metrics, evaluate_aurc_panel
from .perturbations import (
    PerturbationSpec,
    RealizedParent,
    enumerate_proper_masks,
    realize_parent,
    realize_training_parent,
)
from .production import (
    load_confidence_fit_artifact,
    save_confidence_fit_artifact,
    save_tensor_checkpoint,
)
from .roles import (
    Operation,
    PatientMappingDeclaration,
    PrivateExamRecord,
    Role,
    RoleManifest,
    ViewReference,
    save_private_manifest,
)
from .training import (
    MANDATORY_METHODS,
    ClassifierProvenance,
    ConfidenceFitBatch,
    ResearchRunConfig,
    TrainingBinding,
    audit_software_fixture,
    build_learned_method,
    build_method_training_schedule,
    checked_intervention_targets,
    confidence_bundle_scores,
    fit_confidence_method_with_role_access,
)


SMOKE_REPORT_VERSION = "view-risk-p4c-smoke-report/v1"
_SMOKE_IMPLEMENTATION = "view-risk-p4c-synthetic-smoke/v1"
_FULL_MASK = tuple(CANONICAL_VIEWS)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return float(
        sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction
    )


def _cpu_high_water_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB while macOS reports bytes.
    return value if sys.platform == "darwin" else value * 1024


def measure_repeated_cost(
    name: str,
    work: Callable[[], object],
    *,
    device: str | torch.device,
    warmup: int,
    repeats: int,
    _clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, object]:
    """Measure one callable with explicit warmup, synchronization, and units."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("measurement name must be nonempty")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("smoke timing device must be cpu or cuda")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA timing was requested but CUDA is unavailable")

    def synchronize() -> None:
        if target.type == "cuda":
            torch.cuda.synchronize(target)

    for _ in range(warmup):
        work()
    synchronize()
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)

    elapsed_ms: list[float] = []
    for _ in range(repeats):
        synchronize()
        started = _clock_ns()
        work()
        synchronize()
        elapsed_ms.append((_clock_ns() - started) / 1_000_000.0)

    ordered = sorted(elapsed_ms)
    if target.type == "cuda":
        memory: dict[str, object] = {
            "metric": "cuda_max_memory_allocated",
            "bytes": int(torch.cuda.max_memory_allocated(target)),
            "scope": "measurement_after_warmup_since_peak_reset",
        }
        synchronization = "torch_cuda_synchronize_before_and_after_each_call"
    else:
        memory = {
            "metric": "cpu_process_high_water_rss",
            "bytes": _cpu_high_water_rss_bytes(),
            "scope": "process_lifetime_high_water_not_operation_delta",
        }
        synchronization = "not_applicable_cpu"
    return {
        "name": name,
        "measurement_status": "measured",
        "measurement_kind": "wall_clock_repeated_calls",
        "unit": "milliseconds_per_call",
        "device": str(target),
        "warmup_calls": warmup,
        "timed_calls": repeats,
        "median": float(statistics.median(ordered)),
        "minimum": float(ordered[0]),
        "maximum": float(ordered[-1]),
        "p25": _percentile(ordered, 0.25),
        "p75": _percentile(ordered, 0.75),
        "iqr": _percentile(ordered, 0.75) - _percentile(ordered, 0.25),
        "synchronization": synchronization,
        "memory": memory,
    }


def _package_versions() -> dict[str, str]:
    versions = {}
    for distribution in (
        "numpy",
        "Pillow",
        "pydicom",
        "PyYAML",
        "safetensors",
        "scikit-learn",
        "torch",
        "torchvision",
        "transformers",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "unavailable"
    return versions


def _public_resource_status() -> dict[str, object]:
    configured = os.environ.get("HF_HUB_CACHE")
    if configured is None:
        hf_home = os.environ.get("HF_HOME")
        cache_root = (
            Path(hf_home).expanduser() / "hub"
            if hf_home
            else Path.home() / ".cache" / "huggingface" / "hub"
        )
    else:
        cache_root = Path(configured).expanduser()
    result: dict[str, object] = {}
    for name, specification in sorted(BACKBONES.items()):
        model_directory = "models--" + specification.hf_model.replace("/", "--")
        snapshot = cache_root / model_directory / "snapshots" / specification.revision
        try:
            config_available = (snapshot / "config.json").is_file()
            weights_available = any(
                (snapshot / filename).is_file()
                for filename in ("model.safetensors", "pytorch_model.bin")
            )
            tokenizer_available = any(
                (snapshot / filename).is_file()
                for filename in ("tokenizer.json", "vocab.json")
            )
        except OSError:
            config_available = weights_available = tokenizer_available = False
        complete = config_available and weights_available and tokenizer_available
        result[name] = {
            "status": "available_not_content_audited" if complete else "unavailable",
            "pinned_revision": specification.revision,
            "configuration_present": config_available,
            "weights_present": weights_available,
            "tokenizer_present": tokenizer_available,
        }
    return result


def resource_preflight(
    *,
    device: str = "cpu",
    disk_probe: str | Path | None = None,
) -> dict[str, object]:
    """Inspect software/local resources without opening data or using the network."""

    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("preflight device must be cpu or cuda")
    cuda_available = bool(torch.cuda.is_available())
    if target.type == "cuda" and cuda_available:
        try:
            index = target.index if target.index is not None else torch.cuda.current_device()
            requested = {
                "type": "cuda",
                "status": "available",
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
        except (AssertionError, RuntimeError):
            requested = {"type": "cuda", "status": "unavailable"}
    elif target.type == "cuda":
        requested = {"type": "cuda", "status": "unavailable"}
    else:
        requested = {"type": "cpu", "status": "available"}
    probe = Path.cwd() if disk_probe is None else Path(disk_probe)
    try:
        usage = shutil.disk_usage(probe)
        disk = {
            "status": "available",
            "free_bytes": int(usage.free),
            "total_bytes": int(usage.total),
        }
    except OSError:
        disk = {"status": "unavailable", "free_bytes": None, "total_bytes": None}
    return {
        "schema_version": "view-risk-p4c-resource-preflight/v1",
        "mode": "software_resource_preflight",
        "network_attempted": False,
        "downloads_or_installs_attempted": False,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "supported_by_package": sys.version_info[:2] in ((3, 10), (3, 11)),
        },
        "cpu": {
            "logical_count": os.cpu_count(),
            "architecture": platform.machine() or "unavailable",
            "processor": platform.processor() or "unavailable",
        },
        "torch": {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda or "unavailable",
            "cuda_available": cuda_available,
            "cuda_status": "available_not_selected" if cuda_available else "unavailable",
        },
        "requested_device": requested,
        "dependencies": _package_versions(),
        "disk": disk,
        "pinned_public_resources": _public_resource_status(),
        "private_data": "unavailable_not_audited",
        "patient_mapping": "unavailable_not_audited",
        "locked_outcomes_opened": False,
        "real_patient_readiness": "unavailable_not_audited",
        "scope_limit": "P5 owns actual inventory, public-weight loading, and training resources",
    }


class _SmokeVision(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.anchor = nn.Parameter(torch.tensor(1.0))

    def forward(self, *, pixel_values: Tensor) -> SimpleNamespace:
        # Recover the first pre-normalization channel and summarize four fixed
        # spatial regions.  This is an injected software model, not a CLIP proxy.
        raw = pixel_values[:, 0] * LEGACY_IMAGE_STD[0] + LEGACY_IMAGE_MEAN[0]
        height, width = raw.shape[-2:]
        half_h, half_w = height // 2, width // 2
        quadrants = torch.stack(
            (
                raw[:, :half_h, :half_w].mean(dim=(1, 2)),
                raw[:, :half_h, half_w:].mean(dim=(1, 2)),
                raw[:, half_h:, :half_w].mean(dim=(1, 2)),
                raw[:, half_h:, half_w:].mean(dim=(1, 2)),
            ),
            dim=1,
        )
        pooled = F.pad(quadrants * self.anchor, (0, self.hidden_size - 4))
        hidden = torch.stack((pooled, pooled * 0.75 + 0.01, pooled * 0.5 + 0.02), dim=1)
        return SimpleNamespace(pooler_output=pooled, last_hidden_state=hidden)


class _SmokeCLIP(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.vision_model = _SmokeVision(hidden_size)
        self.visual_projection = nn.Linear(hidden_size, 4, bias=False)
        self.text_embedding = nn.Parameter(torch.eye(4))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(6.0)))
        with torch.no_grad():
            self.visual_projection.weight.zero_()
            self.visual_projection.weight[:, :4].copy_(torch.eye(4))

    def get_text_features(self, *, input_ids: Tensor) -> Tensor:
        if input_ids.shape[0] != 4:
            raise ValueError("smoke text input must contain four prompt rows")
        return self.text_embedding

    def get_image_features(self, *, pixel_values: Tensor) -> Tensor:
        vision = self.vision_model(pixel_values=pixel_values)
        return self.visual_projection(vision.pooler_output)


def _role_manifest(role: Role, count: int) -> RoleManifest:
    # Production artifact validation intentionally keeps the accepted dataset
    # namespace even though provenance/readiness remain explicitly synthetic.
    namespace = "RSNA"
    prefix = {
        Role.CLASSIFIER_FIT: "cf",
        Role.CONFIDENCE_FIT: "co",
        Role.TUNE: "tu",
        Role.PILOT: "pi",
    }[role]
    records = []
    for index in range(count):
        record_key = f"s-{prefix}-{index:02d}"
        records.append(
            PrivateExamRecord(
                dataset_namespace=namespace,
                exam_key=record_key,
                patient_key=f"g-{prefix}-{index:02d}",
                density=index % 4,
                source_manifest="generated-fixture",
                views={
                    view: ViewReference(
                        image_id=f"i-{prefix}-{index:02d}-{view}",
                        path=f"generated/{prefix}/{index:02d}/{view}.tensor",
                    )
                    for view in CANONICAL_VIEWS
                },
                role=role,
            )
        )
    return RoleManifest(
        dataset_namespace=namespace,
        source_hashes={"generated-fixture": "1" * 64},
        patient_mapping=PatientMappingDeclaration(
            namespace,
            "2" * 64,
            "deterministic synthetic grouping",
            "P4C software smoke generator",
        ),
        records=records,
    )


def _record_number(record: PrivateExamRecord) -> int:
    return int(record.exam_key.rsplit("-", 1)[-1])


def _synthetic_images(record: PrivateExamRecord) -> dict[str, Tensor]:
    group = 0 if _record_number(record) < 2 else 2
    assignments = (group, group, group + 1, group + 1)
    result: dict[str, Tensor] = {}
    resolution = BACKBONES["vit_b_32"].image_size
    half = resolution // 2
    for view, class_index in zip(CANONICAL_VIEWS, assignments):
        image = torch.full((3, resolution, resolution), 0.05, dtype=torch.float32)
        row = 0 if class_index < 2 else half
        column = 0 if class_index % 2 == 0 else half
        image[:, row : row + half, column : column + half] = 0.95
        result[view] = image
    return result


def _parent_to_device(parent: RealizedParent, device: torch.device) -> RealizedParent:
    if device.type == "cpu":
        return parent
    return RealizedParent(
        MappingProxyType({view: image.to(device) for view, image in parent.images.items()}),
        parent.metadata,
    )


def _realized_parent(
    record: PrivateExamRecord,
    observed: tuple[str, ...],
    *,
    device: torch.device,
    perturbations: Mapping[str, PerturbationSpec] | None = None,
    common_mode: PerturbationSpec | None = None,
) -> RealizedParent:
    parent = realize_parent(
        _synthetic_images(record),
        observed,
        perturbations=perturbations,
        common_mode=common_mode,
    )
    return _parent_to_device(parent, device)


def _make_encoder(
    workspace: Path,
    *,
    fusion_pairs: FusionPairs,
    name: str,
    device: torch.device,
) -> tuple[VerifiedFrozenEncoder, Path]:
    torch.manual_seed(7)
    clip = _SmokeCLIP(BACKBONES["vit_b_32"].hidden_size)
    classifier = MultiViewCLIPClassifier(clip, CANONICAL_VIEWS, fusion_pairs).to(device)
    checkpoint = workspace / "classifier" / f"{name}.safetensors"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_tensor_checkpoint(classifier, checkpoint)
    input_ids = torch.arange(8, dtype=torch.long, device=device).reshape(4, 2)
    return (
        load_verified_frozen_encoder(
            classifier,
            input_ids,
            checkpoint,
            backbone="vit_b_32",
            prompts=PROMPTS,
        ),
        checkpoint,
    )


def _build_bundle(
    manifest: RoleManifest,
    record: PrivateExamRecord,
    *,
    encoder: VerifiedFrozenEncoder,
    operation: Operation,
    role: Role,
    sample_suffix: str,
    parent_factory: Callable[[], RealizedParent],
) -> CacheBundle:
    return build_exam_cache_with_role_access(
        manifest,
        operation=operation,
        role=role,
        exam_key=record.exam_key,
        sample_key=f"software-smoke/{sample_suffix}",
        parent_loader=lambda _record: parent_factory(),
        encoder=encoder,
        implementation_revision=_SMOKE_IMPLEMENTATION,
    )


def _stack_ds_rows(bundles: Sequence[CacheBundle]) -> DSControlRows:
    features = [
        build_ds_features(
            bundle.features.logits_by_view,
            bundle.features.observed_views,
            fusion_pairs=bundle.features.fusion_pairs,
        )
        for bundle in bundles
    ]
    pairs = features[0].fusion_pairs
    if any(item.fusion_pairs != pairs for item in features):
        raise RuntimeError("synthetic DS rows mixed fusion trees")
    stacked = DSBaselineFeatures(
        values=torch.cat([item.values.detach().cpu() for item in features]),
        valid_mask=torch.cat([item.valid_mask.detach().cpu() for item in features]),
        observed_mask=torch.cat([item.observed_mask.detach().cpu() for item in features]),
        conflict_valid_mask=torch.cat(
            [item.conflict_valid_mask.detach().cpu() for item in features]
        ),
        fusion_pairs=pairs,
    )
    return DSControlRows(
        exam_keys=tuple(bundle.provenance.exam_keys[0] for bundle in bundles),
        features=stacked,
        classifier_prediction=torch.cat(
            [bundle.features.prediction.detach().cpu() for bundle in bundles]
        ).long(),
    )


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(_canonical_json(value) + b"\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fit_and_reload_method(
    workspace: Path,
    *,
    method: str,
    config: ResearchRunConfig,
    classifier: ClassifierProvenance,
    manifest: RoleManifest,
    manifest_path: Path,
    input_evidence_path: Path,
    encoder: VerifiedFrozenEncoder,
    training_bundles: tuple[CacheBundle, ...],
    example: CacheBundle,
    device: torch.device,
) -> tuple[nn.Module, dict[str, object]]:
    torch.manual_seed(42)
    model = build_learned_method(
        method,
        config.backbone,
        RSNA_FUSION_PAIRS,
        visual_width=int(
            example.features.projected_by_view[example.features.observed_views[0]].shape[1]
        ),
        text_embeddings=example.features.normalized_text_embeddings,
    ).to(device)
    binding = TrainingBinding(
        protocol_sha256=config.protocol_sha256,
        config_sha256=config.sha256,
        search_table_sha256=config.search_table.sha256,
        manifest_sha256=manifest.manifest_sha256,
        classifier_checkpoint_sha256=classifier.checkpoint_sha256,
        method=method,
        seed=42,
    )
    expected_schedule = build_method_training_schedule(
        manifest.records, epoch=1, seed=42, method=method
    )

    def batches(
        _records: tuple[PrivateExamRecord, ...],
        schedule,
        batch_size: int,
        epoch: int,
    ):
        if epoch != 1 or tuple(schedule) != expected_schedule:
            raise RuntimeError("software smoke training schedule changed")
        if len(training_bundles) > batch_size:
            raise RuntimeError("software smoke batch exceeded the frozen public batch size")
        yield ConfidenceFitBatch(training_bundles)

    checkpoint_directory = workspace / "learned" / method
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    epoch_paths: dict[int, Path] = {}

    def save_epoch(epoch: int, current: nn.Module) -> None:
        path = checkpoint_directory / f"epoch-{epoch}.safetensors"
        save_tensor_checkpoint(current, path)
        epoch_paths[epoch] = path

    result = fit_confidence_method_with_role_access(
        manifest=manifest,
        model=model,
        method=method,
        classifier=classifier,
        config=config,
        seed=42,
        binding=binding,
        batch_loader=batches,
        frozen_encoder=encoder,
        encoder_identity=encoder.identity,
        checkpoint_path=checkpoint_directory / "resume.pt",
        stop_after_epoch=1,
        epoch_callback=save_epoch,
    )
    artifact_path = checkpoint_directory / "fit-artifact.json"
    artifact = save_confidence_fit_artifact(
        artifact_path,
        config=config,
        classifier=classifier,
        classifier_artifact_path=None,
        manifest=manifest,
        manifest_path=manifest_path,
        input_evidence_path=input_evidence_path,
        binding=binding,
        result=result,
        checkpoint_paths=epoch_paths,
    )
    reloaded_artifact = load_confidence_fit_artifact(
        artifact_path, expected_config=config
    )
    if artifact.sha256 != reloaded_artifact.sha256:
        raise RuntimeError("learned fit artifact did not reload exactly")
    reloaded = load_learned_confidence_checkpoint(
        method=method,
        seed=42,
        checkpoint_path=epoch_paths[1],
        config=config,
        example=example,
        device=device,
    )
    return reloaded, {
        "completed_epochs": result.completed_epoch,
        "configured_epochs": config.epochs,
        "optimizer_updates": result.update_count,
        "exposed_synthetic_records": result.exposed_record_count,
        "parameter_count": result.parameter_count,
        "software_only": result.software_only,
        "checkpoint_reloaded": True,
        "artifact_reloaded": True,
        "partial_fit_only": True,
        "model_state_sha256": result.model_state_sha256,
    }


def _rounded(values: Tensor) -> list[float]:
    return [round(float(value), 8) for value in values.detach().cpu().reshape(-1)]


def _mask_name(mask: tuple[str, ...]) -> str:
    return "+".join(mask)


def _run_smoke_workspace(
    workspace: Path,
    *,
    device: str,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA smoke was requested but CUDA is unavailable")
    torch.manual_seed(2026)
    np.random.seed(2026)

    classifier_manifest = _role_manifest(Role.CLASSIFIER_FIT, 2)
    confidence_manifest = _role_manifest(Role.CONFIDENCE_FIT, 4)
    tune_manifest = _role_manifest(Role.TUNE, 4)
    pilot_manifest = _role_manifest(Role.PILOT, 2)
    readiness = audit_software_fixture(
        (classifier_manifest, confidence_manifest, tune_manifest, pilot_manifest)
    )
    if readiness.patient_ready:
        raise RuntimeError("synthetic software audit unexpectedly promoted patient readiness")

    private_directory = workspace / "private-synthetic"
    private_directory.mkdir(parents=True, exist_ok=True)
    confidence_manifest_path = private_directory / "confidence-manifest.json"
    save_private_manifest(
        confidence_manifest,
        confidence_manifest_path,
        private_root=private_directory,
    )
    input_evidence_path = private_directory / "training-cache-evidence.json"
    _write_json(
        input_evidence_path,
        {
            "schema_version": "view-risk-p4c-software-input-evidence/v1",
            "software_only": True,
            "real_patient_readiness": "unavailable_not_audited",
        },
    )

    rsna_encoder, rsna_checkpoint = _make_encoder(
        workspace,
        fusion_pairs=RSNA_FUSION_PAIRS,
        name="rsna-tree",
        device=target_device,
    )
    ddsm_encoder, _ddsm_checkpoint = _make_encoder(
        workspace,
        fusion_pairs=DDSM_FUSION_PAIRS,
        name="ddsm-tree",
        device=target_device,
    )
    initialization = ClassifierProvenance.synthetic_injected(
        "vit_b_32", sha256_file(rsna_checkpoint)
    )
    classifier = ClassifierProvenance.fresh_selected(
        initialization,
        checkpoint_sha256=rsna_encoder.identity.checkpoint_sha256,
        classifier_fit_manifest_sha256=classifier_manifest.manifest_sha256,
        classifier_fit_update_count=1,
        tune_selection_sha256="3" * 64,
        tune_manifest_sha256=tune_manifest.manifest_sha256,
        readiness=readiness,
    )
    if classifier.pilot_eligible or classifier.patient_readiness_verified:
        raise RuntimeError("synthetic classifier provenance became pilot eligible")
    config = ResearchRunConfig.default("RSNA", "vit_b_32")

    all_masks = (*enumerate_proper_masks(), _FULL_MASK)
    clean_by_tree: dict[str, dict[tuple[str, ...], list[CacheBundle]]] = {
        "RSNA": {},
        "DDSM": {},
    }
    for tree, encoder, records in (
        ("RSNA", rsna_encoder, pilot_manifest.records),
        ("DDSM", ddsm_encoder, pilot_manifest.records),
    ):
        for mask in all_masks:
            bundles = []
            for record in records:
                bundles.append(
                    _build_bundle(
                        pilot_manifest,
                        record,
                        encoder=encoder,
                        operation=Operation.PILOT_EVALUATION,
                        role=Role.PILOT,
                        sample_suffix=f"{tree}/clean/{record.exam_key}/{_mask_name(mask)}",
                        parent_factory=lambda record=record, mask=mask: _realized_parent(
                            record, mask, device=target_device
                        ),
                    )
                )
            clean_by_tree[tree][mask] = bundles

    training_schedule = build_method_training_schedule(
        confidence_manifest.records, epoch=1, seed=42, method="candidate"
    )
    training_bundles = []
    record_by_key = {record.exam_key: record for record in confidence_manifest.records}
    for scheduled in training_schedule:
        record = record_by_key[scheduled.exam_key]

        def training_parent(record=record, draw=scheduled.draw):
            parent = realize_training_parent(
                _synthetic_images(record), draw, operation="confidence-fit"
            )
            return _parent_to_device(parent, target_device)

        training_bundles.append(
            _build_bundle(
                confidence_manifest,
                record,
                encoder=rsna_encoder,
                operation=Operation.CONFIDENCE_FITTING,
                role=Role.CONFIDENCE_FIT,
                sample_suffix=f"fit/{record.exam_key}",
                parent_factory=training_parent,
            )
        )
    training_tuple = tuple(training_bundles)
    candidate, candidate_fit = _fit_and_reload_method(
        workspace,
        method="candidate",
        config=config,
        classifier=classifier,
        manifest=confidence_manifest,
        manifest_path=confidence_manifest_path,
        input_evidence_path=input_evidence_path,
        encoder=rsna_encoder,
        training_bundles=training_tuple,
        example=training_tuple[0],
        device=target_device,
    )
    same_input, same_input_fit = _fit_and_reload_method(
        workspace,
        method="same_input_mlp",
        config=config,
        classifier=classifier,
        manifest=confidence_manifest,
        manifest_path=confidence_manifest_path,
        input_evidence_path=input_evidence_path,
        encoder=rsna_encoder,
        training_bundles=training_tuple,
        example=training_tuple[0],
        device=target_device,
    )
    candidate.eval()
    same_input.eval()

    # Persist and reload one real generated cache through the role-authorized API.
    cache_directory = workspace / "cache"
    cache_directory.mkdir(parents=True, exist_ok=True)
    cache_paths = save_cache_bundle(training_tuple[0], cache_directory / "sample.json")
    reloaded_cache = load_cache_bundle(
        cache_paths.metadata,
        expected_provenance=training_tuple[0].provenance,
        manifest=confidence_manifest,
        operation=Operation.CONFIDENCE_FITTING,
        role=Role.CONFIDENCE_FIT,
    )

    def full_role_bundles(manifest: RoleManifest, operation: Operation, role: Role):
        result = []
        for record in manifest.records:
            result.append(
                _build_bundle(
                    manifest,
                    record,
                    encoder=rsna_encoder,
                    operation=operation,
                    role=role,
                    sample_suffix=f"control/{role.value}/{record.exam_key}",
                    parent_factory=lambda record=record: _realized_parent(
                        record, _FULL_MASK, device=target_device
                    ),
                )
            )
        return tuple(result)

    confidence_control_bundles = full_role_bundles(
        confidence_manifest, Operation.CONFIDENCE_FITTING, Role.CONFIDENCE_FIT
    )
    tune_control_bundles = full_role_bundles(
        tune_manifest, Operation.TUNE_SELECTION, Role.TUNE
    )
    control_directory = workspace / "controls"
    control_directory.mkdir(parents=True, exist_ok=True)
    msp_artifact = create_raw_control_artifact(
        control_directory / "msp.json",
        method="msp",
        config=config,
        classifier=classifier,
        seed=42,
    )
    msp_artifact = load_control_artifact(msp_artifact.path)
    ds_artifact = fit_ds_control_artifact_with_role_access(
        control_directory / "ds-logistic.json",
        config=config,
        classifier=classifier,
        seed=42,
        confidence_manifest=confidence_manifest,
        tune_manifest=tune_manifest,
        confidence_reader=lambda records: _stack_ds_rows(
            tuple(
                confidence_control_bundles[index]
                for index, record in enumerate(records)
                if confidence_control_bundles[index].provenance.exam_keys == (
                    record.exam_key,
                )
            )
        ),
        tune_reader=lambda records: _stack_ds_rows(
            tuple(
                tune_control_bundles[index]
                for index, record in enumerate(records)
                if tune_control_bundles[index].provenance.exam_keys == (record.exam_key,)
            )
        ),
    )
    ds_artifact = load_control_artifact(ds_artifact.path)

    representative_specs = {
        "permitted_single": (
            {"L_CC": PerturbationSpec.for_sample(
                "gaussian_noise",
                "mild",
                private_sample_key="software-smoke-permitted",
                cell="representative/permitted/L_CC",
                seed=4242,
            )},
            None,
        ),
        "held_out_single": (
            {"R_CC": PerturbationSpec.for_sample(
                "contrast",
                "moderate",
                private_sample_key="software-smoke-held-out",
                cell="representative/held-out/R_CC",
                seed=4242,
            )},
            None,
        ),
        "common_mode": (
            None,
            PerturbationSpec.for_sample(
                "gaussian_blur",
                "mild",
                private_sample_key="software-smoke-common",
                cell="representative/common/all",
                seed=4242,
            ),
        ),
    }
    stressed_by_tree: dict[str, dict[str, tuple[CacheBundle, ...]]] = {}
    for tree, encoder in (("RSNA", rsna_encoder), ("DDSM", ddsm_encoder)):
        stressed_by_tree[tree] = {}
        for label, (per_view, common) in representative_specs.items():
            bundles = []
            for record in pilot_manifest.records:
                bundles.append(
                    _build_bundle(
                        pilot_manifest,
                        record,
                        encoder=encoder,
                        operation=Operation.PILOT_EVALUATION,
                        role=Role.PILOT,
                        sample_suffix=f"{tree}/stress/{label}/{record.exam_key}",
                        parent_factory=(
                            lambda record=record, per_view=per_view, common=common: (
                                _realized_parent(
                                    record,
                                    _FULL_MASK,
                                    device=target_device,
                                    perturbations=per_view,
                                    common_mode=common,
                                )
                            )
                        ),
                    )
                )
            stressed_by_tree[tree][label] = tuple(bundles)

    panels_by_tree: dict[str, dict[str, tuple[CacheBundle, ...]]] = {}
    for tree in ("RSNA", "DDSM"):
        panels_by_tree[tree] = {
            "clean_four_view": tuple(clean_by_tree[tree][_FULL_MASK]),
            "clean_masks": tuple(
                bundle
                for mask in enumerate_proper_masks()
                for bundle in clean_by_tree[tree][mask]
            ),
            **stressed_by_tree[tree],
        }

    methods = {
        "candidate": {
            "confidence_kind": "probability",
            "training_scope": {
                "mode": "one_epoch_partial_synthetic_fit_reload",
                "fusion_tree": "RSNA",
                "role": Role.CONFIDENCE_FIT.value,
                "record_count": len(confidence_manifest.records),
            },
        },
        "same_input_mlp": {
            "confidence_kind": "probability",
            "training_scope": {
                "mode": "one_epoch_partial_synthetic_fit_reload",
                "fusion_tree": "RSNA",
                "role": Role.CONFIDENCE_FIT.value,
                "record_count": len(confidence_manifest.records),
            },
        },
        "msp": {
            "confidence_kind": "ranking",
            "training_scope": {
                "mode": "no_fit_raw_scalar_artifact_reload",
                "fusion_tree": "not_applicable",
                "record_count": 0,
            },
        },
        "ds_logistic": {
            "confidence_kind": "probability",
            "training_scope": {
                "mode": "synthetic_role_authorized_control_fit_and_reload",
                "fusion_tree": "RSNA",
                "confidence_fit_record_count": len(confidence_manifest.records),
                "tune_record_count": len(tune_manifest.records),
            },
        },
    }

    def score_method(
        method: str, bundle: CacheBundle
    ) -> tuple[Tensor, str, bool]:
        if method == "candidate":
            values = confidence_bundle_scores(candidate, method, bundle)
            return values, "probability", values.shape == bundle.features.prediction.shape
        if method == "same_input_mlp":
            values = confidence_bundle_scores(same_input, method, bundle)
            return values, "probability", values.shape == bundle.features.prediction.shape
        if method == "msp":
            values, kind = analytic_control_confidence(
                msp_artifact, scores=bundle.features.scores
            )
            fixed = bool(
                values.shape == bundle.features.prediction.shape
                and torch.equal(
                    bundle.features.scores.argmax(1), bundle.features.prediction
                )
            )
            return values, kind, fixed
        if method == "ds_logistic":
            ds_features = build_ds_features(
                bundle.features.logits_by_view,
                bundle.features.observed_views,
                fusion_pairs=bundle.features.fusion_pairs,
            )
            values, kind = analytic_control_confidence(
                ds_artifact, ds_features=ds_features
            )
            return values, kind, values.shape == bundle.features.prediction.shape
        raise RuntimeError(f"unsupported smoke method: {method}")

    effects = {-1: 0, 0: 0, 1: 0}
    predictions_fixed = True
    targets_regenerated = True
    singleton_invalid = True
    full_example = clean_by_tree["RSNA"][_FULL_MASK][0]
    full_ds = build_ds_features(
        full_example.features.logits_by_view,
        full_example.features.observed_views,
        fusion_pairs=full_example.features.fusion_pairs,
    )
    record_index_by_key = {
        record.exam_key: index for index, record in enumerate(pilot_manifest.records)
    }
    coverage_matrix: dict[str, dict[str, object]] = {
        method: {
            "training_scope": details["training_scope"],
            "scoring_scope": (
                "two synthetic fusion-tree configurations; all clean masks and "
                "three representative stress panels"
            ),
            "scoring": {"RSNA": {}, "DDSM": {}},
        }
        for method, details in methods.items()
    }
    representative_confidence: dict[str, list[float]] = {}

    for tree, panels in panels_by_tree.items():
        for panel, bundles in panels.items():
            correct: list[int] = []
            predictions: list[int] = []
            targets: list[int] = []
            effect_targets: list[np.ndarray] = []
            effect_probabilities: list[np.ndarray] = []
            effect_valid: list[np.ndarray] = []
            checked_target_count = 0
            panel_is_stressed = panel not in ("clean_four_view", "clean_masks")

            for bundle in bundles:
                regenerated = checked_intervention_targets(bundle.features, bundle.targets)
                target_matches = bool(
                    torch.equal(
                        regenerated.observed_prediction,
                        bundle.targets.observed_prediction,
                    )
                    and torch.equal(
                        regenerated.omission_effects, bundle.targets.omission_effects
                    )
                )
                targets_regenerated &= target_matches
                checked_target_count += bundle.features.batch_size if target_matches else 0
                predictions_fixed &= bool(
                    torch.equal(
                        bundle.targets.observed_prediction,
                        bundle.features.prediction,
                    )
                )
                valid = bundle.targets.valid_removal_mask
                valid_rows = valid.unsqueeze(0).expand(bundle.features.batch_size, -1)
                for value in (
                    bundle.targets.omission_effects[valid_rows].detach().cpu().tolist()
                ):
                    effects[int(value)] += 1
                if len(bundle.features.observed_views) == 1:
                    singleton_invalid &= not bool(valid.any())

                with torch.no_grad():
                    head_output = candidate(
                        prepare_raw_head_inputs(bundle.features, backbone="vit_b_32")
                    )
                predictions_fixed &= bool(
                    torch.equal(
                        head_output.classifier_prediction, bundle.features.prediction
                    )
                )
                effect_targets.append(
                    bundle.targets.omission_effects.detach().cpu().numpy()
                )
                effect_probabilities.append(
                    head_output.reported_auxiliary_probabilities.detach().cpu().numpy()
                )
                effect_valid.append(valid_rows.detach().cpu().numpy())
                correct.extend(
                    (bundle.features.prediction == bundle.targets.labels)
                    .long()
                    .cpu()
                    .tolist()
                )
                predictions.extend(bundle.features.prediction.detach().cpu().tolist())
                targets.extend(bundle.targets.labels.detach().cpu().tolist())

            panel_effect_metrics = effect_metrics(
                np.concatenate(effect_targets),
                np.concatenate(effect_probabilities),
                np.concatenate(effect_valid),
                stressed=np.full(
                    len(correct), panel_is_stressed, dtype=np.bool_
                ),
            )

            for method, details in methods.items():
                confidence: list[float] = []
                prediction_consistent_count = 0
                observed_kind: str | None = None
                for bundle in bundles:
                    values, kind, fixed = score_method(method, bundle)
                    confidence.extend(values.detach().cpu().tolist())
                    prediction_consistent_count += (
                        bundle.features.batch_size if fixed else 0
                    )
                    predictions_fixed &= fixed
                    if observed_kind is None:
                        observed_kind = kind
                    elif observed_kind != kind:
                        raise RuntimeError("control confidence kind changed within a panel")
                expected_kind = details["confidence_kind"]
                if observed_kind != expected_kind:
                    raise RuntimeError(
                        f"{method} produced {observed_kind!r}, expected {expected_kind!r}"
                    )
                metrics = confidence_panel_metrics(
                    correct, confidence, confidence_kind=observed_kind
                )
                panel_report: dict[str, object] = {
                    "sample_count": len(confidence),
                    "checked_target_count": checked_target_count,
                    "prediction_consistent_count": prediction_consistent_count,
                    "confidence_kind": observed_kind,
                    "confidence_metrics": asdict(metrics),
                    "effect_metrics": (
                        json.loads(_canonical_json(asdict(panel_effect_metrics)))
                        if method == "candidate"
                        else "not_applicable_no_effect_output"
                    ),
                    "scored_input_sha256": _json_sha256(
                        {
                            "confidence": [round(value, 8) for value in confidence],
                            "correct": correct,
                            "prediction": predictions,
                            "target": targets,
                            "tree": tree,
                            "panel": panel,
                        }
                    ),
                }
                if panel == "clean_masks":
                    clean_mask_rows = []
                    for bundle, value in zip(bundles, confidence):
                        record_index = record_index_by_key[
                            bundle.provenance.exam_keys[0]
                        ]
                        clean_mask_rows.append(
                            EvaluationPrediction(
                                dataset=f"SYNTHETIC_{tree}",
                                role=Role.PILOT.value,
                                cohort=f"software-smoke-{tree.lower()}",
                                method=method,
                                training_seed=42,
                                patient_id=f"group-{record_index}",
                                exam_id=f"sample-{record_index}",
                                panel="clean_masks",
                                target=int(bundle.targets.labels[0]),
                                prediction=int(bundle.features.prediction[0]),
                                confidence=float(value),
                                confidence_kind=observed_kind,
                                mask=bundle.features.observed_views,
                                realization_id="synthetic-clean-v1",
                            )
                        )
                    clean_mask_panel = evaluate_aurc_panel(clean_mask_rows)
                    panel_report["aurc_panel"] = {
                        "cell_count": clean_mask_panel.n_cells,
                        "exam_count": clean_mask_panel.n_exams,
                        "mean_aurc": clean_mask_panel.mean_aurc,
                    }
                coverage_matrix[method]["scoring"][tree][panel] = panel_report  # type: ignore[index]
                if tree == "RSNA" and panel == "clean_four_view":
                    representative_confidence[method] = [round(confidence[0], 8)]

    # Timings are deliberately excluded from the scientific reproducibility hash.
    timing_parent = _realized_parent(
        pilot_manifest.records[0], _FULL_MASK, device=target_device
    )

    def encoder_work() -> object:
        return rsna_encoder.extract_realized_parent(timing_parent)

    def cache_io_work() -> object:
        return load_cache_bundle(
            cache_paths.metadata,
            expected_provenance=reloaded_cache.provenance,
            manifest=confidence_manifest,
            operation=Operation.CONFIDENCE_FITTING,
            role=Role.CONFIDENCE_FIT,
        )

    def candidate_work() -> object:
        return confidence_bundle_scores(candidate, "candidate", full_example)

    def scalar_work() -> object:
        return analytic_control_confidence(
            msp_artifact, scores=full_example.features.scores
        )

    def ds_work() -> object:
        return analytic_control_confidence(ds_artifact, ds_features=full_ds)

    def end_to_end_work() -> object:
        current = _build_bundle(
            pilot_manifest,
            pilot_manifest.records[0],
            encoder=rsna_encoder,
            operation=Operation.PILOT_EVALUATION,
            role=Role.PILOT,
            sample_suffix="timing/end-to-end",
            parent_factory=lambda: _realized_parent(
                pilot_manifest.records[0], _FULL_MASK, device=target_device
            ),
        )
        return confidence_bundle_scores(candidate, "candidate", current)

    measurements = [
        measure_repeated_cost(
            "encoder_feature_validation",
            encoder_work,
            device=target_device,
            warmup=warmup,
            repeats=repeats,
        ),
        measure_repeated_cost(
            "cache_io_reload",
            cache_io_work,
            device=target_device,
            warmup=warmup,
            repeats=repeats,
        ),
        measure_repeated_cost(
            "candidate_head_inference",
            candidate_work,
            device=target_device,
            warmup=warmup,
            repeats=repeats,
        ),
        measure_repeated_cost(
            "scalar_control_inference",
            scalar_work,
            device=target_device,
            warmup=warmup,
            repeats=repeats,
        ),
        measure_repeated_cost(
            "ds_control_inference",
            ds_work,
            device=target_device,
            warmup=warmup,
            repeats=repeats,
        ),
        measure_repeated_cost(
            "synthetic_end_to_end",
            end_to_end_work,
            device=target_device,
            warmup=warmup,
            repeats=repeats,
        ),
    ]
    cache_bytes = cache_paths.metadata.stat().st_size + cache_paths.tensors.stat().st_size

    scientific_outputs = {
        "clean_predictions_by_tree": {
            tree: {
                _mask_name(mask): [
                    int(bundle.features.prediction.cpu()[0])
                    for bundle in clean_by_tree[tree][mask]
                ]
                for mask in all_masks
            }
            for tree in ("RSNA", "DDSM")
        },
        "signed_effect_support": {
            "-1": effects[-1],
            "0": effects[0],
            "+1": effects[1],
        },
        "representative_confidence": {
            method: values for method, values in representative_confidence.items()
        },
        "coverage_matrix": coverage_matrix,
        "learned_state_sha256": {
            "candidate": candidate_fit["model_state_sha256"],
            "same_input_mlp": same_input_fit["model_state_sha256"],
        },
    }
    methods_exercised = tuple(coverage_matrix)
    report: dict[str, object] = {
        "schema_version": SMOKE_REPORT_VERSION,
        "mode": "synthetic_software_only",
        "status": "software_smoke_passed",
        "network_attempted": False,
        "downloads_or_installs_attempted": False,
        "patient_data_opened": False,
        "locked_outcomes_opened": False,
        "coverage": {
            "nonempty_mask_count": len(all_masks),
            "proper_mask_count": len(enumerate_proper_masks()),
            "singleton_mask_count": sum(len(mask) == 1 for mask in all_masks),
            "fusion_trees_exercised": list(panels_by_tree),
            "signed_effect_support": scientific_outputs["signed_effect_support"],
            "stress_cases_exercised": [
                "gaussian_noise/mild/single/L_CC (confidence-fit-permitted)",
                "contrast/moderate/single/R_CC (held-out-family representative)",
                "gaussian_blur/mild/common/all-observed (common-mode representative)",
            ],
        },
        "methods": {
            "exercised": list(methods_exercised),
            "not_exercised": [
                method for method in MANDATORY_METHODS if method not in methods_exercised
            ],
            "scope": "representative controls only; this is not the mandatory-control study",
            "training_and_scoring_scope": {
                method: {
                    "training_scope": coverage_matrix[method]["training_scope"],
                    "scoring_scope": coverage_matrix[method]["scoring_scope"],
                }
                for method in methods_exercised
            },
        },
        "panels": {
            "exercised": [
                "clean_four_view",
                "all_14_clean_proper_masks",
                "one_permitted_single_view_stress",
                "one_held_out_family_single_view_stress",
                "one_common_mode_stress",
            ],
            "not_exercised": [
                "complete_48_cell_primary_panel",
                "complete_strong_seen_family_panel",
                "complete_common_mode_panel",
                "patient_paired_bootstrap",
                "multi_seed_or_multi_dataset_patient_study",
            ],
        },
        "invariants": {
            "verified_frozen_encoder_used": True,
            "classifier_predictions_fixed_across_methods": predictions_fixed,
            "targets_regenerated_from_realized_parent": targets_regenerated,
            "singleton_removal_slots_invalid": singleton_invalid,
            "cache_saved_and_role_authorized_reload": True,
            "real_patient_readiness_promoted": False,
            "synthetic_classifier_fit_or_tune_performed": False,
        },
        "fit_reload": {
            "candidate": candidate_fit,
            "same_input_mlp": same_input_fit,
            "scalar_artifact_reloaded": True,
            "ds_artifact_reloaded": True,
        },
        "scientific_outputs": scientific_outputs,
        "reproducibility": {
            "definition": "exact canonical JSON hash of scientific_outputs; timings excluded",
            "scientific_sha256": _json_sha256(scientific_outputs),
            "seed": 2026,
            "training_seed": 42,
            "stress_seed": 4242,
        },
        "preflight": resource_preflight(device=str(target_device), disk_probe=workspace),
        "costs": {
            "mode": "tiny_injected_synthetic_measurement",
            "context": {
                "device": str(target_device),
                "batch_size": 1,
                "input_shape": [1, 3, 224, 224],
                "dtype": "float32",
                "python": platform.python_version(),
                "torch": torch.__version__,
                "hardware": (
                    torch.cuda.get_device_name(target_device)
                    if target_device.type == "cuda"
                    else (platform.processor() or platform.machine() or "unavailable")
                ),
            },
            "measurements": measurements,
            "parameters": {
                "candidate_trainable": {
                    "value": count_trainable_parameters(candidate),
                    "status": "measured_exact",
                },
                "same_input_mlp_trainable": {
                    "value": count_trainable_parameters(same_input),
                    "status": "measured_exact",
                },
            },
            "cache_bytes": {
                "value": cache_bytes,
                "status": "measured_exact_for_one_generated_cache_metadata_plus_tensors",
            },
            "protocol_later_measurement_procedure": {
                "warmup_calls": 20,
                "timed_calls": 100,
                "status": "not_run_by_default_cheap_smoke",
            },
            "limitations": [
                "timings are for an injected tiny CPU/CUDA software model, not CLIP",
                "no deployment performance or efficiency improvement is inferred",
                "actual-backbone latency and memory remain unmeasured and resource-dependent",
            ],
        },
        "scientific_conclusions": {
            "empirical_benefit": "unavailable",
            "pilot_go_no_go": "unavailable",
            "real_backbone_efficiency": "unavailable_unmeasured",
            "real_patient_readiness": "unavailable_not_audited",
        },
        "classifier_scope": (
            "tiny injected frozen classifier with synthetic workflow bindings; "
            "no classifier fit or tune selection was performed"
        ),
        "control_output_kinds": {"msp": "ranking", "ds_logistic": "probability"},
        "artifact_summary": {
            "artifacts_are_synthetic_private_runtime_files": True,
            "private_paths_or_identifiers_in_report": False,
            "generated_cache_file_count": 2,
        },
    }
    if not all(
        (
            predictions_fixed,
            targets_regenerated,
            singleton_invalid,
            all(value > 0 for value in effects.values()),
            len(all_masks) == 15,
            all(
                entry["sample_count"] == entry["checked_target_count"]
                == entry["prediction_consistent_count"]
                and entry["confidence_metrics"]["n_exams"]
                == entry["sample_count"]
                and (
                    panel != "clean_masks"
                    or entry["aurc_panel"]["cell_count"] == 14
                )
                for method in methods_exercised
                for tree in panels_by_tree
                for panel, entry in coverage_matrix[method]["scoring"][tree].items()
            ),
        )
    ):
        raise RuntimeError("synthetic smoke integration invariant failed")
    return report


def _validate_requested_output(output_dir: str | Path) -> tuple[Path, Path]:
    final = Path(output_dir).expanduser().resolve()
    if final.exists():
        raise FileExistsError("requested smoke output directory must not already exist")
    if not final.parent.is_dir():
        raise ValueError("requested smoke output parent directory must already exist")
    staging = final.with_name(f".{final.name}.staging-{uuid.uuid4().hex}")
    for candidate in (
        final / "report.json",
        staging / "report.json",
        staging / "cache" / "sample.json",
        staging / "classifier" / "rsna-tree.safetensors",
    ):
        _require_external_or_ignored_destination(candidate)
    return final, staging


def _publish_new_directory(staging: Path, final: Path) -> None:
    """Claim a never-existing final directory without clobbering a late creator."""

    claimed = False
    try:
        final.mkdir(mode=0o700)
        claimed = True
        for child in staging.iterdir():
            child.rename(final / child.name)
        staging.rmdir()
    except BaseException:
        if claimed:
            shutil.rmtree(final, ignore_errors=True)
        raise


def run_synthetic_smoke(
    *,
    output_dir: str | Path | None = None,
    device: str = "cpu",
    warmup: int = 2,
    repeats: int = 5,
) -> dict[str, object]:
    """Run P4C without network/model downloads and return a sanitized report."""

    # Validate timing arguments before creating any task-owned path.
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if warmup > 100 or repeats > 1000:
        raise ValueError("smoke timing calls exceed the bounded CLI limit")

    if output_dir is None:
        with tempfile.TemporaryDirectory(prefix="mmdc-view-risk-p4c-") as temporary:
            workspace = Path(temporary).resolve()
            try:
                _require_external_or_ignored_destination(workspace / "report.json")
            except ValueError:
                cache_parent = Path.cwd().resolve() / ".cache"
                cache_parent.mkdir(exist_ok=True)
                _require_external_or_ignored_destination(cache_parent / "p4c-probe")
                with tempfile.TemporaryDirectory(
                    prefix="mmdc-view-risk-p4c-", dir=cache_parent
                ) as fallback:
                    report = _run_smoke_workspace(
                        Path(fallback), device=device, warmup=warmup, repeats=repeats
                    )
            else:
                report = _run_smoke_workspace(
                    workspace, device=device, warmup=warmup, repeats=repeats
                )
        report["artifacts_retained"] = False
        report["artifact_summary"]["retention"] = "temporary_resources_cleaned"
        return report

    final, staging = _validate_requested_output(output_dir)
    staging.mkdir(mode=0o700)
    try:
        report = _run_smoke_workspace(
            staging, device=device, warmup=warmup, repeats=repeats
        )
        report["artifacts_retained"] = True
        report["artifact_summary"]["retention"] = "requested_new_output_directory"
        _write_json(staging / "report.json", report)
        _publish_new_directory(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


__all__ = [
    "SMOKE_REPORT_VERSION",
    "measure_repeated_cost",
    "resource_preflight",
    "run_synthetic_smoke",
]
