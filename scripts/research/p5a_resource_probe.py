#!/usr/bin/env python3
"""Offline, synthetic-only P5A CUDA and pinned-public-CLIP resource probe."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


# These must be set before importing Transformers or the production loader.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F

from mmdc_clip_f.research.view_risk.features import tensor_sha256
from mmdc_clip_f.research.view_risk.training import (
    ResearchRunConfig,
    load_pinned_public_clip_classifier,
    pinned_public_clip_configuration,
)


REPOSITORY_COMMIT = "9fc8eb758288e196c31f22469d734aa96bb02ee7"
CLASSIFIER_FIT_EXAMS = 3_209


def _mib(value: int) -> float:
    return round(value / (1024**2), 3)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nvidia_smi() -> dict[str, object]:
    fields = (
        "index,name,driver_version,memory.total,memory.used,memory.free,"
        "utilization.gpu,temperature.gpu"
    )
    output = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    values = [part.strip() for part in output.split(",")]
    if len(values) != 8:
        raise RuntimeError("unexpected nvidia-smi field count")
    return {
        "index": int(values[0]),
        "name": values[1],
        "driver_version": values[2],
        "memory_total_mib": int(values[3]),
        "memory_used_mib": int(values[4]),
        "memory_free_mib": int(values[5]),
        "utilization_percent": int(values[6]),
        "temperature_c": int(values[7]),
    }


def _synchronize() -> None:
    torch.cuda.synchronize()


def _make_views(batch_size: int, image_size: int) -> dict[str, torch.Tensor]:
    return {
        view: torch.randn(batch_size, 3, image_size, image_size, device="cuda")
        for view in ("L_CC", "L_MLO", "R_CC", "R_MLO")
    }


def _basic_cuda_probe() -> dict[str, object]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    started = time.perf_counter()
    left = torch.randn(2048, 2048, device="cuda", requires_grad=True)
    right = torch.randn(2048, 2048, device="cuda", requires_grad=True)
    loss = (left @ right).square().mean()
    loss.backward()
    _synchronize()
    result = {
        "operation": "float32 2048x2048 matmul, scalar loss, backward",
        "loss_finite": bool(torch.isfinite(loss).item()),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "allocated_before_mib": _mib(allocated_before),
        "allocated_after_backward_mib": _mib(torch.cuda.memory_allocated()),
        "peak_allocated_mib": _mib(torch.cuda.max_memory_allocated()),
    }
    del loss, left, right
    gc.collect()
    torch.cuda.empty_cache()
    result["allocated_after_cleanup_mib"] = _mib(torch.cuda.memory_allocated())
    return result


def _run_probe() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")

    config = ResearchRunConfig.default("RSNA", "vit_b_32")
    public = pinned_public_clip_configuration(config.backbone)
    snapshot = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--openai--clip-vit-base-patch32"
        / "snapshots"
        / public.revision
    )
    weight_file = snapshot / "pytorch_model.bin"
    required_files = (
        "config.json",
        "merges.txt",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    )
    missing = [name for name in required_files if not (snapshot / name).is_file()]
    if missing:
        raise FileNotFoundError(f"pinned offline snapshot is incomplete: {missing}")

    free_before, total_memory = torch.cuda.mem_get_info()
    evidence: dict[str, object] = {
        "schema_version": "p5a-resource-probe/v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": REPOSITORY_COMMIT,
        "constraints": {
            "offline_environment": {
                "HF_HUB_OFFLINE": os.environ["HF_HUB_OFFLINE"],
                "TRANSFORMERS_OFFLINE": os.environ["TRANSFORMERS_OFFLINE"],
                "HF_HUB_DISABLE_TELEMETRY": os.environ["HF_HUB_DISABLE_TELEMETRY"],
            },
            "input": "synthetic random tensors only",
            "optimizer_constructed": False,
            "optimizer_updates": 0,
            "patient_data_accessed": False,
            "model_downloads_requested": False,
        },
        "software": {
            "python": ".".join(map(str, __import__("sys").version_info[:3])),
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "gpu_initial": _nvidia_smi(),
        "torch_cuda": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_mib": _mib(total_memory),
            "free_memory_before_mib": _mib(free_before),
        },
        "basic_cuda_forward_backward": _basic_cuda_probe(),
        "public_source": {
            "backbone": public.backbone,
            "hf_model": public.hf_model,
            "revision": public.revision,
            "image_size": public.image_size,
            "hidden_size": public.hidden_size,
            "prompts": list(public.prompts),
            "snapshot": str(snapshot),
            "required_files_present": True,
            "weight_file_size_bytes": weight_file.stat().st_size,
            "weight_file_sha256": _sha256_file(weight_file),
        },
        "configured_schedule_observed": {
            "optimizer": config.optimizer.name,
            "learning_rate": config.optimizer.learning_rate,
            "weight_decay": config.optimizer.weight_decay,
            "batch_size": config.optimizer.batch_size,
            "epochs": config.epochs,
        },
    }

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model, input_ids, provenance = load_pinned_public_clip_classifier(
        config.backbone, config.dataset, device="cuda"
    )
    _synchronize()
    evidence["public_load"] = {
        "success": True,
        "elapsed_seconds": round(time.perf_counter() - load_started, 6),
        "peak_allocated_mib": _mib(torch.cuda.max_memory_allocated()),
        "allocated_after_load_mib": _mib(torch.cuda.memory_allocated()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in model.parameters()
        ),
        "buffer_bytes": sum(buffer.numel() * buffer.element_size() for buffer in model.buffers()),
        "module_state_sha256": provenance.checkpoint_sha256,
        "initialization_sha256": provenance.initialization_sha256,
        "public_weight_content_sha256": provenance.public_weight_content_sha256,
        "kind": provenance.kind,
    }
    evidence["tokenization"] = {
        "success": True,
        "shape": list(input_ids.shape),
        "dtype": str(input_ids.dtype),
        "device": input_ids.device.type,
        "token_ids_sha256": tensor_sha256(input_ids),
    }

    # A no-grad call establishes public-load/token/forward compatibility directly.
    model.eval()
    eval_views = _make_views(1, public.image_size)
    torch.cuda.reset_peak_memory_stats()
    _synchronize()
    eval_started = time.perf_counter()
    with torch.no_grad():
        eval_logits = model(eval_views, input_ids)
    _synchronize()
    evidence["forward_compatibility"] = {
        "success": True,
        "batch_size": 1,
        "logits_shape": list(eval_logits.shape),
        "logits_finite": bool(torch.isfinite(eval_logits).all().item()),
        "elapsed_seconds": round(time.perf_counter() - eval_started, 6),
        "peak_allocated_mib": _mib(torch.cuda.max_memory_allocated()),
    }
    del eval_logits, eval_views
    torch.cuda.empty_cache()

    # Warm up the exact float32 model path once without an optimizer or update.
    model.train()
    warmup_views = _make_views(1, public.image_size)
    model.zero_grad(set_to_none=True)
    warmup_logits = model(warmup_views, input_ids)
    F.cross_entropy(warmup_logits, torch.zeros(1, dtype=torch.long, device="cuda")).backward()
    _synchronize()
    del warmup_logits, warmup_views
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()

    batch_size = config.optimizer.batch_size
    batch_views = _make_views(batch_size, public.image_size)
    labels = torch.arange(batch_size, device="cuda", dtype=torch.long) % 4
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    allocated_start = torch.cuda.memory_allocated()
    reserved_start = torch.cuda.memory_reserved()

    _synchronize()
    forward_started = time.perf_counter()
    logits = model(batch_views, input_ids)
    loss = F.cross_entropy(logits, labels)
    _synchronize()
    forward_seconds = time.perf_counter() - forward_started

    backward_started = time.perf_counter()
    loss.backward()
    _synchronize()
    backward_seconds = time.perf_counter() - backward_started
    measured_step_seconds = forward_seconds + backward_seconds
    steps_per_epoch = math.ceil(CLASSIFIER_FIT_EXAMS / batch_size)
    evidence["configured_batch_forward_backward"] = {
        "success": True,
        "context": (
            "single synchronized float32 synthetic four-view batch after one batch-1 "
            "forward/backward warmup; no AMP, optimizer, optimizer state, or update"
        ),
        "batch_size": batch_size,
        "images_per_model_call": batch_size * 4,
        "input_shape_per_view": list(batch_views["L_CC"].shape),
        "logits_shape": list(logits.shape),
        "logits_finite": bool(torch.isfinite(logits).all().item()),
        "loss": float(loss.detach().cpu()),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "forward_seconds": round(forward_seconds, 6),
        "backward_seconds": round(backward_seconds, 6),
        "forward_backward_seconds": round(measured_step_seconds, 6),
        "allocated_start_mib": _mib(allocated_start),
        "reserved_start_mib": _mib(reserved_start),
        "allocated_after_backward_mib": _mib(torch.cuda.memory_allocated()),
        "reserved_after_backward_mib": _mib(torch.cuda.memory_reserved()),
        "peak_allocated_mib": _mib(torch.cuda.max_memory_allocated()),
        "peak_reserved_mib": _mib(torch.cuda.max_memory_reserved()),
        "optimizer_updates": 0,
    }
    evidence["runtime_extrapolation"] = {
        "is_extrapolation": True,
        "classifier_fit_exams": CLASSIFIER_FIT_EXAMS,
        "steps_per_epoch_at_batch_size_6": steps_per_epoch,
        "seconds_per_epoch": round(steps_per_epoch * measured_step_seconds, 3),
        "seconds_for_20_epochs": round(20 * steps_per_epoch * measured_step_seconds, 3),
        "excludes": [
            "image IO and decoding",
            "preprocessing and RandAugment",
            "data transfer",
            "optimizer construction/state/update",
            "checkpoint serialization",
            "tune inference and selection",
            "provenance/content hashing",
        ],
        "warning": "one synthetic batch is not a training throughput benchmark",
    }

    state_bytes = evidence["public_load"]["parameter_bytes"] + evidence["public_load"][
        "buffer_bytes"
    ]
    evidence["storage_implications"] = {
        "model_state_bytes": state_bytes,
        "estimated_20_full_state_checkpoints_bytes": 20 * state_bytes,
        "estimated_50_full_state_checkpoints_bytes": 50 * state_bytes,
        "is_estimate": True,
        "note": "safetensors metadata overhead and resume/selection artifacts are excluded",
    }

    del logits, loss, labels, batch_views, input_ids, model
    gc.collect()
    torch.cuda.empty_cache()
    _synchronize()
    free_after, _ = torch.cuda.mem_get_info()
    evidence["cleanup"] = {
        "allocated_mib": _mib(torch.cuda.memory_allocated()),
        "reserved_mib": _mib(torch.cuda.memory_reserved()),
        "free_memory_after_mib": _mib(free_after),
    }
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    evidence = _run_probe()
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
