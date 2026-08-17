"""Training, validation, and evaluation for the Stage-1 classifier."""

from __future__ import annotations

import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .backbones import PROMPTS, get_backbone
from .config import public_config, resolve_data_paths, resolve_output_root
from .data import FourViewDataset, build_transforms, validate_split_isolation
from .metrics import classification_metrics
from .model import MultiViewCLIPClassifier
from .provenance import (
    atomic_write_json,
    create_run_dir,
    environment_state,
    git_state,
    sha256_file,
    sha256_json,
)


@dataclass(frozen=True)
class Stage1Predictions:
    """Canonical, sample-index-sorted Stage-1 predictions on CPU."""

    sample_indexes: np.ndarray
    labels: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray
    logits: np.ndarray


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _device(value: str | None) -> torch.device:
    if value:
        device = torch.device(value)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, enabled=True)


def _model_settings(config: Mapping[str, Any]):
    model_config = config["model"]
    backbone = get_backbone(str(model_config["backbone"]))
    hf_model = str(model_config.get("hf_model", backbone.hf_model))
    revision = str(model_config.get("revision", backbone.revision))
    attention_implementation = str(model_config["attention_implementation"]).lower()
    prompts = tuple(model_config.get("prompts", PROMPTS))
    if len(prompts) != 4:
        raise ValueError("Exactly four density prompts are required")
    return backbone, hf_model, revision, attention_implementation, prompts


def build_model(
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[MultiViewCLIPClassifier, Tensor]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import CLIPModel, CLIPProcessor

    _backbone, hf_model, revision, attention_implementation, prompts = _model_settings(config)
    clip_model = CLIPModel.from_pretrained(
        hf_model,
        revision=revision,
        attn_implementation=attention_implementation,
    )
    resolved_attention = getattr(clip_model.config, "_attn_implementation", None)
    if resolved_attention != attention_implementation:
        raise RuntimeError(
            "Transformers did not activate the configured CLIP attention implementation: "
            f"requested={attention_implementation!r}, resolved={resolved_attention!r}"
        )
    processor = CLIPProcessor.from_pretrained(hf_model, revision=revision)
    classifier = MultiViewCLIPClassifier(
        clip_model,
        input_order=config["views"]["input_order"],
        fusion_pairs=config["views"]["fusion_pairs"],
    ).to(device)
    input_ids = processor(text=list(prompts), padding=True, return_tensors="pt").input_ids.to(
        device
    )
    return classifier, input_ids


def build_datasets(
    config: Mapping[str, Any],
    data_root: str | None = None,
    manifest_dir: str | None = None,
) -> dict[str, FourViewDataset]:
    root, manifests = resolve_data_paths(config, data_root=data_root, manifest_dir=manifest_dir)
    backbone, _hf_model, _revision, _attention_implementation, _prompts = _model_settings(config)
    train_transform, eval_transform = build_transforms(
        backbone.image_size, config["training"].get("augmentation")
    )
    name = str(config["dataset"]["name"])
    order = config["views"]["input_order"]
    return {
        split: FourViewDataset(
            name,
            manifest,
            root,
            transform=train_transform if split == "train" else eval_transform,
            input_order=order,
        )
        for split, manifest in manifests.items()
    }


def _loader(
    dataset: FourViewDataset,
    training: Mapping[str, Any],
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    workers = int(training.get("num_workers", 4))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _move_views(views: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in views.items()}


def predict_loader(
    model: nn.Module,
    input_ids: Tensor,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    amp: bool = False,
) -> Stage1Predictions:
    """Predict a complete split and return arrays in canonical sample order.

    A complete split must contain every integer sample index exactly once in
    ``0..N-1``.  Enforcing that invariant prevents confidence artifacts from
    silently joining predictions to the wrong exam.
    """
    model.eval()
    indexes: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            views = _move_views(batch["views"], device)
            target = batch["label"]
            sample_index = batch["sample_index"]
            with _autocast(device, amp):
                output = model(views, input_ids)
            if output.ndim != 2 or output.shape[1] != 4:
                raise ValueError("Stage-1 logits must have shape [batch, 4]")
            if not torch.isfinite(output).all():
                raise ValueError("Stage-1 logits contain non-finite values")
            batch_size = int(output.shape[0])
            target_array = (
                target.detach().cpu().numpy() if isinstance(target, Tensor) else np.asarray(target)
            )
            index_array = (
                sample_index.detach().cpu().numpy()
                if isinstance(sample_index, Tensor)
                else np.asarray(sample_index)
            )
            target_array = np.asarray(target_array)
            index_array = np.asarray(index_array)
            if not np.issubdtype(target_array.dtype, np.integer):
                raise ValueError("labels must be integers")
            if not np.issubdtype(index_array.dtype, np.integer):
                raise ValueError("sample_indexes must be integers")
            target_array = target_array.astype(np.int64, copy=False).reshape(-1)
            index_array = index_array.astype(np.int64, copy=False).reshape(-1)
            if target_array.size != batch_size or index_array.size != batch_size:
                raise ValueError("labels and sample_indexes must match the logits batch size")
            if np.any((target_array < 0) | (target_array >= 4)):
                raise ValueError("labels must be in 0..3")
            labels.append(target_array)
            indexes.append(index_array)
            logits.append(output.detach().float().cpu().numpy())

    if not logits:
        raise ValueError("Prediction loader is empty")
    all_indexes = np.concatenate(indexes)
    if np.unique(all_indexes).size != all_indexes.size:
        raise ValueError("Prediction loader contains duplicate sample_indexes")
    order = np.argsort(all_indexes, kind="stable")
    sorted_indexes = all_indexes[order]
    expected = np.arange(sorted_indexes.size, dtype=np.int64)
    if not np.array_equal(sorted_indexes, expected):
        raise ValueError("sample_indexes must contain every canonical index in 0..N-1")

    sorted_labels = np.concatenate(labels)[order]
    sorted_logits = np.concatenate(logits, axis=0)[order]
    shifted = sorted_logits - sorted_logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1).astype(np.int64, copy=False)
    return Stage1Predictions(
        sample_indexes=sorted_indexes,
        labels=sorted_labels,
        predictions=predictions,
        probabilities=probabilities,
        logits=sorted_logits,
    )


def evaluate_loader(
    model: nn.Module,
    input_ids: Tensor,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    collect_hidden_states: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Tensor]] | None]:
    from tqdm import tqdm

    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    sample_indexes: list[np.ndarray] = []
    total_loss = 0.0
    total_count = 0
    batch_accuracy_sum = 0.0
    num_batches = 0
    hidden_batches: list[dict[str, Tensor]] | None = [] if collect_hidden_states else None
    with torch.inference_mode():
        for batch in tqdm(loader, desc="evaluate", leave=False):
            views = _move_views(batch["views"], device)
            target = batch["label"].to(device, dtype=torch.long, non_blocking=True)
            with _autocast(device, amp):
                if collect_hidden_states:
                    output, hidden = model.forward_with_hidden_states(views, input_ids)
                else:
                    output = model(views, input_ids)
                    hidden = None
                loss = F.cross_entropy(output, target)
            probs = F.softmax(output.float(), dim=1)
            batch_size = int(target.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            batch_accuracy_sum += float((probs.argmax(1) == target).float().mean().item())
            num_batches += 1
            labels.append(target.cpu().numpy())
            probabilities.append(probs.cpu().numpy())
            sample_indexes.append(batch["sample_index"].cpu().numpy())
            if hidden_batches is not None and hidden is not None:
                hidden_batches.append(
                    {name: value.detach().cpu() for name, value in hidden.items()}
                )
    if total_count == 0:
        raise RuntimeError("Evaluation loader is empty")
    label_array = np.concatenate(labels)
    probability_array = np.concatenate(probabilities)
    metrics = classification_metrics(label_array, probability_array)
    metrics.update(
        {
            "loss": total_loss / total_count,
            "batch_mean_accuracy": batch_accuracy_sum / num_batches,
        }
    )
    arrays = {
        "labels": label_array,
        "probabilities": probability_array,
        "sample_indexes": np.concatenate(sample_indexes),
    }
    return metrics, arrays, hidden_batches


def _save_weights(model: nn.Module, path: Path) -> None:
    from safetensors.torch import save_file

    state = {
        name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()
    }
    save_file(state, str(path))


def load_weights(model: nn.Module, path: str | Path, device: torch.device) -> None:
    from safetensors.torch import load_file

    state = load_file(str(path), device=str(device))
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:  # strict=True normally raises before this point
        raise RuntimeError(f"Checkpoint mismatch; missing={missing}, unexpected={unexpected}")


def validate_data(
    config: Mapping[str, Any],
    data_root: str | None = None,
    manifest_dir: str | None = None,
    check_files: bool = True,
) -> dict[str, Any]:
    datasets = build_datasets(config, data_root=data_root, manifest_dir=manifest_dir)
    report = validate_split_isolation(datasets)
    expected = config["dataset"].get("expected_counts", {})
    mismatches = {
        split: {"expected": int(expected[split]), "actual": len(datasets[split])}
        for split in expected
        if split in datasets and int(expected[split]) != len(datasets[split])
    }
    report["count_mismatches"] = mismatches
    report["missing_file_counts"] = {}
    if check_files:
        for split, dataset in datasets.items():
            report["missing_file_counts"][split] = len(dataset.validate_files())
    report["valid"] = (
        not report["overlaps"]
        and not any(report["duplicate_group_counts"].values())
        and not mismatches
        and not any(report["missing_file_counts"].values())
    )
    return report


def train_classifier(
    config: Mapping[str, Any],
    data_root: str | None = None,
    manifest_dir: str | None = None,
    output_root: str | None = None,
    run_name: str | None = None,
    device_name: str | None = None,
) -> Path:
    from tqdm import tqdm

    training = config["training"]
    selection = config["selection"]
    seed = int(training["seed"])
    deterministic = bool(training.get("deterministic", False))
    set_seed(seed, deterministic=deterministic)
    device = _device(device_name)
    amp = bool(training.get("amp", False))
    if amp and device.type != "cuda":
        raise RuntimeError("AMP is supported only for CUDA runs in this release")

    datasets = build_datasets(config, data_root=data_root, manifest_dir=manifest_dir)
    split_report = validate_split_isolation(datasets)
    if split_report["overlaps"]:
        raise RuntimeError(f"Dataset splits overlap: {split_report['overlaps']}")
    if any(split_report["duplicate_group_counts"].values()):
        raise RuntimeError(
            f"Dataset splits contain duplicate groups: {split_report['duplicate_group_counts']}"
        )
    selection_split = str(selection.get("split", "valid"))
    if selection_split not in datasets:
        raise ValueError(f"Unknown selection split: {selection_split}")

    output = resolve_output_root(config, output_root)
    prefix = f"{str(config['dataset']['name']).lower()}-{config['model']['backbone']}"
    run_dir = create_run_dir(output, prefix=prefix, run_name=run_name)
    resolved = public_config(config)
    config_hash = sha256_json(resolved)
    atomic_write_json(run_dir / "config.json", resolved)
    atomic_write_json(run_dir / "data.json", split_report)
    atomic_write_json(
        run_dir / "run.json",
        {
            "schema_version": 1,
            "status": "running",
            "config_sha256": config_hash,
            "environment": environment_state(),
            "git": git_state(Path(__file__).resolve().parents[2]),
            "device": str(device),
        },
    )

    model, input_ids = build_model(config, device)
    optimizer_config = training.get("optimizer", {})
    if not isinstance(optimizer_config, Mapping):
        raise ValueError("training.optimizer must be a mapping")
    optimizer_name = str(optimizer_config.get("name", "adam")).lower()
    if optimizer_name != "adam":
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optimizer_config.get("learning_rate", 1e-7)),
        weight_decay=float(optimizer_config.get("weight_decay", 1e-5)),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    train_loader = _loader(datasets["train"], training, True, seed, device)
    selection_loader = _loader(datasets[selection_split], training, False, seed, device)
    metric_name = str(selection["metric"])
    mode = str(selection.get("mode", "min" if metric_name == "loss" else "max"))
    best_value = math.inf if mode == "min" else -math.inf
    best_epoch: int | None = None
    metrics_path = run_dir / "metrics.jsonl"

    try:
        for epoch in range(int(training["epochs"])):
            model.train()
            loss_sum = 0.0
            correct = 0
            count = 0
            batch_accuracy_sum = 0.0
            batches = 0
            for batch in tqdm(train_loader, desc=f"train {epoch}", leave=False):
                views = _move_views(batch["views"], device)
                target = batch["label"].to(device, dtype=torch.long, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device, amp):
                    output = model(views, input_ids)
                    loss = F.cross_entropy(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                predictions = output.detach().argmax(1)
                batch_size = int(target.shape[0])
                loss_sum += float(loss.item()) * batch_size
                batch_correct = int((predictions == target).sum().item())
                correct += batch_correct
                count += batch_size
                batch_accuracy_sum += batch_correct / batch_size
                batches += 1

            validation, _arrays, _hidden = evaluate_loader(
                model, input_ids, selection_loader, device, amp=amp
            )
            train_metrics = {
                "loss": loss_sum / count,
                "accuracy": correct / count,
                "batch_mean_accuracy": batch_accuracy_sum / batches,
            }
            current = float(validation[metric_name])
            improved = current < best_value if mode == "min" else current > best_value
            record = {
                "epoch": epoch,
                "train": train_metrics,
                selection_split: validation,
                "selection_metric": metric_name,
                "selection_value": current,
                "is_best": improved,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            if improved:
                best_value = current
                best_epoch = epoch
                checkpoint = run_dir / "best.safetensors"
                _save_weights(model, checkpoint)
                atomic_write_json(
                    run_dir / "best.json",
                    {
                        "epoch": epoch,
                        "metric": metric_name,
                        "mode": mode,
                        "value": current,
                        "selection_split": selection_split,
                        "config_sha256": config_hash,
                        "manifest_sha256": split_report["manifest_sha256"],
                        "weights_file": checkpoint.name,
                        "weights_sha256": sha256_file(checkpoint),
                    },
                )
        if best_epoch is None:
            raise RuntimeError("Training completed without a selectable checkpoint")
    except BaseException:
        run_metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        run_metadata["status"] = "failed"
        atomic_write_json(run_dir / "run.json", run_metadata)
        raise
    run_metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    run_metadata.update({"status": "complete", "best_epoch": best_epoch})
    atomic_write_json(run_dir / "run.json", run_metadata)
    return run_dir
