"""Executable, provenance-safe MV-ACN training workflow."""

from __future__ import annotations

import copy
import json
import os
import warnings
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..backbones import get_backbone
from ..classifier import (
    build_datasets,
    build_model,
    load_weights,
    predict_loader,
    seed_worker,
    set_seed,
)
from ..config import ConfigError, public_config
from ..provenance import (
    atomic_write_json,
    create_run_dir,
    environment_state,
    git_state,
    sha256_file,
    sha256_json,
)
from .artifacts import (
    TCPArtifact,
    build_tcp_artifact,
    read_tcp_artifact,
    write_tcp_artifact,
)
from .metrics import confidence_selection_metrics
from .model import MVACNConfig, MVACNHead, export_head_state_dict


_SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class VerifiedStage1:
    model: nn.Module
    input_ids: Tensor
    checkpoint_sha256: str
    best: Mapping[str, Any]


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} must be a mapping")
    return value


def validate_confidence_config(config: Mapping[str, Any]) -> None:
    """Validate fields that affect the released confidence protocol."""

    if int(config.get("schema_version", 0)) != 1:
        raise ConfigError("schema_version must be 1")
    dataset = _require_mapping(config, "dataset")
    model = _require_mapping(config, "model")
    views = _require_mapping(config, "views")
    mvacn = _require_mapping(config, "mvacn")
    training = _require_mapping(config, "training")
    selection = _require_mapping(config, "selection")
    _require_mapping(config, "stage1")

    if str(dataset.get("name", "")).upper() not in {"DDSM", "RSNA"}:
        raise ConfigError("dataset.name must be DDSM or RSNA")
    backbone = get_backbone(str(model.get("backbone", "")))
    if str(model.get("hf_model", backbone.hf_model)) != backbone.hf_model:
        raise ConfigError("model.hf_model does not match model.backbone")
    if int(model.get("image_size", 0)) != backbone.image_size:
        raise ConfigError("model.image_size does not match model.backbone")
    if not str(model.get("revision", "")):
        raise ConfigError("model.revision is required")
    attention_implementation = str(model.get("attention_implementation", "")).lower()
    if attention_implementation not in {"eager", "sdpa"}:
        raise ConfigError("model.attention_implementation must be eager or sdpa")
    order = tuple(views.get("input_order", ()))
    if len(order) != 4 or set(order) != {"L_CC", "L_MLO", "R_CC", "R_MLO"}:
        raise ConfigError("views.input_order must contain all four canonical views")

    heads = int(mvacn.get("attention_heads", 0))
    if heads < 1 or backbone.hidden_size % heads:
        raise ConfigError("mvacn.attention_heads must divide the backbone hidden size")
    dropout = float(mvacn.get("dropout", -1))
    if not 0 <= dropout < 1:
        raise ConfigError("mvacn.dropout must be in [0, 1)")
    hidden_dims = tuple(int(value) for value in mvacn.get("mlp_hidden_dims", ()))
    if not hidden_dims or any(value < 1 for value in hidden_dims):
        raise ConfigError("mvacn.mlp_hidden_dims must contain positive integers")
    if str(mvacn.get("output_activation", "sigmoid")).lower() != "sigmoid":
        raise ConfigError("only sigmoid MV-ACN output is supported")

    if int(training.get("epochs", 0)) < 1 or int(training.get("batch_size", 0)) < 1:
        raise ConfigError("training.epochs and training.batch_size must be positive")
    if str(training.get("loss", "mse")).lower() != "mse":
        raise ConfigError("legacy confidence training requires MSE loss")
    optimizer = training.get("optimizer")
    if not isinstance(optimizer, Mapping) or str(optimizer.get("name", "")).lower() != "adam":
        raise ConfigError("training.optimizer.name must be adam")
    if float(optimizer.get("learning_rate", 0)) <= 0:
        raise ConfigError("training.optimizer.learning_rate must be positive")

    if str(selection.get("split", "")) not in _SPLITS:
        raise ConfigError("selection.split must be train, valid, or test")
    if (str(selection.get("metric", "")).lower(), str(selection.get("mode", "")).lower()) != (
        "aupr",
        "max",
    ):
        raise ConfigError("confidence selection must maximize AUPR")
    tie_breaker = selection.get("tie_breaker")
    if not isinstance(tie_breaker, Mapping) or (
        str(tie_breaker.get("metric", "")).lower(),
        str(tie_breaker.get("mode", "")).lower(),
    ) != ("aurc", "min"):
        raise ConfigError("confidence selection tie breaker must minimize AURC")


def load_confidence_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load confidence YAML without applying the Stage-1 metric validator."""

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigError("Configuration root must be a mapping")
    config = copy.deepcopy(loaded)
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    validate_confidence_config(config)
    return config


def _resolve_config_path(config: Mapping[str, Any], value: str | os.PathLike[str]) -> Path:
    expanded = os.path.expanduser(os.path.expandvars(os.fspath(value)))
    if "$" in expanded:
        raise ConfigError(f"unresolved environment variable in path: {value}")
    path = Path(expanded)
    return path.resolve() if path.is_absolute() else (Path(config["_config_dir"]) / path).resolve()


def _resolve_tcp_root(config: Mapping[str, Any], override: str | os.PathLike[str] | None) -> Path:
    if override is not None:
        return Path(override).resolve()
    stage1 = _require_mapping(config, "stage1")
    value = stage1.get("tcp_root")
    if value is None:
        raise ConfigError("stage1.tcp_root is required when --tcp-root/--output-root is omitted")
    return _resolve_config_path(config, os.fspath(value))


def _resolve_output_root(
    config: Mapping[str, Any], override: str | os.PathLike[str] | None
) -> Path:
    if override is not None:
        return Path(override).resolve()
    output = _require_mapping(config, "output")
    return _resolve_config_path(config, os.fspath(output.get("root", "runs/confidence")))


def _device(name: str | None) -> torch.device:
    device = torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type) if enabled else nullcontext()


def _safe_child(root: Path, filename: str) -> Path:
    if not filename or Path(filename).name != filename:
        raise RuntimeError("artifact metadata contains an unsafe filename")
    return root / filename


def _load_verified_stage1(
    stage1_config: Mapping[str, Any],
    stage1_run_dir: str | os.PathLike[str],
    device: torch.device,
) -> VerifiedStage1:
    run_dir = Path(stage1_run_dir).resolve()
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if run.get("status") != "complete":
        raise RuntimeError("Stage-1 run is not marked complete")
    best = json.loads((run_dir / "best.json").read_text(encoding="utf-8"))
    config_hash = sha256_json(public_config(stage1_config))
    if best.get("config_sha256") != config_hash:
        raise RuntimeError("Stage-1 checkpoint configuration does not match")
    checkpoint = _safe_child(run_dir, str(best.get("weights_file", "")))
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != best.get("weights_sha256"):
        raise RuntimeError("Stage-1 checkpoint SHA-256 mismatch")
    model, input_ids = build_model(stage1_config, device)
    load_weights(model, checkpoint, device)
    return VerifiedStage1(model, input_ids, checkpoint_hash, best)


def _validate_config_pair(
    confidence_config: Mapping[str, Any], stage1_config: Mapping[str, Any]
) -> None:
    confidence_model = confidence_config["model"]
    stage1_model = stage1_config["model"]
    if (
        str(confidence_config["dataset"]["name"]).upper()
        != str(stage1_config["dataset"]["name"]).upper()
    ):
        raise ConfigError("confidence and Stage-1 configs use different datasets")
    if confidence_model["backbone"] != stage1_model["backbone"]:
        raise ConfigError("confidence and Stage-1 configs use different backbones")
    for field in (
        "hf_model",
        "revision",
        "image_size",
        "attention_implementation",
    ):
        if confidence_model.get(field) != stage1_model.get(field):
            raise ConfigError(f"confidence and Stage-1 configs use different model.{field}")
    if tuple(confidence_config["views"]["input_order"]) != tuple(
        stage1_config["views"]["input_order"]
    ):
        raise ConfigError("confidence and Stage-1 configs use different view orders")


def _data_config(
    stage1_config: Mapping[str, Any],
    confidence_config: Mapping[str, Any],
    *,
    augmented_train: bool,
) -> dict[str, Any]:
    config = copy.deepcopy(stage1_config)
    config["training"] = copy.deepcopy(confidence_config["training"])
    if not augmented_train:
        config["training"]["augmentation"] = {"name": "none"}
    return config


def _make_loader(
    dataset,
    training: Mapping[str, Any],
    *,
    shuffle: bool,
    drop_last: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    workers = int(training.get("num_workers", 0))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _validate_manifest(verified: VerifiedStage1, split: str, manifest_sha256: str) -> None:
    manifests = verified.best.get("manifest_sha256")
    if not isinstance(manifests, Mapping) or manifests.get(split) != manifest_sha256:
        raise RuntimeError(f"{split} manifest SHA-256 does not match the Stage-1 run")


def build_tcp_artifacts(
    confidence_config: Mapping[str, Any],
    stage1_config: Mapping[str, Any],
    stage1_run_dir: str | os.PathLike[str],
    *,
    splits: Sequence[str] = _SPLITS,
    data_root: str | None = None,
    manifest_dir: str | None = None,
    output_root: str | os.PathLike[str] | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Build sanitized, strict TCP artifacts from one verified Stage-1 run."""

    validate_confidence_config(confidence_config)
    _validate_config_pair(confidence_config, stage1_config)
    requested = tuple(splits)
    if (
        not requested
        or len(requested) != len(set(requested))
        or any(split not in _SPLITS for split in requested)
    ):
        raise ValueError("splits must be unique values from train, valid, test")
    device = _device(device_name)
    verified = _load_verified_stage1(stage1_config, stage1_run_dir, device)
    datasets = build_datasets(
        _data_config(stage1_config, confidence_config, augmented_train=False),
        data_root=data_root,
        manifest_dir=manifest_dir,
    )
    root = _resolve_tcp_root(confidence_config, output_root)
    root.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    for split in requested:
        dataset = datasets[split]
        _validate_manifest(verified, split, dataset.manifest_sha256)
        loader = _make_loader(
            dataset,
            confidence_config["training"],
            shuffle=False,
            drop_last=False,
            seed=int(confidence_config["training"]["seed"]),
            device=device,
        )
        prediction = predict_loader(
            verified.model,
            verified.input_ids,
            loader,
            device,
            amp=bool(confidence_config["training"].get("amp", False)),
        )
        artifact = build_tcp_artifact(
            prediction.probabilities,
            prediction.labels,
            predictions=prediction.predictions,
            sample_indexes=prediction.sample_indexes,
            split=split,
            stage1_checkpoint_sha256=verified.checkpoint_sha256,
            manifest_sha256=dataset.manifest_sha256,
        )
        filename = f"{split}.jsonl"
        write_tcp_artifact(artifact, root / filename)
        records[split] = {
            "file": filename,
            "n_samples": len(artifact),
            **asdict(artifact.provenance),
        }
    index = {
        "schema_version": 1,
        "dataset": confidence_config["dataset"]["name"],
        "backbone": confidence_config["model"]["backbone"],
        "artifacts": records,
    }
    atomic_write_json(root / "index.json", index)
    return index


def _tcp_path(root: Path, split: str) -> Path:
    candidates = [root / f"{split}.jsonl", root / f"{split}.csv"]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"expected exactly one TCP artifact for {split}: "
            + ", ".join(path.name for path in candidates)
        )
    return existing[0]


def _load_tcp(
    root: Path,
    split: str,
    dataset,
    checkpoint_sha256: str,
) -> TCPArtifact:
    artifact = read_tcp_artifact(_tcp_path(root, split))
    if artifact.provenance.split != split:
        raise RuntimeError(f"TCP artifact split mismatch for {split}")
    if artifact.provenance.manifest_sha256 != dataset.manifest_sha256:
        raise RuntimeError(f"TCP artifact manifest mismatch for {split}")
    if artifact.provenance.stage1_checkpoint_sha256 != checkpoint_sha256:
        raise RuntimeError(f"TCP artifact Stage-1 checkpoint mismatch for {split}")
    if len(artifact) != len(dataset):
        raise RuntimeError(f"TCP artifact sample count mismatch for {split}")
    return artifact


def _batch_arrays(batch: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    indexes = batch["sample_index"]
    labels = batch["label"]
    if isinstance(indexes, Tensor):
        indexes = indexes.detach().cpu().numpy()
    if isinstance(labels, Tensor):
        labels = labels.detach().cpu().numpy()
    indexes = np.asarray(indexes, dtype=np.int64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(indexes) != len(labels) or len(np.unique(indexes)) != len(indexes):
        raise RuntimeError("batch sample indexes are duplicated or misaligned")
    return indexes, labels


def tcp_targets_for_batch(
    artifact: TCPArtifact,
    batch: Mapping[str, Any],
    device: torch.device,
) -> Tensor:
    """Strict sample-index join used during MV-ACN training and inference."""

    indexes, labels = _batch_arrays(batch)
    targets: list[float] = []
    for sample_index, label in zip(indexes, labels, strict=True):
        if sample_index < 0 or sample_index >= len(artifact):
            raise RuntimeError(f"sample index {sample_index} is absent from TCP artifact")
        record = artifact.records[int(sample_index)]
        if record.sample_index != sample_index or record.label != label:
            raise RuntimeError(f"TCP join mismatch at sample index {sample_index}")
        targets.append(record.tcp)
    return torch.as_tensor(targets, dtype=torch.float32, device=device)


def _move_views(batch: Mapping[str, Any], device: torch.device) -> dict[str, Tensor]:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in batch["views"].items()}


def _hidden_views(
    stage1_model: nn.Module,
    views: Mapping[str, Tensor],
    input_order: Sequence[str],
) -> tuple[Tensor, ...]:
    _embeddings, hidden = stage1_model.encode_views(views)
    missing = set(input_order) - set(hidden)
    if missing:
        raise RuntimeError(f"Stage-1 hidden states are missing views: {sorted(missing)}")
    return tuple(hidden[name] for name in input_order)


def _head_config(confidence_config: Mapping[str, Any]) -> MVACNConfig:
    backbone = get_backbone(str(confidence_config["model"]["backbone"]))
    mvacn = confidence_config["mvacn"]
    return MVACNConfig(
        hidden_dim=backbone.hidden_size,
        attention_heads=int(mvacn["attention_heads"]),
        mlp_dims=tuple(int(value) for value in mvacn["mlp_hidden_dims"]),
        dropout=float(mvacn["dropout"]),
        drop_cls_tokens=bool(mvacn["drop_view_cls_tokens"]),
    )


def _train_epoch(
    head: MVACNHead,
    stage1_model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    artifact: TCPArtifact,
    input_order: Sequence[str],
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    amp: bool,
) -> float:
    head.train()
    stage1_model.eval()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        views = _move_views(batch, device)
        targets = tcp_targets_for_batch(artifact, batch, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            with torch.no_grad():
                hidden = _hidden_views(stage1_model, views, input_order)
            confidence = head.predict_confidence(*hidden)
            loss = torch.nn.functional.mse_loss(confidence, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        count = len(targets)
        total_loss += float(loss.item()) * count
        total_count += count
    if total_count == 0:
        raise RuntimeError("confidence training loader is empty")
    return total_loss / total_count


def _head_confidence(
    head: MVACNHead,
    stage1_model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    artifact: TCPArtifact,
    input_order: Sequence[str],
    device: torch.device,
    amp: bool,
) -> np.ndarray:
    head.eval()
    stage1_model.eval()
    indexes: list[np.ndarray] = []
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            batch_indexes, _labels = _batch_arrays(batch)
            _ = tcp_targets_for_batch(artifact, batch, device)
            views = _move_views(batch, device)
            with _autocast(device, amp):
                hidden = _hidden_views(stage1_model, views, input_order)
                confidence = head.predict_confidence(*hidden)
            indexes.append(batch_indexes)
            values.append(confidence.detach().float().cpu().numpy())
    if not values:
        raise RuntimeError("confidence evaluation loader is empty")
    all_indexes = np.concatenate(indexes)
    if len(np.unique(all_indexes)) != len(all_indexes):
        raise RuntimeError("confidence loader contains duplicate sample indexes")
    order = np.argsort(all_indexes, kind="stable")
    if not np.array_equal(all_indexes[order], np.arange(len(artifact))):
        raise RuntimeError("confidence loader is missing canonical sample indexes")
    return np.concatenate(values)[order]


def _save_head(head: MVACNHead, path: Path) -> None:
    from safetensors.torch import save_file

    save_file(export_head_state_dict(head), str(path))


def _is_better(aupr: float, aurc: float, best_aupr: float, best_aurc: float) -> bool:
    return aupr > best_aupr or (aupr == best_aupr and aurc < best_aurc)


def _write_run_status(run_dir: Path, status: str, **updates: Any) -> None:
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    metadata.update({"status": status, **updates})
    atomic_write_json(run_dir / "run.json", metadata)


def train_confidence(
    confidence_config: Mapping[str, Any],
    stage1_config: Mapping[str, Any],
    stage1_run_dir: str | os.PathLike[str],
    tcp_root: str | os.PathLike[str],
    *,
    data_root: str | None = None,
    manifest_dir: str | None = None,
    output_root: str | os.PathLike[str] | None = None,
    run_name: str | None = None,
    device_name: str | None = None,
    allow_test_selection: bool = False,
) -> Path:
    """Train an MV-ACN head while keeping the verified Stage-1 model frozen."""

    validate_confidence_config(confidence_config)
    _validate_config_pair(confidence_config, stage1_config)
    selection_split = str(confidence_config["selection"]["split"])
    if selection_split == "test" and not allow_test_selection:
        raise ValueError(
            "test-informed checkpoint selection is legacy behavior; pass "
            "--allow-test-selection explicitly to reproduce it"
        )
    biased_legacy_protocol = selection_split == "test"
    if biased_legacy_protocol:
        warnings.warn(
            "LEGACY PROTOCOL: selecting the MV-ACN checkpoint on the test split is "
            "biased. This mode is provided only to reproduce the "
            "archived experiment.",
            RuntimeWarning,
            stacklevel=2,
        )
    training = confidence_config["training"]
    seed = int(training["seed"])
    set_seed(seed, deterministic=bool(training.get("deterministic", False)))
    device = _device(device_name)
    amp = bool(training.get("amp", False))
    if amp and device.type != "cuda":
        raise RuntimeError("AMP confidence training requires CUDA")
    verified = _load_verified_stage1(stage1_config, stage1_run_dir, device)
    verified.model.requires_grad_(False)
    verified.model.eval()

    datasets = build_datasets(
        _data_config(stage1_config, confidence_config, augmented_train=True),
        data_root=data_root,
        manifest_dir=manifest_dir,
    )
    for split in {"train", selection_split}:
        _validate_manifest(verified, split, datasets[split].manifest_sha256)
    tcp_directory = Path(tcp_root).resolve()
    artifacts = {
        split: _load_tcp(tcp_directory, split, datasets[split], verified.checkpoint_sha256)
        for split in {"train", selection_split}
    }

    output = _resolve_output_root(confidence_config, output_root)
    prefix = (
        f"{str(confidence_config['dataset']['name']).lower()}-"
        f"{confidence_config['model']['backbone']}-confidence"
    )
    run_dir = create_run_dir(output, prefix=prefix, run_name=run_name)
    config_hash = sha256_json(public_config(confidence_config))
    stage1_config_hash = sha256_json(public_config(stage1_config))
    atomic_write_json(run_dir / "config.json", public_config(confidence_config))
    atomic_write_json(
        run_dir / "run.json",
        {
            "schema_version": 1,
            "status": "running",
            "confidence_config_sha256": config_hash,
            "stage1_config_sha256": stage1_config_hash,
            "stage1_checkpoint_sha256": verified.checkpoint_sha256,
            "selection_split": selection_split,
            "test_selection_explicitly_allowed": bool(allow_test_selection),
            "biased_legacy_protocol": biased_legacy_protocol,
            "tcp_provenance": {
                split: asdict(artifact.provenance) for split, artifact in artifacts.items()
            },
            "environment": environment_state(),
            "git": git_state(Path(__file__).resolve().parents[3]),
        },
    )

    head = MVACNHead(_head_config(confidence_config)).to(device)
    optimizer_config = training["optimizer"]
    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    train_loader = _make_loader(
        datasets["train"],
        training,
        shuffle=True,
        drop_last=True,
        seed=seed,
        device=device,
    )
    selection_loader = _make_loader(
        datasets[selection_split],
        training,
        shuffle=False,
        drop_last=False,
        seed=seed,
        device=device,
    )
    input_order = tuple(confidence_config["views"]["input_order"])
    correct = np.asarray(
        [int(record.correct) for record in artifacts[selection_split].records],
        dtype=np.int64,
    )
    best_aupr = -float("inf")
    best_aurc = float("inf")
    best_epoch: int | None = None
    metrics_path = run_dir / "metrics.jsonl"
    try:
        for epoch in range(int(training["epochs"])):
            train_loss = _train_epoch(
                head,
                verified.model,
                train_loader,
                artifacts["train"],
                input_order,
                optimizer,
                scaler,
                device,
                amp,
            )
            selection_confidence = _head_confidence(
                head,
                verified.model,
                selection_loader,
                artifacts[selection_split],
                input_order,
                device,
                amp,
            )
            metrics = confidence_selection_metrics(correct, selection_confidence)
            if metrics.aupr is None:
                raise RuntimeError("AUPR is undefined because selection correctness has one class")
            improved = _is_better(metrics.aupr, metrics.aurc, best_aupr, best_aurc)
            record = {
                "epoch": epoch,
                "train": {"mse": train_loss},
                selection_split: metrics.to_dict(),
                "selection_metric": "aupr",
                "selection_value": metrics.aupr,
                "tie_breaker": "aurc",
                "tie_breaker_value": metrics.aurc,
                "is_best": improved,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            if improved:
                best_aupr, best_aurc, best_epoch = metrics.aupr, metrics.aurc, epoch
                weights = run_dir / "best.safetensors"
                _save_head(head, weights)
                atomic_write_json(
                    run_dir / "best.json",
                    {
                        "epoch": epoch,
                        "metric": "aupr",
                        "mode": "max",
                        "value": metrics.aupr,
                        "tie_breaker": {"metric": "aurc", "mode": "min", "value": metrics.aurc},
                        "selection_split": selection_split,
                        "confidence_config_sha256": config_hash,
                        "stage1_config_sha256": stage1_config_hash,
                        "stage1_checkpoint_sha256": verified.checkpoint_sha256,
                        "stage1_prediction_sha256": artifacts[
                            selection_split
                        ].provenance.stage1_prediction_sha256,
                        "manifest_sha256": artifacts[selection_split].provenance.manifest_sha256,
                        "mvacn": asdict(head.config),
                        "weights_file": weights.name,
                        "weights_sha256": sha256_file(weights),
                    },
                )
        if best_epoch is None:
            raise RuntimeError("confidence training completed without a best checkpoint")
    except BaseException:
        _write_run_status(run_dir, "failed")
        raise
    _write_run_status(run_dir, "complete", best_epoch=best_epoch)
    return run_dir
