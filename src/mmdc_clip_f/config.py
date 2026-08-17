"""Configuration loading and validation.

The release intentionally keeps experiment behavior in YAML instead of hidden
defaults.  This module performs structural validation without adding a heavy
configuration-framework dependency.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml

from .backbones import get_backbone


class ConfigError(ValueError):
    """Raised when an experiment configuration is incomplete or inconsistent."""


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing required setting {context}.{key}")
    return mapping[key]


def _expanded_path(value: str | os.PathLike[str], base: Path) -> Path:
    raw = os.path.expanduser(os.path.expandvars(os.fspath(value)))
    if "$" in raw:
        raise ConfigError(f"Unresolved environment variable in path: {value}")
    path = Path(raw)
    return path if path.is_absolute() else (base / path).resolve()


def set_dotted(config: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cursor = config
    for key in keys[:-1]:
        child = cursor.setdefault(key, {})
        if not isinstance(child, MutableMapping):
            raise ConfigError(f"Cannot override {dotted_key}: {key} is not a mapping")
        cursor = child
    cursor[keys[-1]] = value


def load_config(
    path: str | os.PathLike[str],
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigError("Configuration root must be a mapping")

    config: dict[str, Any] = copy.deepcopy(loaded)
    for key, value in (overrides or {}).items():
        if value is not None:
            set_dotted(config, key, value)
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    validate_classifier_config(config)
    return config


def validate_classifier_config(config: Mapping[str, Any]) -> None:
    if int(config.get("schema_version", 0)) != 1:
        raise ConfigError("schema_version must be 1")

    dataset = _require(config, "dataset", "config")
    model = _require(config, "model", "config")
    views = _require(config, "views", "config")
    training = _require(config, "training", "config")
    selection = _require(config, "selection", "config")
    for name, section in (
        ("dataset", dataset),
        ("model", model),
        ("views", views),
        ("training", training),
        ("selection", selection),
    ):
        if not isinstance(section, Mapping):
            raise ConfigError(f"{name} must be a mapping")

    dataset_name = str(_require(dataset, "name", "dataset")).upper()
    if dataset_name not in {"DDSM", "RSNA"}:
        raise ConfigError("dataset.name must be DDSM or RSNA")

    spec = get_backbone(str(_require(model, "backbone", "model")))
    if str(model.get("hf_model", spec.hf_model)) != spec.hf_model:
        raise ConfigError("model.hf_model does not match model.backbone")
    if int(model.get("image_size", spec.image_size)) != spec.image_size:
        raise ConfigError("model.image_size does not match model.backbone")
    attention_implementation = str(_require(model, "attention_implementation", "model")).lower()
    if attention_implementation not in {"eager", "sdpa"}:
        raise ConfigError("model.attention_implementation must be eager or sdpa")

    order = tuple(_require(views, "input_order", "views"))
    expected = {"L_CC", "L_MLO", "R_CC", "R_MLO"}
    if len(order) != 4 or set(order) != expected:
        raise ConfigError(f"views.input_order must contain exactly {sorted(expected)}")
    pairs = _require(views, "fusion_pairs", "views")
    if len(pairs) != 2 or any(len(pair) != 2 for pair in pairs):
        raise ConfigError("views.fusion_pairs must contain two pairs")
    flattened = [view for pair in pairs for view in pair]
    if set(flattened) != expected or len(flattened) != 4:
        raise ConfigError("views.fusion_pairs must use every view exactly once")

    is_confidence = isinstance(config.get("stage1"), Mapping) and isinstance(
        config.get("mvacn"), Mapping
    )
    expected_modes = (
        {"aupr": "max", "aurc": "min", "brier": "min"}
        if is_confidence
        else {"accuracy": "max", "batch_mean_accuracy": "max", "loss": "min"}
    )
    metric = str(_require(selection, "metric", "selection")).lower()
    if metric not in expected_modes:
        choices = ", ".join(expected_modes)
        profile = "confidence" if is_confidence else "classifier"
        raise ConfigError(f"selection.metric for a {profile} config must be one of: {choices}")
    mode = str(selection.get("mode", expected_modes[metric])).lower()
    if mode not in {"min", "max"}:
        raise ConfigError("selection.mode must be min or max")
    if mode != expected_modes[metric]:
        raise ConfigError(f"selection.metric {metric} must use mode {expected_modes[metric]}")
    if int(training.get("batch_size", 0)) < 1:
        raise ConfigError("training.batch_size must be positive")
    if int(training.get("epochs", 0)) < 1:
        raise ConfigError("training.epochs must be positive")


def resolve_data_paths(
    config: Mapping[str, Any],
    data_root: str | os.PathLike[str] | None = None,
    manifest_dir: str | os.PathLike[str] | None = None,
) -> tuple[Path, dict[str, Path]]:
    dataset = config["dataset"]
    base = Path(config["_config_dir"])
    # FourViewDataset consumes the image directory specifically.  Resolving
    # the broader data root here would recreate the unpublished DDSM symlink
    # dependency that this release is intended to remove.
    configured_image_root = _require(dataset, "image_root", "dataset")
    resolved_root = _expanded_path(data_root or configured_image_root, base)
    manifests = dataset.get("manifests")
    if not isinstance(manifests, Mapping):
        raise ConfigError("dataset.manifests must map train/valid/test to files")
    resolved: dict[str, Path] = {}
    for split in ("train", "valid", "test"):
        value = _require(manifests, split, "dataset.manifests")
        if manifest_dir is None:
            resolved[split] = _expanded_path(value, base)
        else:
            resolved[split] = Path(manifest_dir).resolve() / Path(value).name
    return resolved_root, resolved


def resolve_output_root(
    config: Mapping[str, Any],
    output_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve a run root relative to the experiment configuration file."""
    base = Path(config["_config_dir"])
    output = config.get("output", {})
    if not isinstance(output, Mapping):
        raise ConfigError("output must be a mapping")
    configured = output_root or output.get("root", "runs")
    return _expanded_path(configured, base)


def public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return behavioral settings without private loader keys or local paths."""
    public = {
        key: copy.deepcopy(value) for key, value in config.items() if not str(key).startswith("_")
    }

    dataset = public.get("dataset")
    if isinstance(dataset, MutableMapping):
        for key in ("data_root", "image_root"):
            if key in dataset:
                dataset[key] = "<local>"
        manifests = dataset.get("manifests")
        if isinstance(manifests, MutableMapping):
            for split, path in list(manifests.items()):
                manifests[split] = Path(os.fspath(path)).name

    output = public.get("output")
    if isinstance(output, MutableMapping) and "root" in output:
        output["root"] = "<local>"

    stage1 = public.get("stage1")
    if isinstance(stage1, MutableMapping):
        for key in ("checkpoint", "tcp_root"):
            if key in stage1:
                stage1[key] = "<local>"
    return public
