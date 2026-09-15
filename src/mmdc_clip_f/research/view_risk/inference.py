"""Reload verified confidence artifacts and score label-free frozen features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from .artifacts import analytic_control_confidence, load_control_artifact
from .baselines import build_ds_features
from .cache import CacheBundle
from .evaluation import ModelArtifactSelection
from .head_inputs import prepare_raw_head_inputs
from .production import load_tensor_checkpoint_state
from .training import (
    LEARNED_TORCH_METHODS,
    ClassifierProvenance,
    ResearchRunConfig,
    build_learned_method,
    confidence_bundle_scores,
)


@dataclass(frozen=True)
class ConfidenceBatchOutput:
    confidence: Tensor
    confidence_kind: str
    classifier_prediction: Tensor


def load_learned_confidence_model(
    selection: ModelArtifactSelection,
    *,
    config: ResearchRunConfig,
    classifier: ClassifierProvenance,
    example: CacheBundle,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Reconstruct one accepted method and load its rehashed selected state."""

    selection.verify_current_artifact()
    if (
        selection.method not in LEARNED_TORCH_METHODS
        or selection.config_sha256 != config.sha256
        or selection.classifier_checkpoint_sha256 != classifier.checkpoint_sha256
        or selection.tune_manifest_sha256 != classifier.tune_manifest_sha256
    ):
        raise ValueError("learned confidence selection binding is stale")
    features = example.features
    torch.manual_seed(selection.seed)
    model = build_learned_method(
        selection.method,
        config.backbone,
        features.fusion_pairs,
        visual_width=int(
            features.projected_by_view[features.observed_views[0]].shape[1]
        ),
        text_embeddings=features.normalized_text_embeddings,
    ).to(device)
    model.load_state_dict(load_tensor_checkpoint_state(selection.artifact_path), strict=True)
    model.eval()
    return model


def load_learned_confidence_checkpoint(
    *,
    method: str,
    seed: int,
    checkpoint_path: str | Path | None = None,
    config: ResearchRunConfig,
    example: CacheBundle,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Build one frozen-table architecture and load an already-verified epoch state."""

    if method not in LEARNED_TORCH_METHODS or seed not in config.seeds:
        raise ValueError("confidence checkpoint method/seed is outside the frozen table")
    if checkpoint_path is None:
        raise ValueError("confidence checkpoint path is required")
    features = example.features
    torch.manual_seed(seed)
    model = build_learned_method(
        method,
        config.backbone,
        features.fusion_pairs,
        visual_width=int(
            features.projected_by_view[features.observed_views[0]].shape[1]
        ),
        text_embeddings=features.normalized_text_embeddings,
    ).to(device)
    model.load_state_dict(load_tensor_checkpoint_state(checkpoint_path), strict=True)
    model.eval()
    return model


def score_confidence_cache_bundle(
    selection: ModelArtifactSelection,
    bundle: CacheBundle,
    *,
    learned_model: nn.Module | None = None,
) -> ConfidenceBatchOutput:
    """Score one bounded cache using only label-free feature tensors."""

    selection.verify_current_artifact()
    prediction = bundle.features.prediction.detach()
    if selection.method in LEARNED_TORCH_METHODS:
        if learned_model is None:
            raise ValueError("learned confidence scoring requires the selected model")
        with torch.no_grad():
            confidence = confidence_bundle_scores(
                learned_model, selection.method, bundle
            ).cpu()
        return ConfidenceBatchOutput(confidence, "probability", prediction.cpu())
    if learned_model is not None:
        raise ValueError("analytic confidence scoring cannot receive a learned model")
    artifact = load_control_artifact(selection.artifact_path)
    raw = prepare_raw_head_inputs(
        bundle.features, backbone=bundle.provenance.backbone
    )
    ds_features = (
        build_ds_features(
            bundle.features.logits_by_view,
            bundle.features.observed_views,
            fusion_pairs=bundle.features.fusion_pairs,
        )
        if selection.method == "ds_logistic"
        else None
    )
    confidence, kind = analytic_control_confidence(
        artifact,
        scores=bundle.features.scores,
        ds_features=ds_features,
        omission_score_differences=raw.omission_score_differences,
        removal_valid_mask=raw.removal_valid_mask,
    )
    if confidence.shape != prediction.shape or not torch.isfinite(confidence).all():
        raise RuntimeError("analytic confidence output shape/value is invalid")
    return ConfidenceBatchOutput(confidence.detach().cpu(), kind, prediction.cpu())


__all__ = [
    "ConfidenceBatchOutput",
    "load_learned_confidence_model",
    "load_learned_confidence_checkpoint",
    "score_confidence_cache_bundle",
]
