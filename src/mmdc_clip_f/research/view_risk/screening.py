"""Batched scientific helpers for a development-only signed-effect screen.

These artifacts deliberately cannot qualify the full P5B pilot. Input preparation,
head architecture, loss and scheduled exposures reuse the study implementation.
"""
from dataclasses import dataclass, fields
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from sklearn.metrics import roc_auc_score

from .fusion import RSNA_FUSION_PAIRS
from .head_inputs import RawHeadInputs
from .inputs import CANONICAL_VIEWS
from .metrics import confidence_panel_metrics
from .perturbations import realize_parent, validate_training_sample_spec
from .targets import InterventionTargets
from .training import build_learned_method


SCREENING_METHODS = ("candidate", "no_effect_supervision", "magnitude_effects")
RAW_FIELDS = tuple(f.name for f in fields(RawHeadInputs) if f.name not in
                   ("backbone", "token_policy", "fusion_pairs", "view_order"))
TARGET_FIELDS = ("observed_prediction", "observed_error", "omission_predictions",
                 "omission_effects", "omission_labels", "valid_removal_mask")


@dataclass(frozen=True)
class ScreeningBatch:
    raw: RawHeadInputs
    targets: InterventionTargets


def matched_screening_model(method: str, seed: int, backbone: str):
    """Match all shared parameters, including the post-auxiliary risk MLP."""
    if method not in SCREENING_METHODS:
        raise ValueError("method is outside the screening comparison")
    # Building the binary auxiliary classifier consumes fewer draws than the
    # signed classifier. Copy every shape-matched tensor from a common reference
    # so this does not accidentally change the risk MLP initialization.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference = build_learned_method("candidate", backbone, RSNA_FUSION_PAIRS)
        if method == "candidate":
            return reference
        torch.manual_seed(seed)
        model = build_learned_method(method, backbone, RSNA_FUSION_PAIRS)
        source = reference.state_dict()
        state = model.state_dict()
        for name, value in state.items():
            if source[name].shape == value.shape:
                state[name] = source[name].clone()
        model.load_state_dict(state, strict=True)
        return model


def pack_screening_batch(
    groups: Sequence[tuple[RawHeadInputs, InterventionTargets]],
    row_indices: Sequence[Sequence[int]],
) -> ScreeningBatch:
    """Restore manifest order after homogeneous-mask extraction.

    For mixed masks, targets use the four canonical *column names*, with a
    per-row validity mask. Loss normalization remains per parent, including a
    zero auxiliary term for singleton parents.
    """
    if not groups or len(groups) != len(row_indices):
        raise ValueError("groups and row indices must align")
    raw0 = groups[0][0]
    indices = [int(i) for group in row_indices for i in group]
    if sorted(indices) != list(range(len(indices))):
        raise ValueError("row indices must be a permutation of all original rows")
    order = torch.tensor(np.argsort(indices), device=raw0.current_scores.device)
    for (raw, target), rows in zip(groups, row_indices):
        if raw.batch_size != len(rows):
            raise ValueError("row count differs from extracted inputs")
        if (raw.backbone, raw.token_policy, raw.fusion_pairs) != (
            raw0.backbone, raw0.token_policy, raw0.fusion_pairs,
        ) or target.fusion_pairs != raw.fusion_pairs:
            raise ValueError("group input/fusion identities disagree")
        if not torch.equal(raw.current_prediction, target.observed_prediction):
            raise ValueError("target prediction is stale")
        valid = target.valid_removal_mask
        if valid.ndim == 1:
            valid = valid[None].expand(raw.batch_size, -1)
        if not torch.equal(valid, raw.removal_valid_mask):
            raise ValueError("target validity mask is stale")
    raw = RawHeadInputs(**{
        name: torch.cat([getattr(r, name) for r, _ in groups])[order].detach()
        for name in RAW_FIELDS
    }, backbone=raw0.backbone, token_policy=raw0.token_policy, fusion_pairs=raw0.fusion_pairs)
    target_tensors = {}
    for name in TARGET_FIELDS:
        values = []
        for r, t in groups:
            value = getattr(t, name)
            if name == "valid_removal_mask" and value.ndim == 1:
                value = value[None].expand(r.batch_size, -1)
            values.append(value)
        target_tensors[name] = torch.cat(values)[order].detach()
    return ScreeningBatch(raw, InterventionTargets(
        observed_views=CANONICAL_VIEWS, fusion_pairs=raw.fusion_pairs, **target_tensors,
    ))


def screening_tensors(batch: ScreeningBatch) -> dict[str, Tensor]:
    return {**{f"raw.{n}": getattr(batch.raw, n).detach().cpu().contiguous() for n in RAW_FIELDS},
            **{f"target.{n}": getattr(batch.targets, n).detach().cpu().contiguous()
               for n in TARGET_FIELDS}}


def screening_batch_from_tensors(tensors: dict[str, Tensor], backbone: str) -> ScreeningBatch:
    if set(tensors) != ({f"raw.{n}" for n in RAW_FIELDS} | {f"target.{n}" for n in TARGET_FIELDS}):
        raise ValueError("screening tensor cache schema is invalid")
    raw = RawHeadInputs(**{n: tensors[f"raw.{n}"] for n in RAW_FIELDS}, backbone=backbone,
                        token_policy="exclude_cls", fusion_pairs=RSNA_FUSION_PAIRS)
    targets = InterventionTargets(**{n: tensors[f"target.{n}"] for n in TARGET_FIELDS},
                                  observed_views=CANONICAL_VIEWS, fusion_pairs=RSNA_FUSION_PAIRS)
    # Reuse validation even for a single already-batched group.
    return pack_screening_batch([(raw, targets)], [list(range(raw.batch_size))])


def slice_screening_batch(batch: ScreeningBatch, index, *, device) -> ScreeningBatch:
    raw = RawHeadInputs(**{n: getattr(batch.raw, n)[index].to(device) for n in RAW_FIELDS},
                        backbone=batch.raw.backbone, token_policy=batch.raw.token_policy,
                        fusion_pairs=batch.raw.fusion_pairs)
    target = InterventionTargets(**{n: getattr(batch.targets, n)[index].to(device)
                                   for n in TARGET_FIELDS}, observed_views=CANONICAL_VIEWS,
                                 fusion_pairs=batch.targets.fusion_pairs)
    return ScreeningBatch(raw, target)


def validate_screening_labels(batch: ScreeningBatch, labels: Tensor) -> None:
    target, raw = batch.targets, batch.raw
    if labels.shape != raw.current_prediction.shape or labels.dtype != torch.long or (
        (labels < 0) | (labels > 3)
    ).any():
        raise ValueError("screening labels must be aligned integers in 0..3")
    expected_error = (raw.current_prediction != labels).long()
    valid = raw.removal_valid_mask
    if not torch.equal(target.observed_error, expected_error):
        raise ValueError("screening error targets are stale")
    if not torch.equal(target.omission_predictions, raw.omission_prediction):
        raise ValueError("screening omission predictions are stale")
    expected_effect = expected_error[:, None] - (raw.omission_prediction != labels[:, None]).long()
    if not torch.equal(target.omission_effects[valid], expected_effect[valid]) or not torch.equal(
        target.omission_labels[valid], expected_effect[valid] + 1,
    ):
        raise ValueError("screening omission effects are stale")


def realize_scheduled_parent(images, draw):
    validate_training_sample_spec(draw, operation="confidence-fit")
    if draw.stress_mode == "common":
        return realize_parent(images, draw.observed_views, common_mode=draw.perturbation)
    return realize_parent(images, draw.observed_views,
                          perturbations={v: draw.perturbation for v in draw.stressed_views})


def summarize_screening_scores(error, confidence) -> dict:
    """Keep clean and balanced 16-cell endpoints separate; never pool patients."""
    error, confidence = np.asarray(error), np.asarray(confidence)
    if error.shape != confidence.shape or error.ndim != 2 or error.shape[0] != 17:
        raise ValueError("screening requires 17 aligned cells")
    cells = []
    for e, c in zip(error, confidence):
        metric = confidence_panel_metrics(1 - e, c, confidence_kind="probability")
        cells.append({"aurc": metric.aurc, "error_ap": metric.error_positive_ap.value,
                      "error_auroc": float(roc_auc_score(e, -c)) if len(np.unique(e)) == 2 else None,
                      "brier": metric.brier, "accuracy": metric.accuracy,
                      "risk80": metric.risk_at_80.risk, "risk90": metric.risk_at_90.risk})
    return {"clean_aurc": cells[0]["aurc"],
            "balanced_aurc": float(np.mean([x["aurc"] for x in cells[1:]])), "cells": cells}


def choose_screening_epoch(history: Sequence[dict]) -> dict:
    if not history or len({x["epoch"] for x in history}) != len(history):
        raise ValueError("checkpoint history must be nonempty with unique epochs")
    return min(history, key=lambda x: (x["balanced_aurc"], x["epoch"]))
