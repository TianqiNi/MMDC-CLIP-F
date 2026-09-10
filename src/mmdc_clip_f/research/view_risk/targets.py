"""Exact label-supervised targets derived from observed-view interventions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor

from .fusion import (
    RSNA_FUSION_PAIRS,
    FusionPairs,
    fuse_view_logits,
    validate_fusion_pairs,
)
from .inputs import CANONICAL_VIEWS, NUM_CLASSES


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


@dataclass(frozen=True)
class InterventionTargets:
    """Observed error and per-view removal effects in canonical columns.

    Invalid removal entries contain -1 and must be ignored using
    valid_removal_mask. For valid entries, omission_labels is the
    three-class encoding omission_effects + 1.
    """

    observed_views: tuple[str, ...]
    observed_prediction: Tensor
    observed_error: Tensor
    omission_predictions: Tensor
    omission_effects: Tensor
    omission_labels: Tensor
    valid_removal_mask: Tensor
    fusion_pairs: FusionPairs
    view_order: tuple[str, ...] = field(default=CANONICAL_VIEWS, init=False)


def _validate_labels(labels: Tensor, batch_size: int, device: torch.device) -> None:
    if not isinstance(labels, Tensor):
        raise TypeError("labels must be a torch.Tensor")
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if labels.shape[0] != batch_size:
        raise ValueError("labels must match the logits batch size")
    if labels.dtype not in _INTEGER_DTYPES:
        raise TypeError("labels must use an integer dtype")
    if labels.device != device:
        raise ValueError("labels and logits must be on the same device")
    if ((labels < 0) | (labels >= NUM_CLASSES)).any():
        raise ValueError("labels must be in range 0..3")


def build_intervention_targets(
    logits_by_view: Mapping[str, Tensor],
    labels: Tensor,
    observed: Iterable[str] | Sequence[bool] | Tensor,
    *,
    fusion_pairs: Sequence[Sequence[str]] = RSNA_FUSION_PAIRS,
) -> InterventionTargets:
    """Build exact omission targets using one validated configured fusion tree.

    The API accepts no classifier or image inputs and runs under no_grad.
    Consequently target construction cannot update Stage-1 parameters or turn a
    missing view into an encoded placeholder image.
    """

    pairs = validate_fusion_pairs(fusion_pairs)
    with torch.no_grad():
        observed_result = fuse_view_logits(logits_by_view, observed, fusion_pairs=pairs)
        batch_size = observed_result.scores.shape[0]
        _validate_labels(labels, batch_size, observed_result.scores.device)

        observed_prediction = observed_result.scores.argmax(dim=1)
        observed_error = (observed_prediction != labels).to(torch.long)
        output_shape = (batch_size, len(CANONICAL_VIEWS))
        omission_predictions = torch.full(output_shape, -1, dtype=torch.long, device=labels.device)
        omission_effects = torch.full_like(omission_predictions, -1)
        omission_labels = torch.full_like(omission_predictions, -1)
        valid_removal_mask = torch.zeros(
            len(CANONICAL_VIEWS),
            dtype=torch.bool,
            device=labels.device,
        )

        if len(observed_result.observed_views) > 1:
            for column, removed_view in enumerate(CANONICAL_VIEWS):
                if removed_view not in observed_result.observed_views:
                    continue
                remaining = tuple(
                    view for view in observed_result.observed_views if view != removed_view
                )
                omission_result = fuse_view_logits(
                    logits_by_view,
                    remaining,
                    fusion_pairs=pairs,
                )
                prediction = omission_result.scores.argmax(dim=1)
                omission_error = (prediction != labels).to(torch.long)
                effect = observed_error - omission_error
                omission_predictions[:, column] = prediction
                omission_effects[:, column] = effect
                omission_labels[:, column] = effect + 1
                valid_removal_mask[column] = True

    return InterventionTargets(
        observed_views=observed_result.observed_views,
        observed_prediction=observed_prediction,
        observed_error=observed_error,
        omission_predictions=omission_predictions,
        omission_effects=omission_effects,
        omission_labels=omission_labels,
        valid_removal_mask=valid_removal_mask,
        fusion_pairs=pairs,
    )
