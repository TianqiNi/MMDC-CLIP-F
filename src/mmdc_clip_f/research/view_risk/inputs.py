"""Named-view input contracts for intervention-supervised research."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor


CANONICAL_VIEWS = ("L_CC", "L_MLO", "R_CC", "R_MLO")
NUM_CLASSES = 4


class NoObservedViewsError(ValueError):
    """Raised when a caller requests fusion without any observed evidence."""


def canonicalize_observed_views(
    observed: Iterable[str] | Sequence[bool] | Tensor,
) -> tuple[str, ...]:
    """Return a validated selection in canonical view order.

    A mask is a length-four boolean sequence or tensor corresponding to
    :data:`CANONICAL_VIEWS`. A name iterable may arrive in any order, but cannot
    contain duplicates.
    """

    if isinstance(observed, Tensor):
        if observed.ndim != 1 or observed.shape[0] != len(CANONICAL_VIEWS):
            raise ValueError("observed view mask must have length 4")
        if observed.dtype != torch.bool:
            raise TypeError("observed view tensor mask must have boolean dtype")
        selected = tuple(
            view
            for view, included in zip(CANONICAL_VIEWS, observed.detach().cpu().tolist())
            if included
        )
    else:
        if isinstance(observed, str):
            raise TypeError("observed views must be an iterable of names, not one string")
        materialized = tuple(observed)
        if materialized and all(isinstance(value, bool) for value in materialized):
            if len(materialized) != len(CANONICAL_VIEWS):
                raise ValueError("observed view mask must have length 4")
            selected = tuple(
                view for view, included in zip(CANONICAL_VIEWS, materialized) if included
            )
        else:
            if not all(isinstance(view, str) for view in materialized):
                raise TypeError("observed view names must be strings")
            if len(materialized) != len(set(materialized)):
                raise ValueError("observed views contain a duplicate name")
            unknown = set(materialized).difference(CANONICAL_VIEWS)
            if unknown:
                raise ValueError(f"observed views contain unknown names: {sorted(unknown)}")
            selected_names = set(materialized)
            selected = tuple(view for view in CANONICAL_VIEWS if view in selected_names)

    if not selected:
        raise NoObservedViewsError("at least one observed view is required")
    return selected


def validate_logits_by_view(logits_by_view: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    """Validate and snapshot a nonempty named mapping of raw four-class logits."""

    if not isinstance(logits_by_view, Mapping):
        raise TypeError("logits_by_view must be a mapping")
    supplied = dict(logits_by_view)
    if not supplied:
        raise ValueError("logits_by_view cannot be empty")
    unknown = set(supplied).difference(CANONICAL_VIEWS)
    if unknown:
        raise ValueError(f"logits_by_view contains unknown views: {sorted(unknown)}")

    reference_batch: int | None = None
    reference_dtype: torch.dtype | None = None
    reference_device: torch.device | None = None
    for view in CANONICAL_VIEWS:
        if view not in supplied:
            continue
        logits = supplied[view]
        if not isinstance(logits, Tensor):
            raise TypeError(f"logits for {view} must be a torch.Tensor")
        if not logits.is_floating_point():
            raise TypeError(f"logits for {view} must have a floating dtype")
        if logits.ndim != 2 or logits.shape[1] != NUM_CLASSES:
            raise ValueError(f"logits for {view} must have shape [batch, four classes]")
        if not torch.isfinite(logits).all():
            raise ValueError(f"logits for {view} must be finite")
        if reference_batch is None:
            reference_batch = logits.shape[0]
            reference_dtype = logits.dtype
            reference_device = logits.device
        elif logits.shape[0] != reference_batch:
            raise ValueError("all view logits must have the same batch size")
        elif logits.dtype != reference_dtype:
            raise ValueError("all view logits must have the same dtype")
        elif logits.device != reference_device:
            raise ValueError("all view logits must be on the same device")

    ordered = {view: supplied[view] for view in CANONICAL_VIEWS if view in supplied}
    return MappingProxyType(ordered)


def _validate_representations(
    representations: Mapping[str, Tensor],
    logits_by_view: Mapping[str, Tensor],
) -> Mapping[str, Tensor]:
    if not isinstance(representations, Mapping):
        raise TypeError("frozen_representations must be a mapping")
    supplied = dict(representations)
    if set(supplied) != set(logits_by_view):
        raise ValueError("frozen representations must have the same view names as logits")

    batch_size = next(iter(logits_by_view.values())).shape[0]
    device = next(iter(logits_by_view.values())).device
    for view, representation in supplied.items():
        if not isinstance(representation, Tensor):
            raise TypeError(f"frozen representation for {view} must be a torch.Tensor")
        if not representation.is_floating_point() or representation.ndim < 2:
            raise ValueError(
                f"frozen representation for {view} must be a floating tensor with a batch axis"
            )
        if representation.shape[0] != batch_size:
            raise ValueError("frozen representations must match the logits batch size")
        if representation.device != device:
            raise ValueError("frozen representations and logits must be on the same device")
        if representation.requires_grad:
            raise ValueError("frozen representations must not require gradients")
        if not torch.isfinite(representation).all():
            raise ValueError(f"frozen representation for {view} must be finite")

    ordered = {view: supplied[view] for view in CANONICAL_VIEWS if view in supplied}
    return MappingProxyType(ordered)


@dataclass(frozen=True)
class InferenceViewInputs:
    """Available inference evidence and optional frozen features, never labels.

    Representations may be rank two or higher, so target-only and pooled-feature
    workflows do not have to fabricate token sequences.
    """

    logits_by_view: Mapping[str, Tensor]
    frozen_representations: Mapping[str, Tensor] | None = None

    def __post_init__(self) -> None:
        logits = validate_logits_by_view(self.logits_by_view)
        object.__setattr__(self, "logits_by_view", logits)
        if self.frozen_representations is not None:
            representations = _validate_representations(self.frozen_representations, logits)
            object.__setattr__(self, "frozen_representations", representations)

    @property
    def available_views(self) -> tuple[str, ...]:
        """Names with supplied evidence, always in canonical order."""

        return tuple(self.logits_by_view)
