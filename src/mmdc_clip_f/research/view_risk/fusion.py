"""Legacy-compatible Dempster--Shafer fusion for named view subsets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, cast

import torch
from torch import Tensor
from torch.nn import functional as F

from .inputs import (
    CANONICAL_VIEWS,
    NUM_CLASSES,
    canonicalize_observed_views,
    validate_logits_by_view,
)


FusionPairs = tuple[tuple[str, str], tuple[str, str]]
RSNA_FUSION_PAIRS: FusionPairs = (
    ("L_CC", "L_MLO"),
    ("R_CC", "R_MLO"),
)
DDSM_FUSION_PAIRS: FusionPairs = (
    ("L_CC", "R_CC"),
    ("L_MLO", "R_MLO"),
)


class InvalidDSFusionError(ValueError):
    """Raised when legacy DS arithmetic produces an invalid numerical state."""


@dataclass(frozen=True)
class DSFusionStats:
    """Dirichlet evidence diagnostics for one fused batch."""

    evidence: Tensor
    strength: Tensor
    uncertainty: Tensor
    conflicts: tuple[Tensor, ...]


@dataclass(frozen=True)
class SubsetFusionResult:
    """Classifier outputs and evidence state for one named view subset."""

    scores: Tensor
    fused_alpha: Tensor
    observed_views: tuple[str, ...]
    fusion_pairs: FusionPairs
    stats: DSFusionStats


def validate_fusion_pairs(
    fusion_pairs: Sequence[Sequence[str]],
) -> FusionPairs:
    """Validate a two-branch tree and preserve its configured operation order."""

    try:
        pairs = tuple(tuple(pair) for pair in fusion_pairs)
    except TypeError as error:
        raise TypeError(
            "fusion_pairs must partition canonical views into two ordered pairs"
        ) from error
    flattened = tuple(view for pair in pairs for view in pair)
    if (
        len(pairs) != 2
        or any(len(pair) != 2 for pair in pairs)
        or len(flattened) != len(CANONICAL_VIEWS)
        or set(flattened) != set(CANONICAL_VIEWS)
    ):
        raise ValueError("fusion_pairs must partition canonical views into two ordered pairs")
    return cast(FusionPairs, pairs)


def _require_finite(value: Tensor, stage: str) -> Tensor:
    if not torch.isfinite(value).all():
        raise InvalidDSFusionError(f"legacy DS fusion produced a nonfinite {stage}")
    return value


def _require_positive(value: Tensor, stage: str) -> Tensor:
    _require_finite(value, stage)
    if not (value > 0).all():
        raise InvalidDSFusionError(f"legacy DS fusion produced an invalid non-positive {stage}")
    return value


def _combine_two(alpha1: Tensor, alpha2: Tensor) -> tuple[Tensor, Tensor]:
    """Apply unchanged classifier arithmetic and return alpha plus conflict."""

    strength1 = _require_positive(alpha1.sum(dim=1, keepdim=True), "source strength")
    strength2 = _require_positive(alpha2.sum(dim=1, keepdim=True), "source strength")
    evidence1 = alpha1 - 1
    evidence2 = alpha2 - 1
    belief1 = _require_finite(evidence1 / strength1.expand_as(evidence1), "source belief")
    belief2 = _require_finite(evidence2 / strength2.expand_as(evidence2), "source belief")
    uncertainty1 = _require_positive(NUM_CLASSES / strength1, "source uncertainty")
    uncertainty2 = _require_positive(NUM_CLASSES / strength2, "source uncertainty")

    products = _require_finite(
        torch.bmm(belief1.unsqueeze(2), belief2.unsqueeze(1)),
        "belief product",
    )
    agreement = torch.diagonal(products, dim1=-2, dim2=-1).sum(-1)
    conflict = _require_finite(products.sum(dim=(1, 2)) - agreement, "conflict")
    denominator = _require_positive((1 - conflict).unsqueeze(1), "conflict denominator")
    fused_belief = _require_finite(
        (
            belief1 * belief2
            + belief1 * uncertainty2.expand_as(belief1)
            + belief2 * uncertainty1.expand_as(belief2)
        )
        / denominator,
        "fused belief",
    )
    fused_uncertainty = _require_positive(
        uncertainty1 * uncertainty2 / denominator,
        "fused uncertainty",
    )
    fused_strength = _require_positive(NUM_CLASSES / fused_uncertainty, "fused strength")
    fused_alpha = _require_finite(
        fused_belief * fused_strength.expand_as(fused_belief) + 1,
        "fused alpha",
    )
    return fused_alpha, conflict


def fuse_view_logits(
    logits_by_view: Mapping[str, Tensor],
    observed: Iterable[str] | Sequence[bool] | Tensor,
    *,
    fusion_pairs: Sequence[Sequence[str]] = RSNA_FUSION_PAIRS,
) -> SubsetFusionResult:
    """Fuse a nonempty named subset of raw batch-by-four per-view logits.

    The configured two-branch tree is pruned without reordering. Missing views
    contribute no computation or gradient. The default is the legacy RSNA tree;
    callers must select the DDSM tree for DDSM-derived logits.
    """

    pairs = validate_fusion_pairs(fusion_pairs)
    logits = validate_logits_by_view(logits_by_view)
    observed_views = canonicalize_observed_views(observed)
    missing = set(observed_views).difference(logits)
    if missing:
        raise ValueError(f"logits_by_view is missing observed views: {sorted(missing)}")

    alpha_by_view = {
        view: _require_finite(F.softplus(logits[view]) + 1, f"source alpha for {view}")
        for view in observed_views
    }
    conflicts: list[Tensor] = []
    group_alphas: list[Tensor] = []
    for pair in pairs:
        present = [alpha_by_view[view] for view in pair if view in alpha_by_view]
        if len(present) == 2:
            combined, conflict = _combine_two(present[0], present[1])
            conflicts.append(conflict)
            group_alphas.append(combined)
        elif present:
            group_alphas.append(present[0])

    if len(group_alphas) == 2:
        fused_alpha, conflict = _combine_two(group_alphas[0], group_alphas[1])
        conflicts.append(conflict)
    else:
        fused_alpha = group_alphas[0]

    fused_alpha = _require_finite(fused_alpha, "final fused alpha")
    strength = _require_positive(fused_alpha.sum(dim=1, keepdim=True), "final strength")
    evidence = _require_finite(fused_alpha - 1, "final evidence")
    uncertainty = _require_positive(NUM_CLASSES / strength, "final uncertainty")
    scores = _require_finite(F.softplus(fused_alpha), "classifier scores")
    stats = DSFusionStats(
        evidence=evidence,
        strength=strength,
        uncertainty=uncertainty,
        conflicts=tuple(conflicts),
    )
    return SubsetFusionResult(
        scores=scores,
        fused_alpha=fused_alpha,
        observed_views=observed_views,
        fusion_pairs=pairs,
        stats=stats,
    )
