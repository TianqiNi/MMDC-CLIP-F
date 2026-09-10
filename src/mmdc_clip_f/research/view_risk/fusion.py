"""Legacy-compatible Dempster--Shafer fusion for named view subsets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .inputs import (
    NUM_CLASSES,
    canonicalize_observed_views,
    validate_logits_by_view,
)


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
    stats: DSFusionStats


def _combine_two(alpha1: Tensor, alpha2: Tensor) -> tuple[Tensor, Tensor]:
    """Apply the classifier's two-source rule and return alpha plus conflict."""

    strength1 = alpha1.sum(dim=1, keepdim=True)
    strength2 = alpha2.sum(dim=1, keepdim=True)
    evidence1 = alpha1 - 1
    evidence2 = alpha2 - 1
    belief1 = evidence1 / strength1.expand_as(evidence1)
    belief2 = evidence2 / strength2.expand_as(evidence2)
    uncertainty1 = NUM_CLASSES / strength1
    uncertainty2 = NUM_CLASSES / strength2

    products = torch.bmm(belief1.unsqueeze(2), belief2.unsqueeze(1))
    agreement = torch.diagonal(products, dim1=-2, dim2=-1).sum(-1)
    conflict = products.sum(dim=(1, 2)) - agreement
    denominator = (1 - conflict).unsqueeze(1)
    fused_belief = (
        belief1 * belief2
        + belief1 * uncertainty2.expand_as(belief1)
        + belief2 * uncertainty1.expand_as(belief2)
    ) / denominator
    fused_uncertainty = uncertainty1 * uncertainty2 / denominator
    fused_strength = NUM_CLASSES / fused_uncertainty
    fused_alpha = fused_belief * fused_strength.expand_as(fused_belief) + 1
    return fused_alpha, conflict


def fuse_view_logits(
    logits_by_view: Mapping[str, Tensor],
    observed: Iterable[str] | Sequence[bool] | Tensor,
) -> SubsetFusionResult:
    """Fuse a nonempty named subset of raw ``[batch, 4]`` per-view logits.

    Names and masks are canonicalized before use. Missing views contribute no
    computation or gradient; under Dempster's rule this is equivalent to vacuous
    zero evidence. The four-view path preserves the legacy pair/pair fusion order.
    """

    logits = validate_logits_by_view(logits_by_view)
    observed_views = canonicalize_observed_views(observed)
    missing = set(observed_views).difference(logits)
    if missing:
        raise ValueError(f"logits_by_view is missing observed views: {sorted(missing)}")

    alpha_by_view = {view: F.softplus(logits[view]) + 1 for view in observed_views}
    conflicts: list[Tensor] = []
    group_alphas: list[Tensor] = []
    for pair in (("L_CC", "L_MLO"), ("R_CC", "R_MLO")):
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

    strength = fused_alpha.sum(dim=1, keepdim=True)
    stats = DSFusionStats(
        evidence=fused_alpha - 1,
        strength=strength,
        uncertainty=NUM_CLASSES / strength,
        conflicts=tuple(conflicts),
    )
    return SubsetFusionResult(
        scores=F.softplus(fused_alpha),
        fused_alpha=fused_alpha,
        observed_views=observed_views,
        stats=stats,
    )
