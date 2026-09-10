"""Foundations for named-view intervention supervision."""

from .fusion import (
    DDSM_FUSION_PAIRS,
    RSNA_FUSION_PAIRS,
    DSFusionStats,
    FusionPairs,
    InvalidDSFusionError,
    SubsetFusionResult,
    fuse_view_logits,
    validate_fusion_pairs,
)
from .inputs import CANONICAL_VIEWS, InferenceViewInputs, NoObservedViewsError
from .targets import InterventionTargets, build_intervention_targets

__all__ = [
    "CANONICAL_VIEWS",
    "DDSM_FUSION_PAIRS",
    "DSFusionStats",
    "FusionPairs",
    "InferenceViewInputs",
    "InterventionTargets",
    "InvalidDSFusionError",
    "NoObservedViewsError",
    "RSNA_FUSION_PAIRS",
    "SubsetFusionResult",
    "build_intervention_targets",
    "fuse_view_logits",
    "validate_fusion_pairs",
]
