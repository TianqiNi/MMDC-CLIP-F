"""Foundations for named-view intervention supervision."""

from .fusion import DSFusionStats, SubsetFusionResult, fuse_view_logits
from .inputs import CANONICAL_VIEWS, InferenceViewInputs, NoObservedViewsError
from .targets import InterventionTargets, build_intervention_targets

__all__ = [
    "CANONICAL_VIEWS",
    "DSFusionStats",
    "InferenceViewInputs",
    "InterventionTargets",
    "NoObservedViewsError",
    "SubsetFusionResult",
    "build_intervention_targets",
    "fuse_view_logits",
]
