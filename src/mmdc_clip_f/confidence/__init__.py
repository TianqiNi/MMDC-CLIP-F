"""Provenance-safe confidence estimation for MMDC-CLIP-F.

The package deliberately separates the small MV-ACN head from the frozen Stage-1
classifier. TCP artifacts therefore identify the exact Stage-1 checkpoint and
prediction stream that produced them.
"""

from .artifacts import (
    ArtifactProvenance,
    ProvenanceMismatchError,
    TCPArtifact,
    TCPRecord,
    build_tcp_artifact,
    read_tcp_artifact,
    validate_tcp_records,
    write_tcp_artifact,
)
from .metrics import ConfidenceMetrics, confidence_selection_metrics
from .model import MVACNConfig, MVACNHead, export_head_state_dict

__all__ = [
    "ArtifactProvenance",
    "ConfidenceMetrics",
    "MVACNConfig",
    "MVACNHead",
    "ProvenanceMismatchError",
    "TCPArtifact",
    "TCPRecord",
    "build_tcp_artifact",
    "confidence_selection_metrics",
    "export_head_state_dict",
    "read_tcp_artifact",
    "validate_tcp_records",
    "write_tcp_artifact",
]
