"""Confidence metrics used to select an MV-ACN checkpoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ConfidenceMetrics:
    """Metrics treating confidence as a predictor of classifier correctness."""

    n_samples: int
    aupr: float | None
    aurc: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(array.astype(np.float64)).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _probability_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = _vector(values, name).astype(np.float64)
    if (array < 0).any() or (array > 1).any():
        raise ValueError(f"{name} must be in [0, 1]")
    return array


def _average_precision(correct: np.ndarray, confidence: np.ndarray) -> float:
    """Non-interpolated average precision with score ties evaluated as groups."""

    order = np.argsort(-confidence, kind="stable")
    sorted_confidence = confidence[order]
    sorted_correct = correct[order]
    positives = int(sorted_correct.sum())
    cumulative_positive = 0
    cumulative_count = 0
    average_precision = 0.0
    index = 0
    while index < len(sorted_correct):
        end = index + 1
        while end < len(sorted_correct) and sorted_confidence[end] == sorted_confidence[index]:
            end += 1
        group_positive = int(sorted_correct[index:end].sum())
        cumulative_positive += group_positive
        cumulative_count = end
        precision = cumulative_positive / cumulative_count
        average_precision += (group_positive / positives) * precision
        index = end
    return float(average_precision)


def confidence_selection_metrics(
    correct: Sequence[int] | np.ndarray,
    confidence: Sequence[float] | np.ndarray,
) -> ConfidenceMetrics:
    """Compute AUPR and AURC for classifier correctness prediction."""

    correct_values = _vector(correct, "correct").astype(np.float64)
    if not np.isin(correct_values, [0.0, 1.0]).all():
        raise ValueError("correct must contain only 0 and 1")
    correct_array = correct_values.astype(np.int64)
    confidence_array = _probability_vector(confidence, "confidence")
    if len(correct_array) != len(confidence_array):
        raise ValueError("correct and confidence must have the same length")

    order = np.argsort(-confidence_array, kind="stable")
    cumulative_risk = np.cumsum(1 - correct_array[order]) / np.arange(1, len(correct_array) + 1)
    aupr = (
        _average_precision(correct_array, confidence_array)
        if len(np.unique(correct_array)) == 2
        else None
    )
    return ConfidenceMetrics(
        n_samples=len(correct_array),
        aupr=aupr,
        aurc=float(cumulative_risk.mean()),
    )
