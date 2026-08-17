"""Explicit metric definitions used by classifier evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int = 4,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if labels.size == 0:
        raise ValueError("labels must not be empty")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if np.any((labels < 0) | (labels >= num_classes)):
        raise ValueError(f"labels must be in [0, {num_classes - 1}]")
    if probabilities.shape != (labels.size, num_classes):
        raise ValueError(f"probabilities must have shape ({labels.size}, {num_classes})")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain non-finite values")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("probability rows must sum to 1")
    predictions = probabilities.argmax(axis=1)
    classes = np.arange(num_classes)
    matrix = confusion_matrix(labels, predictions, labels=classes)
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        labels, predictions, labels=classes, zero_division=0
    )
    result: dict[str, Any] = {
        "n": int(labels.size),
        "accuracy": float((labels == predictions).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=classes, average="macro", zero_division=0)
        ),
        "confusion_matrix": matrix.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "per_class_support": support.astype(int).tolist(),
        "auc_definition": "multiclass one-vs-rest over softmax probabilities",
        "macro_ovr_auc": None,
        "weighted_ovr_auc": None,
    }
    if set(labels.tolist()) == set(classes.tolist()):
        result["macro_ovr_auc"] = float(
            roc_auc_score(labels, probabilities, labels=classes, multi_class="ovr", average="macro")
        )
        result["weighted_ovr_auc"] = float(
            roc_auc_score(
                labels, probabilities, labels=classes, multi_class="ovr", average="weighted"
            )
        )
    return result
