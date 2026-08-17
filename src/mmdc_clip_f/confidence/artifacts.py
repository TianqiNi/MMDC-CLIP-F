"""Strict TCP artifact schema and Stage-1 provenance checks."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TCP_FIELDS = (
    "stage1_checkpoint_sha256",
    "stage1_prediction_sha256",
    "manifest_sha256",
    "split",
    "sample_index",
    "label",
    "prediction",
    "correct",
    "tcp",
)


class ProvenanceMismatchError(ValueError):
    """Raised when results do not originate from the same Stage-1 predictions."""


def _validate_sha256(value: str, field: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256 digest")


@dataclass(frozen=True)
class ArtifactProvenance:
    """Identity shared by all records derived from one classifier inference run."""

    stage1_checkpoint_sha256: str
    stage1_prediction_sha256: str
    manifest_sha256: str
    split: str

    def __post_init__(self) -> None:
        _validate_sha256(self.stage1_checkpoint_sha256, "stage1_checkpoint_sha256")
        _validate_sha256(self.stage1_prediction_sha256, "stage1_prediction_sha256")
        _validate_sha256(self.manifest_sha256, "manifest_sha256")
        if not isinstance(self.split, str) or not self.split.strip():
            raise ValueError("split must be a non-empty string")


@dataclass(frozen=True)
class TCPRecord:
    """One sample from a true-class-probability (TCP) artifact."""

    stage1_checkpoint_sha256: str
    stage1_prediction_sha256: str
    manifest_sha256: str
    split: str
    sample_index: int
    label: int
    prediction: int
    correct: bool
    tcp: float

    def __post_init__(self) -> None:
        _ = self.provenance
        if isinstance(self.sample_index, bool) or int(self.sample_index) != self.sample_index:
            raise ValueError("sample_index must be an integer")
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        for field, value in (("label", self.label), ("prediction", self.prediction)):
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if not isinstance(self.correct, (bool, np.bool_)):
            raise ValueError("correct must be a boolean")
        if bool(self.correct) != (self.label == self.prediction):
            raise ValueError("correct must equal (label == prediction)")
        if not math.isfinite(float(self.tcp)) or not 0.0 <= float(self.tcp) <= 1.0:
            raise ValueError("tcp must be finite and in [0, 1]")

    @property
    def provenance(self) -> ArtifactProvenance:
        return ArtifactProvenance(
            stage1_checkpoint_sha256=self.stage1_checkpoint_sha256,
            stage1_prediction_sha256=self.stage1_prediction_sha256,
            manifest_sha256=self.manifest_sha256,
            split=self.split,
        )

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "TCPRecord":
        missing = set(_TCP_FIELDS) - set(row)
        if missing:
            raise ValueError(f"TCP record is missing fields: {sorted(missing)}")
        unexpected = set(row) - set(_TCP_FIELDS)
        if unexpected:
            raise ValueError(f"TCP record has unexpected fields: {sorted(unexpected)}")
        correct_value = row["correct"]
        if isinstance(correct_value, str):
            normalized = correct_value.strip().lower()
            if normalized not in {"true", "false", "1", "0"}:
                raise ValueError(f"invalid correct value: {correct_value!r}")
            correct_value = normalized in {"true", "1"}
        return cls(
            stage1_checkpoint_sha256=str(row["stage1_checkpoint_sha256"]),
            stage1_prediction_sha256=str(row["stage1_prediction_sha256"]),
            manifest_sha256=str(row["manifest_sha256"]),
            split=str(row["split"]),
            sample_index=int(row["sample_index"]),
            label=int(row["label"]),
            prediction=int(row["prediction"]),
            correct=bool(correct_value),
            tcp=float(row["tcp"]),
        )


def _prediction_sha256(index_prediction_pairs: Iterable[tuple[int, int]]) -> str:
    canonical = json.dumps(
        sorted(index_prediction_pairs),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _integer_vector(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain integers") from error
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} must contain integers")
    return numeric.astype(np.int64)


def validate_tcp_records(
    records: Iterable[TCPRecord],
    *,
    expected_sample_indexes: Iterable[int] | None = None,
) -> tuple[TCPRecord, ...]:
    """Validate completeness, uniqueness, and provenance, returning index order.

    With no explicit expected indexes, a complete artifact must use the canonical
    contiguous range ``0..n-1``. Passing an expected set supports a predefined subset.
    """

    materialized = tuple(records)
    if not materialized:
        raise ValueError("TCP artifact cannot be empty")
    if not all(isinstance(record, TCPRecord) for record in materialized):
        raise TypeError("records must contain TCPRecord instances")

    indexes = [record.sample_index for record in materialized]
    if len(indexes) != len(set(indexes)):
        duplicates = sorted({index for index in indexes if indexes.count(index) > 1})
        raise ValueError(f"duplicate sample indexes: {duplicates}")

    if expected_sample_indexes is None:
        expected = set(range(len(materialized)))
    else:
        expected_list = list(expected_sample_indexes)
        if len(expected_list) != len(set(expected_list)):
            raise ValueError("expected_sample_indexes contains duplicates")
        expected = set(expected_list)
    actual = set(indexes)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(f"sample index mismatch; missing={missing}, unexpected={unexpected}")

    provenance = materialized[0].provenance
    for record in materialized[1:]:
        if record.provenance != provenance:
            raise ProvenanceMismatchError("TCP records contain mixed provenance")

    actual_prediction_hash = _prediction_sha256(
        (record.sample_index, record.prediction) for record in materialized
    )
    if actual_prediction_hash != provenance.stage1_prediction_sha256:
        raise ProvenanceMismatchError(
            "stage1_prediction_sha256 does not match the artifact predictions"
        )
    return tuple(sorted(materialized, key=lambda record: record.sample_index))


@dataclass(frozen=True)
class TCPArtifact:
    """Validated collection of TCP records."""

    records: tuple[TCPRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", validate_tcp_records(self.records))

    @property
    def provenance(self) -> ArtifactProvenance:
        return self.records[0].provenance

    def __len__(self) -> int:
        return len(self.records)


def build_tcp_artifact(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    split: str,
    stage1_checkpoint_sha256: str,
    manifest_sha256: str,
    predictions: Sequence[int] | np.ndarray | None = None,
    sample_indexes: Sequence[int] | np.ndarray | None = None,
) -> TCPArtifact:
    """Create a strict TCP artifact from Stage-1 class probabilities."""

    _validate_sha256(stage1_checkpoint_sha256, "stage1_checkpoint_sha256")
    _validate_sha256(manifest_sha256, "manifest_sha256")
    probability_array = np.asarray(probabilities, dtype=np.float64)
    label_array = _integer_vector(labels, "labels")
    if probability_array.ndim != 2 or probability_array.shape[1] < 2:
        raise ValueError("probabilities must have shape [samples, classes>=2]")
    if label_array.ndim != 1 or len(label_array) != len(probability_array):
        raise ValueError("labels must be one-dimensional and match probabilities")
    if len(label_array) == 0:
        raise ValueError("cannot build an empty TCP artifact")
    if not np.isfinite(probability_array).all():
        raise ValueError("probabilities must be finite")
    if (probability_array < 0).any() or (probability_array > 1).any():
        raise ValueError("probabilities must be in [0, 1]")
    if not np.allclose(probability_array.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("each probability row must sum to one")
    if (label_array < 0).any() or (label_array >= probability_array.shape[1]).any():
        raise ValueError("labels contain an out-of-range class index")

    inferred_predictions = probability_array.argmax(axis=1).astype(np.int64)
    if predictions is None:
        prediction_array = inferred_predictions
    else:
        prediction_array = _integer_vector(predictions, "predictions")
        if prediction_array.shape != label_array.shape:
            raise ValueError("predictions must have the same shape as labels")
        if not np.array_equal(prediction_array, inferred_predictions):
            raise ValueError("predictions must equal argmax(probabilities)")

    if sample_indexes is None:
        index_array = np.arange(len(label_array), dtype=np.int64)
    else:
        index_array = _integer_vector(sample_indexes, "sample_indexes")
        if index_array.shape != label_array.shape:
            raise ValueError("sample_indexes must have the same shape as labels")
    if not np.array_equal(index_array, np.arange(len(index_array), dtype=np.int64)):
        raise ValueError("sample_indexes must be in canonical contiguous order 0..n-1")

    prediction_hash = _prediction_sha256(
        zip(index_array.tolist(), prediction_array.tolist(), strict=True)
    )
    records = tuple(
        TCPRecord(
            stage1_checkpoint_sha256=stage1_checkpoint_sha256,
            stage1_prediction_sha256=prediction_hash,
            manifest_sha256=manifest_sha256,
            split=split,
            sample_index=int(sample_index),
            label=int(label),
            prediction=int(prediction),
            correct=bool(label == prediction),
            tcp=float(probability_array[row_index, label]),
        )
        for row_index, (sample_index, label, prediction) in enumerate(
            zip(index_array, label_array, prediction_array, strict=True)
        )
    )
    return TCPArtifact(records)


def write_tcp_artifact(artifact: TCPArtifact, path: str | Path) -> None:
    """Write a validated artifact as ``.jsonl`` or ``.csv``."""

    if not isinstance(artifact, TCPArtifact):
        raise TypeError("artifact must be a TCPArtifact")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix == ".jsonl":
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            for record in artifact.records:
                handle.write(json.dumps(record.to_mapping(), sort_keys=True) + "\n")
        return
    if target.suffix == ".csv":
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_TCP_FIELDS)
            writer.writeheader()
            writer.writerows(record.to_mapping() for record in artifact.records)
        return
    raise ValueError("TCP artifact path must end in .jsonl or .csv")


def read_tcp_artifact(path: str | Path) -> TCPArtifact:
    """Read and fully validate a ``.jsonl`` or ``.csv`` TCP artifact."""

    source = Path(path)
    if source.suffix == ".jsonl":
        rows: list[Mapping[str, object]] = []
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"blank JSONL record at line {line_number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL line {line_number} is not an object")
                rows.append(value)
    elif source.suffix == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError("TCP artifact path must end in .jsonl or .csv")
    return TCPArtifact(tuple(TCPRecord.from_mapping(row) for row in rows))
