"""Dataset readers and validation for four-view mammography exams.

No split manifests or medical images are distributed with this package.  The
loaders consume paths supplied by an authorized local dataset installation.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .provenance import sha256_file


VIEWS = ("L_CC", "L_MLO", "R_CC", "R_MLO")
RSNA_DENSITY = {"A": 0, "B": 1, "C": 2, "D": 3}


class DataValidationError(ValueError):
    """Raised when a manifest or local dataset is inconsistent."""


@dataclass(frozen=True)
class ExamRecord:
    sample_index: int
    group_id: str
    label: int
    view_ids: Mapping[str, str]


def _validate_label(label: int, location: str) -> int:
    if label not in range(4):
        raise DataValidationError(f"Density label at {location} must be in [0, 3], got {label}")
    return label


def _validate_view_ids(view_ids: Mapping[str, str], location: str) -> None:
    if any(not value for value in view_ids.values()):
        raise DataValidationError(f"Missing view identifier at {location}")
    if len(set(view_ids.values())) != len(VIEWS):
        raise DataValidationError(f"View identifiers must be unique within an exam at {location}")


def read_ddsm_manifest(path: str | Path) -> list[ExamRecord]:
    manifest = Path(path)
    records: list[ExamRecord] = []
    seen_groups: set[str] = set()
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row_number, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) != 6:
                raise DataValidationError(
                    f"{manifest}:{row_number} must contain exam_id, four view paths, and density"
                )
            exam_id, lcc, rcc, lmlo, rmlo, raw_label = (value.strip() for value in row)
            if not exam_id:
                raise DataValidationError(f"Missing exam_id at {manifest}:{row_number}")
            if exam_id in seen_groups:
                raise DataValidationError(f"Duplicate group identifier at {manifest}:{row_number}")
            view_ids = {"L_CC": lcc, "L_MLO": lmlo, "R_CC": rcc, "R_MLO": rmlo}
            _validate_view_ids(view_ids, f"{manifest}:{row_number}")
            try:
                parsed_label = int(raw_label)
            except ValueError as exc:
                raise DataValidationError(f"Invalid density at {manifest}:{row_number}") from exc
            label = _validate_label(parsed_label, f"{manifest}:{row_number}")
            records.append(
                ExamRecord(
                    sample_index=len(records),
                    group_id=exam_id,
                    label=label,
                    view_ids=view_ids,
                )
            )
            seen_groups.add(exam_id)
    if not records:
        raise DataValidationError(f"Manifest is empty: {manifest}")
    return records


def read_rsna_manifest(path: str | Path) -> list[ExamRecord]:
    manifest = Path(path)
    records: list[ExamRecord] = []
    seen_groups: set[str] = set()
    required = {"patient_id", "density", *VIEWS}
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise DataValidationError(f"{manifest} is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            if not all((row.get(view) or "").strip() for view in VIEWS):
                raise DataValidationError(f"Missing view identifier at {manifest}:{row_number}")
            density = (row.get("density") or "").strip().upper()
            if density not in RSNA_DENSITY:
                raise DataValidationError(
                    f"Invalid RSNA density at {manifest}:{row_number}: {density!r}"
                )
            patient_id = (row.get("patient_id") or "").strip()
            if not patient_id:
                raise DataValidationError(f"Missing patient_id at {manifest}:{row_number}")
            if patient_id in seen_groups:
                raise DataValidationError(f"Duplicate group identifier at {manifest}:{row_number}")
            view_ids = {view: (row[view] or "").strip() for view in VIEWS}
            _validate_view_ids(view_ids, f"{manifest}:{row_number}")
            records.append(
                ExamRecord(
                    sample_index=len(records),
                    group_id=patient_id,
                    label=RSNA_DENSITY[density],
                    view_ids=view_ids,
                )
            )
            seen_groups.add(patient_id)
    if not records:
        raise DataValidationError(f"Manifest is empty: {manifest}")
    return records


def _load_ddsm_image(root: Path, record: ExamRecord, view: str) -> Image.Image:
    path = root / record.view_ids[view]
    with Image.open(path) as image:
        return image.convert("RGB")


def _load_rsna_image(root: Path, record: ExamRecord, view: str) -> Image.Image:
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("RSNA DICOM loading requires pydicom and a JPEG2000 decoder") from exc
    path = root / record.group_id / f"{record.view_ids[view]}.dcm"
    dcm = pydicom.dcmread(path)
    pixels = dcm.pixel_array.astype(np.float32)
    pixels -= pixels.min()
    pixels /= pixels.max() + 1e-5
    pixels = (pixels * 255).astype(np.uint8)
    return Image.fromarray(pixels).convert("RGB")


class FourViewDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        dataset_name: str,
        manifest_path: str | Path,
        image_root: str | Path,
        transform: Callable[[Image.Image], Any] | None,
        input_order: Iterable[str] = VIEWS,
    ) -> None:
        self.dataset_name = dataset_name.upper()
        self.manifest_path = Path(manifest_path).resolve()
        self.image_root = Path(image_root).resolve()
        self.transform = transform
        self.input_order = tuple(input_order)
        if len(self.input_order) != 4 or set(self.input_order) != set(VIEWS):
            raise DataValidationError(
                "input_order must contain all four canonical views exactly once"
            )
        if self.dataset_name == "DDSM":
            self.records = read_ddsm_manifest(self.manifest_path)
            self._loader = _load_ddsm_image
        elif self.dataset_name == "RSNA":
            self.records = read_rsna_manifest(self.manifest_path)
            self._loader = _load_rsna_image
        else:
            raise DataValidationError("dataset_name must be DDSM or RSNA")
        self.manifest_sha256 = sha256_file(self.manifest_path)

    def __len__(self) -> int:
        return len(self.records)

    def view_path(self, record: ExamRecord, view: str) -> Path:
        if self.dataset_name == "DDSM":
            return self.image_root / record.view_ids[view]
        return self.image_root / record.group_id / f"{record.view_ids[view]}.dcm"

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        views: dict[str, Any] = {}
        for view in self.input_order:
            image = self._loader(self.image_root, record, view)
            views[view] = self.transform(image) if self.transform else image
        return {"views": views, "label": record.label, "sample_index": record.sample_index}

    def validate_files(self) -> list[str]:
        missing: list[str] = []
        for record in self.records:
            for view in self.input_order:
                path = self.view_path(record, view)
                if not path.is_file():
                    missing.append(str(path))
        return missing


def validate_split_isolation(datasets: Mapping[str, FourViewDataset]) -> dict[str, Any]:
    group_counts = {
        split: Counter(record.group_id for record in dataset.records)
        for split, dataset in datasets.items()
    }
    groups: dict[str, set[str]] = {split: set(counts) for split, counts in group_counts.items()}
    overlaps: dict[str, list[str]] = {}
    names = sorted(groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            shared = groups[left].intersection(groups[right])
            if shared:
                # IDs are not returned; only counts are safe for logs/reports.
                overlaps[f"{left}:{right}"] = [f"{len(shared)} shared group(s)"]
    return {
        "counts": {split: len(dataset) for split, dataset in datasets.items()},
        "class_counts": {
            split: {
                str(label): sum(record.label == label for record in dataset.records)
                for label in range(4)
            }
            for split, dataset in datasets.items()
        },
        "manifest_sha256": {split: dataset.manifest_sha256 for split, dataset in datasets.items()},
        # Only aggregate counts are reported; group identifiers remain private.
        "duplicate_group_counts": {
            split: sum(count > 1 for count in counts.values())
            for split, counts in group_counts.items()
        },
        "overlaps": overlaps,
    }


def build_transforms(image_size: int, augmentation: Mapping[str, Any] | str | None):
    from torchvision import transforms

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_steps: list[Any] = []
    if augmentation == "randaugment":
        augmentation = {"name": "randaugment"}
    if isinstance(augmentation, Mapping):
        name = str(augmentation.get("name", "none")).lower()
        if name == "randaugment":
            train_steps.append(
                transforms.RandAugment(
                    num_ops=int(augmentation.get("num_ops", 3)),
                    magnitude=int(augmentation.get("magnitude", 9)),
                    num_magnitude_bins=int(augmentation.get("num_magnitude_bins", 31)),
                )
            )
        elif name not in {"none", "off"}:
            raise DataValidationError(f"Unsupported augmentation: {name}")
    train_steps.extend(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize]
    )
    eval_steps = [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize]
    return transforms.Compose(train_steps), transforms.Compose(eval_steps)
