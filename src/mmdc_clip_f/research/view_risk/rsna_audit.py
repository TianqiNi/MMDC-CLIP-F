"""Bounded real-data readiness audit for the local RSNA-SMBC installation.

The locked test CSV is handled by a deliberately separate identity projection.
It never becomes an ``ExamRecord`` or ``PrivateExamRecord`` and its outcomes and
image identifiers are not accessed.  All identifier-bearing outputs are private;
the public result contains aggregate counts and aggregate artifact digests only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from mmdc_clip_f.data import RSNA_DENSITY
from mmdc_clip_f.provenance import sha256_file

from .cache import _require_external_or_ignored_destination
from .inputs import CANONICAL_VIEWS
from .production import default_private_image_reader
from .roles import (
    NON_TEST_ROLES,
    LockedPatientIdentityDenylist,
    PatientMappingDeclaration,
    PrivateExamRecord,
    Role,
    ViewReference,
    assign_patient_roles,
    patient_identity_digest,
    save_private_manifest,
    validate_inventory,
)
from .training import audit_real_data_readiness, save_readiness_audit


PUBLIC_SCHEMA_VERSION = "view-risk-p5a-rsna-audit/v1"
PRIVATE_RUN_SCHEMA_VERSION = "view-risk-p5a-rsna-private-run/v1"
DENYLIST_FILE_SCHEMA_VERSION = "view-risk-p5a-rsna-identity-projection/v1"
MAPPING_FILE_SCHEMA_VERSION = "view-risk-p5a-rsna-patient-mapping/v1"
IMAGE_JOURNAL_SCHEMA_VERSION = "view-risk-p5a-rsna-image-journal/v1"
IMAGE_AUDIT_SEMANTICS_VERSION = "view-risk-production-decoded-rgb/v1"
PROGRESS_SCHEMA_VERSION = "view-risk-p5a-rsna-progress/v1"
DATASET_NAMESPACE = "RSNA"
EXPECTED_HEADER = ("patient_id", "density", *CANONICAL_VIEWS)


class AuditInterrupted(RuntimeError):
    """Intentional test-only stop after durable image-audit progress."""


@dataclass(frozen=True)
class _MetadataScan:
    records: tuple[PrivateExamRecord, ...]
    row_count: int
    density_counts: Mapping[int, int]
    problems: Mapping[str, int]
    patient_count: int
    view_count: int


@dataclass(frozen=True)
class _ImageScan:
    records: tuple[PrivateExamRecord, ...]
    expected_count: int
    readable_count: int
    decoded_count: int
    missing_count: int
    unreadable_count: int
    decode_failure_count: int
    byte_group_count: int
    byte_extra_count: int
    pixel_group_count: int
    pixel_extra_count: int
    byte_collection_sha256: str | None
    pixel_collection_sha256: str | None

    @property
    def blocked(self) -> bool:
        return any(
            (
                self.missing_count,
                self.unreadable_count,
                self.decode_failure_count,
                self.byte_extra_count,
                self.pixel_extra_count,
            )
        )


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _hash_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _write_json_exclusive(path: Path, value: object, *, private: bool) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to clobber existing audit artifact: {path.name}") from exc
    if private:
        path.chmod(0o600)


def _read_json_object(path: Path, *, description: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is unreadable or corrupt") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} is unreadable or corrupt")
    return value


def _ensure_private_document(
    path: Path, expected: Mapping[str, object], *, resume: bool, description: str
) -> None:
    if resume:
        if not path.is_file():
            raise ValueError(f"resume {description} is missing")
        if _read_json_object(path, description=description) != dict(expected):
            raise ValueError(f"resume {description} binding changed")
        return
    _write_json_exclusive(path, expected, private=True)


def _progress_path(public_report: Path) -> Path:
    return public_report.with_name(f"{public_report.stem}-progress.json")


def _write_progress(
    path: Path,
    value: Mapping[str, object],
    *,
    resume: bool,
    journal_header_sha256: str,
) -> None:
    if path.exists():
        if not resume:
            raise FileExistsError("refusing to clobber existing public audit progress")
        prior = _read_json_object(path, description="public audit progress")
        if (
            prior.get("schema_version") != PROGRESS_SCHEMA_VERSION
            or prior.get("journal_header_sha256") != journal_header_sha256
        ):
            raise ValueError("public audit progress binding disagrees with the private journal")
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        _write_json_exclusive(temporary, value, private=False)
        os.replace(temporary, path)
        return
    _write_json_exclusive(path, value, private=False)


def _safe_private_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    try:
        _require_external_or_ignored_destination(root / "audit-probe")
    except ValueError as exc:
        raise ValueError("private output must be external to git or explicitly ignored") from exc
    return root


def _validate_run_name(run_name: str) -> str:
    candidate = Path(run_name)
    if (
        not isinstance(run_name, str)
        or not run_name
        or candidate.is_absolute()
        or candidate.name != run_name
        or len(candidate.parts) != 1
        or run_name in {".", ".."}
    ):
        raise ValueError("run_name must be one non-empty path component")
    return run_name


def _project_identity_rows(
    header: Sequence[str], rows: Iterable[Sequence[str]]
) -> tuple[str, ...]:
    """Project only patient IDs from locked rows.

    The row implementation may make every non-identity cell inaccessible; this
    function only indexes the ``patient_id`` cell.  Row length is structural
    schema information, not an outcome value.
    """

    normalized_header = tuple(header)
    if normalized_header != EXPECTED_HEADER:
        raise ValueError("locked test CSV does not have the official RSNA schema")
    patient_index = normalized_header.index("patient_id")
    result: list[str] = []
    for row in rows:
        if len(row) != len(normalized_header):
            raise ValueError("locked test CSV contains a malformed row")
        patient_id = row[patient_index].strip()
        if not patient_id:
            raise ValueError("locked test CSV contains a missing patient identity")
        result.append(patient_id)
    if not result:
        raise ValueError("locked test CSV is empty")
    return tuple(result)


def _read_locked_identities(path: Path) -> tuple[str, ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, ())
            return _project_identity_rows(header, reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError("locked test identity projection could not be read") from exc


def _empty_problems() -> dict[str, int]:
    return {
        "schema_error_count": 0,
        "malformed_row_count": 0,
        "missing_patient_id_count": 0,
        "missing_view_identifier_count": 0,
        "invalid_density_count": 0,
        "duplicate_exam_or_patient_count": 0,
        "repeated_view_identifier_count": 0,
        "unsafe_identifier_count": 0,
    }


def _identifier_is_safe_path_component(value: str) -> bool:
    component = Path(value)
    return (
        bool(value)
        and not component.is_absolute()
        and len(component.parts) == 1
        and component.name == value
        and value not in {".", ".."}
    )


def _scan_non_test_manifests(manifests: Mapping[str, Path]) -> _MetadataScan:
    problems = _empty_problems()
    records: list[PrivateExamRecord] = []
    patients: set[str] = set()
    image_ids: set[str] = set()
    densities: Counter[int] = Counter()
    row_count = 0
    view_count = 0
    for source_name, path in manifests.items():
        try:
            handle = path.open("r", encoding="utf-8", newline="")
        except (OSError, UnicodeError):
            problems["schema_error_count"] += 1
            continue
        with handle:
            reader = csv.reader(handle)
            try:
                header = tuple(next(reader, ()))
            except csv.Error:
                problems["schema_error_count"] += 1
                continue
            if header != EXPECTED_HEADER:
                problems["schema_error_count"] += 1
                continue
            try:
                for row in reader:
                    row_count += 1
                    if len(row) != len(header):
                        problems["malformed_row_count"] += 1
                        continue
                    patient_id = row[0].strip()
                    density_value = row[1].strip().upper()
                    view_ids = tuple(row[index].strip() for index in range(2, 6))
                    invalid = False
                    if not patient_id:
                        problems["missing_patient_id_count"] += 1
                        invalid = True
                    if any(not value for value in view_ids):
                        problems["missing_view_identifier_count"] += 1
                        invalid = True
                    if density_value not in RSNA_DENSITY:
                        problems["invalid_density_count"] += 1
                        invalid = True
                    if patient_id and patient_id in patients:
                        problems["duplicate_exam_or_patient_count"] += 1
                        invalid = True
                    if len(set(view_ids)) != len(view_ids):
                        problems["repeated_view_identifier_count"] += 1
                        invalid = True
                    repeated_global = sum(
                        bool(image_id) and image_id in image_ids for image_id in view_ids
                    )
                    if repeated_global:
                        problems["repeated_view_identifier_count"] += repeated_global
                        invalid = True
                    identifiers = (patient_id, *view_ids)
                    if any(
                        value and not _identifier_is_safe_path_component(value)
                        for value in identifiers
                    ):
                        problems["unsafe_identifier_count"] += 1
                        invalid = True
                    if invalid:
                        if patient_id:
                            patients.add(patient_id)
                        image_ids.update(value for value in view_ids if value)
                        continue
                    patients.add(patient_id)
                    image_ids.update(view_ids)
                    density = RSNA_DENSITY[density_value]
                    densities[density] += 1
                    views = {
                        view: ViewReference(
                            image_id=image_id,
                            path=str(Path(patient_id) / f"{image_id}.dcm"),
                        )
                        for view, image_id in zip(CANONICAL_VIEWS, view_ids)
                    }
                    records.append(
                        PrivateExamRecord(
                            dataset_namespace=DATASET_NAMESPACE,
                            exam_key=patient_id,
                            patient_key=patient_id,
                            density=density,
                            source_manifest=source_name,
                            views=views,
                        )
                    )
                    view_count += len(CANONICAL_VIEWS)
            except csv.Error:
                problems["malformed_row_count"] += 1
    return _MetadataScan(
        records=tuple(records),
        row_count=row_count,
        density_counts={density: densities[density] for density in range(4)},
        problems=problems,
        patient_count=len(patients),
        view_count=view_count,
    )


def _duplicate_counts(values: Sequence[str]) -> tuple[int, int]:
    counts = Counter(values)
    return sum(count > 1 for count in counts.values()), sum(count - 1 for count in counts.values())


def _decoded_pixel_sha256(image) -> str:
    payload = (
        b"view-risk-production-decoded-rgb/v1\0"
        + image.mode.encode("ascii")
        + b"\0"
        + str(image.size[0]).encode("ascii")
        + b"x"
        + str(image.size[1]).encode("ascii")
        + b"\0"
        + image.tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def _audit_record_images(task):
    (
        exam_key,
        patient_key,
        density,
        source_manifest,
        serialized_views,
        image_root,
        target_views,
    ) = task
    record = PrivateExamRecord(
        dataset_namespace=DATASET_NAMESPACE,
        exam_key=exam_key,
        patient_key=patient_key,
        density=density,
        source_manifest=source_manifest,
        views={
            view: ViewReference(image_id=image_id, path=path)
            for view, image_id, path in serialized_views
        },
    )
    root = Path(image_root)
    results = []
    for view in target_views:
        reference = record.views[view]
        path = (root / reference.path).resolve()
        if path == root or root not in path.parents or path.suffix.lower() != ".dcm":
            results.append((record.exam_key, view, "unreadable", None, None))
            continue
        if not path.is_file():
            results.append((record.exam_key, view, "missing", None, None))
            continue
        try:
            content_hash = sha256_file(path)
        except OSError:
            results.append((record.exam_key, view, "unreadable", None, None))
            continue
        try:
            image = default_private_image_reader(record, view, root)
            try:
                pixel_hash = _decoded_pixel_sha256(image)
            finally:
                image.close()
        except Exception:  # decoder errors are counted, never copied into public evidence
            results.append((record.exam_key, view, "decode_failure", content_hash, None))
            continue
        results.append((record.exam_key, view, "passed", content_hash, pixel_hash))
    return tuple(results)


def _journal_slots(
    records: Sequence[PrivateExamRecord],
) -> tuple[tuple[PrivateExamRecord, str], ...]:
    return tuple((record, view) for record in records for view in CANONICAL_VIEWS)


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _load_or_create_journal(
    path: Path,
    *,
    header: Mapping[str, object],
    slots: Sequence[tuple[PrivateExamRecord, str]],
    resume: bool,
) -> tuple[dict[int, dict[str, object]], int, str]:
    header_sha256 = _hash_json(header)
    header_document = {
        "kind": "header",
        "schema_version": IMAGE_JOURNAL_SCHEMA_VERSION,
        "header_sha256": header_sha256,
        "header": dict(header),
    }
    if not resume:
        try:
            with path.open("xb") as handle:
                handle.write(_canonical_json(header_document) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            raise FileExistsError("refusing to clobber existing private image journal") from exc
        path.chmod(0o600)
        return {}, 0, header_sha256
    if not path.is_file():
        raise ValueError("resume image journal is missing")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError("image journal is unreadable") from exc
    discarded = 0
    if payload and not payload.endswith(b"\n"):
        boundary = payload.rfind(b"\n") + 1
        if boundary == 0:
            raise ValueError("image journal is corrupt")
        try:
            with path.open("r+b") as handle:
                handle.truncate(boundary)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise ValueError("incomplete final journal append could not be discarded") from exc
        payload = payload[:boundary]
        discarded = 1
    lines = payload.splitlines()
    if not lines:
        raise ValueError("image journal is corrupt")
    try:
        loaded_header = json.loads(lines[0])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("image journal is corrupt") from exc
    if loaded_header != header_document:
        existing = loaded_header.get("header", {}) if isinstance(loaded_header, dict) else {}
        if isinstance(existing, dict) and existing.get("source_hashes") != header.get(
            "source_hashes"
        ):
            raise ValueError("journal binding changed source manifests")
        raise ValueError("journal binding disagrees with current audit inputs or semantics")
    latest: dict[int, dict[str, object]] = {}
    expected_sequence = 1
    allowed = {
        "kind",
        "sequence",
        "slot_index",
        "revision",
        "exam_key",
        "view",
        "relative_path",
        "status",
        "file_sha256",
        "pixel_sha256",
    }
    for raw_line in lines[1:]:
        try:
            entry = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("image journal is corrupt") from exc
        if not isinstance(entry, dict) or set(entry) != allowed or entry.get("kind") != "image":
            raise ValueError("image journal is corrupt")
        sequence = entry["sequence"]
        slot_index = entry["slot_index"]
        revision = entry["revision"]
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence != expected_sequence
            or isinstance(slot_index, bool)
            or not isinstance(slot_index, int)
            or slot_index not in range(len(slots))
            or isinstance(revision, bool)
            or not isinstance(revision, int)
        ):
            raise ValueError("image journal is corrupt")
        record, view = slots[slot_index]
        prior_revision = int(latest[slot_index]["revision"]) if slot_index in latest else 0
        if (
            revision != prior_revision + 1
            or entry["exam_key"] != record.exam_key
            or entry["view"] != view
            or entry["relative_path"] != record.views[view].path
            or entry["status"] not in {"passed", "missing", "unreadable", "decode_failure"}
        ):
            raise ValueError("image journal is corrupt or inconsistent")
        file_sha256 = entry["file_sha256"]
        pixel_sha256 = entry["pixel_sha256"]
        if entry["status"] == "passed":
            valid_hashes = _is_sha256(file_sha256) and _is_sha256(pixel_sha256)
        elif entry["status"] == "decode_failure":
            valid_hashes = _is_sha256(file_sha256) and pixel_sha256 is None
        else:
            valid_hashes = file_sha256 is None and pixel_sha256 is None
        if not valid_hashes:
            raise ValueError("image journal is corrupt")
        latest[slot_index] = entry
        expected_sequence += 1
    return latest, discarded, header_sha256


def _append_journal_results(
    path: Path,
    results: Sequence[tuple[str, str, str, str | None, str | None]],
    *,
    slots: Sequence[tuple[PrivateExamRecord, str]],
    slot_lookup: Mapping[tuple[str, str], int],
    latest: dict[int, dict[str, object]],
) -> None:
    sequence = sum(int(entry["revision"]) for entry in latest.values())
    documents = []
    for exam_key, view, status, file_sha256, pixel_sha256 in results:
        slot_index = slot_lookup[(exam_key, view)]
        record, expected_view = slots[slot_index]
        prior_revision = int(latest[slot_index]["revision"]) if slot_index in latest else 0
        sequence += 1
        entry: dict[str, object] = {
            "kind": "image",
            "sequence": sequence,
            "slot_index": slot_index,
            "revision": prior_revision + 1,
            "exam_key": record.exam_key,
            "view": expected_view,
            "relative_path": record.views[expected_view].path,
            "status": status,
            "file_sha256": file_sha256,
            "pixel_sha256": pixel_sha256,
        }
        documents.append(entry)
        latest[slot_index] = entry
    encoded = b"".join(_canonical_json(document) + b"\n" for document in documents)
    with path.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _verify_completed_entries(
    latest: Mapping[int, Mapping[str, object]],
    *,
    slots: Sequence[tuple[PrivateExamRecord, str]],
    image_root: Path,
    re_audit_changed: bool,
) -> tuple[dict[int, dict[str, object]], set[int]]:
    reusable: dict[int, dict[str, object]] = {}
    changed: set[int] = set()
    root = image_root.resolve()
    for slot_index, entry in latest.items():
        record, view = slots[slot_index]
        path = (root / record.views[view].path).resolve()
        status = entry["status"]
        current_matches = False
        if status == "missing":
            current_matches = not path.is_file()
        elif status == "unreadable":
            try:
                sha256_file(path)
            except OSError:
                current_matches = True
        else:
            try:
                current_matches = sha256_file(path) == entry["file_sha256"]
            except OSError:
                current_matches = False
        if current_matches:
            reusable[slot_index] = dict(entry)
        else:
            changed.add(slot_index)
    if changed and not re_audit_changed:
        raise ValueError(
            f"changed completed image detected ({len(changed)}); explicit re-audit is required"
        )
    return reusable, changed


def _image_progress(
    *,
    header_sha256: str,
    latest: Mapping[int, Mapping[str, object]],
    expected_count: int,
    status: str,
    reused_count: int,
    changed_count: int,
    reaudited_count: int,
    discarded_count: int,
) -> dict[str, object]:
    counts = Counter(str(entry["status"]) for entry in latest.values())
    return {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "journal_header_sha256": header_sha256,
        "dataset": "RSNA-SMBC",
        "status": status,
        "completed_image_count": len(latest),
        "expected_image_count": expected_count,
        "pending_image_count": expected_count - len(latest),
        "status_counts": {
            name: counts[name] for name in ("passed", "missing", "unreadable", "decode_failure")
        },
        "resume": {
            "reused_image_count": reused_count,
            "changed_image_count": changed_count,
            "reaudited_image_count": reaudited_count,
            "discarded_incomplete_final_append_count": discarded_count,
        },
        "readiness_promoted": False,
        "test_evaluation_unlocked": False,
    }


def _scan_from_journal_entries(
    records: Sequence[PrivateExamRecord],
    latest: Mapping[int, Mapping[str, object]],
) -> _ImageScan:
    slots = _journal_slots(records)
    if set(latest) != set(range(len(slots))):
        raise ValueError("partial image journal cannot be promoted to readiness")
    byte_hashes = [
        str(latest[index]["file_sha256"])
        for index in range(len(slots))
        if latest[index]["file_sha256"] is not None
    ]
    pixel_hashes = [
        str(latest[index]["pixel_sha256"])
        for index in range(len(slots))
        if latest[index]["pixel_sha256"] is not None
    ]
    statuses = Counter(str(latest[index]["status"]) for index in range(len(slots)))
    byte_groups, byte_extras = _duplicate_counts(byte_hashes)
    pixel_groups, pixel_extras = _duplicate_counts(pixel_hashes)
    complete = not any(
        (
            statuses["missing"],
            statuses["unreadable"],
            statuses["decode_failure"],
            byte_extras,
            pixel_extras,
        )
    )
    bound_records: list[PrivateExamRecord] = []
    if complete:
        slot_lookup = {
            (record.exam_key, view): index for index, (record, view) in enumerate(slots)
        }
        for record in records:
            bound_records.append(
                replace(
                    record,
                    views={
                        view: replace(
                            record.views[view],
                            content_sha256=str(
                                latest[slot_lookup[(record.exam_key, view)]]["file_sha256"]
                            ),
                        )
                        for view in CANONICAL_VIEWS
                    },
                )
            )
    return _ImageScan(
        records=tuple(bound_records),
        expected_count=len(slots),
        readable_count=len(byte_hashes),
        decoded_count=len(pixel_hashes),
        missing_count=statuses["missing"],
        unreadable_count=statuses["unreadable"],
        decode_failure_count=statuses["decode_failure"],
        byte_group_count=byte_groups,
        byte_extra_count=byte_extras,
        pixel_group_count=pixel_groups,
        pixel_extra_count=pixel_extras,
        byte_collection_sha256=_hash_json(
            {"algorithm": "sha256", "members": sorted(byte_hashes)}
        ),
        pixel_collection_sha256=_hash_json(
            {
                "algorithm": "production-decoded-rgb-sha256/v1",
                "members": sorted(pixel_hashes),
            }
        ),
    )


def _audit_images(
    records: Sequence[PrivateExamRecord],
    image_root: Path,
    *,
    workers: int,
    journal_path: Path,
    journal_header: Mapping[str, object],
    progress_path: Path,
    resume: bool,
    re_audit_changed: bool,
    stop_after_new_records: int | None,
) -> tuple[_ImageScan, dict[str, int], str]:
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 16:
        raise ValueError("image audit workers must be an integer in [1, 16]")
    if stop_after_new_records is not None and (
        isinstance(stop_after_new_records, bool)
        or not isinstance(stop_after_new_records, int)
        or stop_after_new_records < 1
    ):
        raise ValueError("stop_after_new_records must be a positive integer")
    root = image_root.resolve()
    slots = _journal_slots(records)
    slot_lookup = {
        (record.exam_key, view): index for index, (record, view) in enumerate(slots)
    }
    loaded, discarded_count, header_sha256 = _load_or_create_journal(
        journal_path, header=journal_header, slots=slots, resume=resume
    )
    if resume:
        valid_latest, changed = _verify_completed_entries(
            loaded,
            slots=slots,
            image_root=root,
            re_audit_changed=re_audit_changed,
        )
    else:
        valid_latest, changed = {}, set()
    journal_latest = dict(loaded)
    reused_count = len(valid_latest)
    reaudited_count = 0
    progress_resume = resume and progress_path.exists()
    _write_progress(
        progress_path,
        _image_progress(
            header_sha256=header_sha256,
            latest=valid_latest,
            expected_count=len(slots),
            status="in_progress",
            reused_count=reused_count,
            changed_count=len(changed),
            reaudited_count=reaudited_count,
            discarded_count=discarded_count,
        ),
        resume=progress_resume,
        journal_header_sha256=header_sha256,
    )
    pending_by_exam: dict[str, list[str]] = {}
    for slot_index, (record, view) in enumerate(slots):
        if slot_index not in valid_latest:
            pending_by_exam.setdefault(record.exam_key, []).append(view)
    tasks = tuple(
        (
            record.exam_key,
            record.patient_key,
            record.density,
            record.source_manifest,
            tuple(
                (view, record.views[view].image_id, record.views[view].path)
                for view in CANONICAL_VIEWS
            ),
            str(root),
            tuple(pending_by_exam[record.exam_key]),
        )
        for record in records
        if record.exam_key in pending_by_exam
    )
    if workers == 1:
        audited_records = map(_audit_record_images, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        audited_records = executor.map(_audit_record_images, tasks, chunksize=1)
    try:
        completed_new_records = 0
        for record_results in audited_records:
            _append_journal_results(
                journal_path,
                record_results,
                slots=slots,
                slot_lookup=slot_lookup,
                latest=journal_latest,
            )
            for exam_key, view, _status, _content_hash, _pixel_hash in record_results:
                slot_index = slot_lookup[(exam_key, view)]
                valid_latest[slot_index] = journal_latest[slot_index]
            reaudited_count += sum(
                slot_lookup[(exam_key, view)] in changed
                for exam_key, view, _status, _content_hash, _pixel_hash in record_results
            )
            completed_new_records += 1
            _write_progress(
                progress_path,
                _image_progress(
                    header_sha256=header_sha256,
                    latest=valid_latest,
                    expected_count=len(slots),
                    status="in_progress",
                    reused_count=reused_count,
                    changed_count=len(changed),
                    reaudited_count=reaudited_count,
                    discarded_count=discarded_count,
                ),
                resume=True,
                journal_header_sha256=header_sha256,
            )
            if (
                stop_after_new_records is not None
                and completed_new_records >= stop_after_new_records
            ):
                raise AuditInterrupted("intentional bounded interruption after durable progress")
    finally:
        if executor is not None:
            executor.shutdown()
    scan = _scan_from_journal_entries(records, journal_latest)
    _write_progress(
        progress_path,
        _image_progress(
            header_sha256=header_sha256,
            latest=journal_latest,
            expected_count=len(slots),
            status="image_audit_complete",
            reused_count=reused_count,
            changed_count=len(changed),
            reaudited_count=reaudited_count,
            discarded_count=discarded_count,
        ),
        resume=True,
        journal_header_sha256=header_sha256,
    )
    return (
        scan,
        {
            "reused_image_count": reused_count,
            "changed_image_count": len(changed),
            "reaudited_image_count": reaudited_count,
            "discarded_incomplete_final_append_count": discarded_count,
        },
        header_sha256,
    )


def _load_prior_hashes(path: Path) -> Mapping[str, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        hashes = value["rsna"]["source_manifest_sha256"]
        train = hashes["train"]
        valid = hashes["valid"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("prior inventory does not contain RSNA source hashes") from exc
    if not all(isinstance(item, str) and len(item) == 64 for item in (train, valid)):
        raise ValueError("prior inventory contains invalid RSNA source hashes")
    return {"train": train, "valid": valid}


def _role_counts(values: Mapping[Role | str, int] | None) -> Mapping[Role, int] | None:
    if values is None:
        return None
    result: dict[Role, int] = {}
    for key, count in values.items():
        role = Role(key)
        if role not in NON_TEST_ROLES or role in result:
            raise ValueError("requested counts must contain each non-test role exactly once")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("requested role counts must be nonnegative integers")
        result[role] = count
    if set(result) != set(NON_TEST_ROLES):
        raise ValueError("requested counts must contain each non-test role exactly once")
    return result


def _public_assignment(summary) -> dict[str, object]:
    return {
        "requested_exam_counts": {
            role.value: summary.requested_exam_counts[role] for role in NON_TEST_ROLES
        },
        "actual_exam_counts": {
            role.value: summary.actual_exam_counts[role] for role in NON_TEST_ROLES
        },
        "actual_patient_counts": {
            role.value: summary.actual_patient_counts[role] for role in NON_TEST_ROLES
        },
        "class_exam_counts": {
            role.value: {
                str(density): summary.class_exam_counts[role][density] for density in range(4)
            }
            for role in NON_TEST_ROLES
        },
        "patient_stratum_counts": {
            str(density): summary.patient_stratum_counts[density] for density in range(4)
        },
        "conflicting_patient_count": summary.conflicting_patient_count,
        "stratification_rule": summary.stratification_rule,
        "targets_exact": summary.group_constraints.targets_exact,
        "multi_exam_patient_count": summary.group_constraints.multi_exam_patient_count,
        "max_exams_per_patient": summary.group_constraints.max_exams_per_patient,
        "exclusion_counts": dict(summary.exclusion_counts),
    }


def audit_rsna_readiness(
    *,
    train_manifest: str | Path,
    validation_manifest: str | Path,
    locked_test_manifest: str | Path,
    image_root: str | Path,
    prior_inventory: str | Path,
    private_root: str | Path,
    public_report: str | Path,
    run_name: str,
    requested_counts: Mapping[Role | str, int] | None = None,
    seed: int = 42,
    image_workers: int = 4,
    resume: bool = False,
    re_audit_changed: bool = False,
    _stop_after_new_records: int | None = None,
) -> dict[str, object]:
    """Audit RSNA and persist private bindings plus sanitized public evidence.

    Promotion is fail-closed.  Role assignment is not attempted until metadata,
    locked-patient isolation, source bindings, DICOM byte uniqueness, production
    decoding, and decoded-pixel uniqueness all pass.
    """

    started = time.monotonic()
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != 42:
        raise ValueError("the RSNA readiness audit requires assignment seed 42")
    if (
        isinstance(image_workers, bool)
        or not isinstance(image_workers, int)
        or not 1 <= image_workers <= 16
    ):
        raise ValueError("image_workers must be an integer in [1, 16]")
    if not isinstance(resume, bool) or not isinstance(re_audit_changed, bool):
        raise TypeError("resume flags must be booleans")
    if re_audit_changed and not resume:
        raise ValueError("re_audit_changed requires explicit resume")
    normalized_counts = _role_counts(requested_counts)
    effective_counts = normalized_counts or {
        Role.CLASSIFIER_FIT: 3209,
        Role.CONFIDENCE_FIT: 987,
        Role.TUNE: 370,
        Role.PILOT: 371,
    }
    private_base = _safe_private_root(private_root)
    run_component = _validate_run_name(run_name)
    report_path = Path(public_report).expanduser().resolve()
    if report_path.suffix.lower() != ".json":
        raise ValueError("public report path must end in .json")
    if report_path.exists():
        raise FileExistsError("refusing to clobber existing public audit report")
    progress_report_path = _progress_path(report_path)
    run_dir = private_base / run_component
    if resume:
        if not run_dir.is_dir():
            raise ValueError("resume requires an existing private audit run directory")
    else:
        if run_dir.exists():
            raise FileExistsError("refusing to clobber existing private audit run directory")
        if progress_report_path.exists():
            raise FileExistsError("refusing to clobber existing public audit progress")
        private_base.mkdir(parents=True, exist_ok=True)
        private_base.chmod(0o700)
        run_dir.mkdir(mode=0o700)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    sources = {
        "train": Path(train_manifest).expanduser().resolve(),
        "valid": Path(validation_manifest).expanduser().resolve(),
    }
    test_path = Path(locked_test_manifest).expanduser().resolve()
    images = Path(image_root).expanduser().resolve()
    prior_path = Path(prior_inventory).expanduser().resolve()
    observed_source_hashes: dict[str, str] = {}
    blockers: list[dict[str, object]] = []
    audits: dict[str, str] = {
        "locked_test_identity_projection": "not_performed",
        "locked_test_outcomes": "not_performed_by_design",
        "locked_test_images": "not_performed_by_design",
        "official_patient_schema": "not_performed",
        "source_hash_binding": "not_performed",
        "production_dicom_file_bytes": "not_performed",
        "production_dicom_decode": "not_performed",
        "duplicate_file_bytes": "not_performed",
        "duplicate_decoded_pixels": "not_performed",
        "role_assignment": "not_performed",
        "inventory_validation": "not_performed",
        "real_readiness_api": "not_performed",
    }
    artifacts: dict[str, Path] = {}
    artifact_hashes: dict[str, str] = {}
    assignment_public: dict[str, object] | None = None
    inventory_public: dict[str, object] | None = None
    image_public: dict[str, object] | None = None
    resume_public = {
        "reused_image_count": 0,
        "changed_image_count": 0,
        "reaudited_image_count": 0,
        "discarded_incomplete_final_append_count": 0,
    }
    metadata_public: dict[str, object]
    locked_public: dict[str, object]

    locked_identities = _read_locked_identities(test_path)
    audits["locked_test_identity_projection"] = "performed"
    locked_duplicate_count = len(locked_identities) - len(set(locked_identities))
    if locked_duplicate_count:
        blockers.append({"code": "duplicate_locked_test_patient_identity", "count": locked_duplicate_count})
    locked_digests = frozenset(
        patient_identity_digest(DATASET_NAMESPACE, identity) for identity in locked_identities
    )
    denylist_path = run_dir / "locked-test-identity-denylist.json"
    denylist_document = {
        "schema_version": DENYLIST_FILE_SCHEMA_VERSION,
        "dataset_namespace": DATASET_NAMESPACE,
        "patient_identity_digests": sorted(locked_digests),
    }
    _ensure_private_document(
        denylist_path,
        denylist_document,
        resume=resume,
        description="locked test identity denylist",
    )
    artifacts["locked_test_identity_denylist"] = denylist_path
    denylist_file_hash = sha256_file(denylist_path)
    artifact_hashes["locked_test_identity_denylist"] = denylist_file_hash
    denylist = LockedPatientIdentityDenylist(
        dataset_namespace=DATASET_NAMESPACE,
        source_sha256=denylist_file_hash,
        patient_identity_digests=locked_digests,
    )

    try:
        observed_source_hashes = {name: sha256_file(path) for name, path in sources.items()}
        prior_hashes = _load_prior_hashes(prior_path)
        mismatches = sum(
            observed_source_hashes[name] != prior_hashes[name] for name in ("train", "valid")
        )
        audits["source_hash_binding"] = "passed" if not mismatches else "failed"
        if mismatches:
            if resume:
                raise ValueError("journal binding changed source manifests")
            blockers.append({"code": "prior_inventory_source_hash_mismatch", "count": mismatches})
    except ValueError:
        if resume:
            raise
        audits["source_hash_binding"] = "failed"
        blockers.append({"code": "source_hash_binding_unavailable", "count": 1})
    except OSError:
        audits["source_hash_binding"] = "failed"
        blockers.append({"code": "source_hash_binding_unavailable", "count": 1})

    metadata = _scan_non_test_manifests(sources)
    metadata_problem_count = sum(metadata.problems.values())
    audits["official_patient_schema"] = "passed" if not metadata_problem_count else "failed"
    for name, count in metadata.problems.items():
        if count:
            blockers.append({"code": name, "count": count})
    overlap_count = len(set(locked_identities).intersection(record.patient_key for record in metadata.records))
    if overlap_count:
        blockers.append({"code": "locked_test_patient_overlap", "count": overlap_count})
    expected_total = sum(effective_counts.values())
    if metadata.row_count != expected_total:
        blockers.append(
            {
                "code": "non_test_exam_count_target_mismatch",
                "expected": expected_total,
                "observed": metadata.row_count,
            }
        )
    locked_public = {
        "identity_field": "patient_id",
        "identity_count": len(locked_identities),
        "duplicate_identity_count": locked_duplicate_count,
        "overlap_count": overlap_count,
        "outcome_values_consulted": False,
        "images_accessed": False,
        "evaluation_unlocked": False,
    }
    metadata_public = {
        "row_count": metadata.row_count,
        "eligible_record_count": len(metadata.records),
        "patient_count": metadata.patient_count,
        "expected_named_view_count": metadata.view_count,
        "density_counts": {str(key): value for key, value in metadata.density_counts.items()},
        **dict(metadata.problems),
    }

    mapping_path: Path | None = None
    mapping_hash: str | None = None
    mapping: PatientMappingDeclaration | None = None
    if not metadata_problem_count and observed_source_hashes:
        mapping_path = run_dir / "patient-mapping-binding.json"
        mapping_document = {
            "schema_version": MAPPING_FILE_SCHEMA_VERSION,
            "dataset_namespace": DATASET_NAMESPACE,
            "official_field": "patient_id",
            "verification_method": "direct official RSNA patient_id field",
            "verification_authority": "authorized local P5A schema audit; no external signoff claimed",
            "source_hashes": observed_source_hashes,
            "entries": [
                {
                    "source_manifest": record.source_manifest,
                    "exam_key": record.exam_key,
                    "patient_key": record.patient_key,
                }
                for record in sorted(metadata.records, key=lambda item: item.exam_key)
            ],
        }
        _ensure_private_document(
            mapping_path,
            mapping_document,
            resume=resume,
            description="patient mapping",
        )
        artifacts["patient_mapping_binding"] = mapping_path
        mapping_hash = sha256_file(mapping_path)
        artifact_hashes["patient_mapping_binding"] = mapping_hash
        mapping = PatientMappingDeclaration(
            dataset_namespace=DATASET_NAMESPACE,
            mapping_source_sha256=mapping_hash,
            verification_method="direct official RSNA patient_id field",
            verification_authority=(
                "authorized local P5A schema audit; no external custodian signoff claimed"
            ),
        )

    metadata_blocker_codes = {
        "duplicate_locked_test_patient_identity",
        "prior_inventory_source_hash_mismatch",
        "source_hash_binding_unavailable",
        "locked_test_patient_overlap",
        "non_test_exam_count_target_mismatch",
        *metadata.problems.keys(),
    }
    stop_before_images = any(item["code"] in metadata_blocker_codes for item in blockers)
    image_scan: _ImageScan | None = None
    journal_header_sha256: str | None = None
    if not stop_before_images:
        if mapping_hash is None:
            raise ValueError("image audit requires the bound official patient mapping")
        journal_path = run_dir / "image-audit-journal.jsonl"
        journal_header = {
            "schema_version": IMAGE_JOURNAL_SCHEMA_VERSION,
            "audit_semantics_version": IMAGE_AUDIT_SEMANTICS_VERSION,
            "dataset_namespace": DATASET_NAMESPACE,
            "source_hashes": observed_source_hashes,
            "source_paths": {name: str(path) for name, path in sources.items()},
            "locked_test_manifest_path": str(test_path),
            "locked_identity_denylist_sha256": denylist_file_hash,
            "patient_mapping_sha256": mapping_hash,
            "prior_inventory_path": str(prior_path),
            "prior_inventory_sha256": sha256_file(prior_path),
            "image_root": str(images),
            "canonical_views": list(CANONICAL_VIEWS),
            "expected_image_count": len(metadata.records) * len(CANONICAL_VIEWS),
            "record_inventory_sha256": _hash_json(
                [
                    {
                        "exam_key": record.exam_key,
                        "patient_key": record.patient_key,
                        "density": record.density,
                        "source_manifest": record.source_manifest,
                        "views": {
                            view: {
                                "image_id": record.views[view].image_id,
                                "path": record.views[view].path,
                            }
                            for view in CANONICAL_VIEWS
                        },
                    }
                    for record in metadata.records
                ]
            ),
            "assignment_seed": seed,
            "requested_exam_counts": {
                role.value: effective_counts[role] for role in NON_TEST_ROLES
            },
            "image_workers": image_workers,
        }
        image_scan, resume_public, journal_header_sha256 = _audit_images(
            metadata.records,
            images,
            workers=image_workers,
            journal_path=journal_path,
            journal_header=journal_header,
            progress_path=progress_report_path,
            resume=resume,
            re_audit_changed=re_audit_changed,
            stop_after_new_records=_stop_after_new_records,
        )
        artifacts["image_audit_journal"] = journal_path
        artifacts["public_progress"] = progress_report_path
        artifact_hashes["image_audit_journal"] = sha256_file(journal_path)
        artifact_hashes["image_audit_journal_header"] = journal_header_sha256
        audits["production_dicom_file_bytes"] = "performed"
        audits["production_dicom_decode"] = "performed"
        audits["duplicate_file_bytes"] = "failed" if image_scan.byte_extra_count else "passed"
        audits["duplicate_decoded_pixels"] = (
            "failed" if image_scan.pixel_extra_count else "passed"
        )
        image_public = {
            "format": "DICOM",
            "decoder_semantics": "accepted production reader normalized RGB",
            "expected_file_count": image_scan.expected_count,
            "readable_file_count": image_scan.readable_count,
            "decoded_file_count": image_scan.decoded_count,
            "missing_file_count": image_scan.missing_count,
            "unreadable_file_count": image_scan.unreadable_count,
            "decode_failure_count": image_scan.decode_failure_count,
            "duplicates": {
                "identical_file_byte_group_count": image_scan.byte_group_count,
                "identical_file_byte_extra_count": image_scan.byte_extra_count,
                "decoded_pixel_group_count": image_scan.pixel_group_count,
                "decoded_pixel_extra_count": image_scan.pixel_extra_count,
            },
            "cached_png_substitution_used": False,
        }
        for code, count in (
            ("missing_dicom_file", image_scan.missing_count),
            ("unreadable_dicom_file", image_scan.unreadable_count),
            ("dicom_decode_failure", image_scan.decode_failure_count),
            ("duplicate_file_bytes", image_scan.byte_extra_count),
            ("duplicate_decoded_pixels", image_scan.pixel_extra_count),
        ):
            if count:
                blockers.append({"code": code, "count": count})
        if image_scan.byte_collection_sha256 is not None:
            artifact_hashes["dicom_byte_collection"] = image_scan.byte_collection_sha256
        if image_scan.pixel_collection_sha256 is not None:
            artifact_hashes["decoded_pixel_collection"] = image_scan.pixel_collection_sha256

    readiness = None
    if not blockers and image_scan is not None and mapping is not None and mapping_path is not None:
        try:
            assignment = assign_patient_roles(
                image_scan.records,
                patient_mapping=mapping,
                source_hashes=observed_source_hashes,
                counts=normalized_counts,
                seed=seed,
            )
        except (TypeError, ValueError):
            blockers.append({"code": "accepted_role_assignment_refused", "count": 1})
            audits["role_assignment"] = "failed"
        else:
            assignment_public = _public_assignment(assignment.summary)
            if not assignment.summary.group_constraints.targets_exact:
                blockers.append(
                    {"code": "patient_grouping_prevented_exact_role_targets", "count": 1}
                )
                audits["role_assignment"] = "failed"
            else:
                audits["role_assignment"] = "passed"
                try:
                    inventory = validate_inventory(
                        (assignment.manifest,), locked_patient_denylists=(denylist,)
                    )
                except (TypeError, ValueError):
                    blockers.append({"code": "accepted_inventory_validation_refused", "count": 1})
                    audits["inventory_validation"] = "failed"
                else:
                    audits["inventory_validation"] = "passed"
                    inventory_public = {
                        "dataset_count": inventory.dataset_count,
                        "exam_count": inventory.exam_count,
                        "patient_count": inventory.patient_count,
                        "image_count": inventory.image_count,
                        "content_hash_count": inventory.content_hash_count,
                        "locked_denylist_dataset_count": inventory.locked_denylist_dataset_count,
                        "locked_identity_count": inventory.locked_identity_count,
                        "collision_counts": {
                            "image_id": inventory.image_id_collision_count,
                            "image_path": inventory.image_path_collision_count,
                            "content": inventory.content_collision_count,
                        },
                    }
                    try:
                        candidate_readiness = audit_real_data_readiness(
                            (assignment.manifest,),
                            locked_patient_denylist=denylist,
                            source_files=sources,
                            patient_mapping_file=mapping_path,
                            locked_denylist_file=denylist_path,
                            image_root=images,
                        )
                    except (OSError, TypeError, ValueError):
                        blockers.append(
                            {"code": "accepted_real_readiness_audit_refused", "count": 1}
                        )
                        audits["real_readiness_api"] = "failed"
                    else:
                        audits["real_readiness_api"] = "passed"
                        role_manifest_path = run_dir / "role-manifest.json"
                        binding_path = run_dir / "role-manifest-binding.json"
                        readiness_path = run_dir / "readiness.json"
                        if any(
                            path.exists()
                            for path in (role_manifest_path, binding_path, readiness_path)
                        ):
                            raise FileExistsError(
                                "refusing to clobber incomplete final readiness artifacts"
                            )
                        binding = save_private_manifest(
                            assignment.manifest, role_manifest_path, private_root=private_base
                        )
                        artifacts["role_manifest"] = role_manifest_path
                        artifact_hashes["role_manifest_file"] = sha256_file(role_manifest_path)
                        artifact_hashes["role_manifest_payload"] = binding.manifest_sha256
                        _write_json_exclusive(
                            binding_path,
                            {
                                "schema_version": binding.schema_version,
                                "dataset_namespace": binding.dataset_namespace,
                                "source_hashes": dict(binding.source_hashes),
                                "manifest_sha256": binding.manifest_sha256,
                            },
                            private=True,
                        )
                        artifacts["role_manifest_binding"] = binding_path
                        artifact_hashes["role_manifest_binding"] = sha256_file(binding_path)
                        save_readiness_audit(candidate_readiness, readiness_path)
                        artifacts["readiness"] = readiness_path
                        artifact_hashes["readiness_file"] = sha256_file(readiness_path)
                        artifact_hashes["readiness_payload"] = candidate_readiness.sha256
                        readiness = candidate_readiness

    status = "ready" if readiness is not None and not blockers else "blocked"
    elapsed = time.monotonic() - started
    private_run_path = run_dir / "run-manifest.json"
    artifacts["private_run_manifest"] = private_run_path
    private_run = {
        "schema_version": PRIVATE_RUN_SCHEMA_VERSION,
        "status": status,
        "dataset_namespace": DATASET_NAMESPACE,
        "configuration": {
            "seed": seed,
            "image_workers": image_workers,
            "resumed": resume,
            "re_audit_changed": re_audit_changed,
            "requested_counts": {
                role.value: effective_counts[role] for role in NON_TEST_ROLES
            },
        },
        "inputs": {
            "train_manifest": str(sources["train"]),
            "validation_manifest": str(sources["valid"]),
            "locked_test_manifest": str(test_path),
            "image_root": str(images),
            "prior_inventory": str(prior_path),
        },
        "outputs": {
            **{name: str(path) for name, path in artifacts.items()},
            "public_report": str(report_path),
        },
        "source_hashes": {
            **observed_source_hashes,
        },
        "artifact_hashes": artifact_hashes,
        "audits": audits,
        "blockers": blockers,
        "runtime_seconds": elapsed,
    }
    _write_json_exclusive(private_run_path, private_run, private=True)
    artifact_hashes["private_run_manifest"] = sha256_file(private_run_path)

    report: dict[str, object] = {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "dataset": "RSNA-SMBC",
        "scope": "RSNA only; no MINI-DDSM or cross-dataset cleanliness claim",
        "status": status,
        "data_readiness": "verified" if status == "ready" else "blocked",
        "weight_compute_readiness": "not_assessed",
        "scientific_performance": "not_assessed",
        "test_evaluation_unlocked": False,
        "configuration": {
            "assignment_seed": seed,
            "image_audit_workers": image_workers,
            "patient_grouping_mandatory": True,
            "canonical_views": list(CANONICAL_VIEWS),
            "requested_exam_counts": private_run["configuration"]["requested_counts"],
            "primary_image_source": "original production DICOM",
            "cached_png_substitution_used": False,
        },
        "source_manifest_sha256": {
            **observed_source_hashes,
        },
        "prior_inventory_source_hash_match": audits["source_hash_binding"] == "passed",
        "audits": audits,
        "observations": {
            "locked_test_isolation": locked_public,
            "metadata": metadata_public,
            "image_audit": image_public,
            "assignment": assignment_public,
            "inventory": inventory_public,
            "resume": resume_public,
        },
        "aggregate_artifact_hashes": artifact_hashes,
        "blockers": blockers,
        "missing_prerequisites": [item["code"] for item in blockers],
        "runtime_seconds": elapsed,
        "limitations": [
            "Locked test outcomes and images were not accessed.",
            "The audit does not load public model weights or establish compute readiness.",
            "The audit reports no scientific performance and does not unlock test evaluation.",
            "MINI-DDSM patient mapping remains blocked and was not audited.",
        ],
    }
    _write_json_exclusive(report_path, report, private=False)
    if journal_header_sha256 is not None and progress_report_path.exists():
        progress = _read_json_object(progress_report_path, description="public audit progress")
        progress.update(
            {
                "status": "finalized",
                "final_audit_status": status,
                "readiness_promoted": status == "ready",
            }
        )
        _write_progress(
            progress_report_path,
            progress,
            resume=True,
            journal_header_sha256=journal_header_sha256,
        )
    return report
