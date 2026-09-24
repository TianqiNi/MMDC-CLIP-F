from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import mmdc_clip_f.research.view_risk.rsna_audit as rsna_audit_module
from mmdc_clip_f.research.view_risk.roles import (
    ManifestBinding,
    NON_TEST_ROLES,
    load_private_manifest,
)
from mmdc_clip_f.research.view_risk.rsna_audit import (
    AuditInterrupted,
    _project_identity_rows,
    audit_rsna_readiness,
)
from mmdc_clip_f.research.view_risk.training import load_verified_readiness_audit


VIEWS = ("L_CC", "L_MLO", "R_CC", "R_MLO")


class _IdentityOnlyRow:
    def __init__(self, patient_id: str) -> None:
        self.patient_id = patient_id

    def __len__(self) -> int:
        return 6

    def __getitem__(self, index: int) -> str:
        if index != 0:
            raise AssertionError("locked outcome/view value was consulted")
        return self.patient_id


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[tuple[str, str, str, str, str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("patient_id", "density", *VIEWS))
        writer.writerows(rows)


def _write_dicom(path: Path, pattern: int) -> None:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.zeros((4, 4), dtype=np.uint16)
    pixels.flat[pattern % pixels.size] = 4095
    pixels.flat[(pattern * 7 + 3) % pixels.size] = pattern + 1
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.Rows = 4
    dataset.Columns = 4
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 12
    dataset.HighBit = 11
    dataset.PixelRepresentation = 0
    dataset.PixelData = pixels.tobytes()
    pydicom.dcmwrite(path, dataset, enforce_file_format=True)


def _fixture(
    root: Path,
    *,
    locked_patient: str = "locked-only",
    duplicate_pixels: bool = False,
    duplicate_bytes: bool = False,
    missing_file: bool = False,
    missing_view: bool = False,
    missing_patient: bool = False,
) -> dict[str, Path]:
    source = root / "source"
    images = source / "train_images"
    source.mkdir(parents=True)
    rows = []
    for patient_index in range(4):
        patient = "" if missing_patient and patient_index == 0 else f"development-{patient_index}"
        view_ids = [f"image-{patient_index}-{view_index}" for view_index in range(4)]
        if missing_view and patient_index == 0:
            view_ids[2] = ""
        rows.append((patient, "ABCD"[patient_index], *view_ids))
        if patient:
            for view_index, image_id in enumerate(view_ids):
                if not image_id:
                    continue
                pattern = patient_index * 4 + view_index
                if duplicate_pixels and pattern == 1:
                    pattern = 0
                _write_dicom(images / patient / f"{image_id}.dcm", pattern)
    if duplicate_bytes:
        shutil.copyfile(
            images / "development-0" / "image-0-0.dcm",
            images / "development-0" / "image-0-1.dcm",
        )
    if missing_file:
        (images / "development-0" / "image-0-0.dcm").unlink()
    train = source / "train.csv"
    validation = source / "validation.csv"
    test = source / "test.csv"
    _write_csv(train, rows[:2])
    _write_csv(validation, rows[2:])
    _write_csv(test, [(locked_patient, "OUTCOME-MUST-NOT-BE-READ", "x", "y", "z", "w")])
    prior = root / "prior.json"
    prior.write_text(
        json.dumps(
            {
                "rsna": {
                    "source_manifest_sha256": {
                        "train": _sha256(train),
                        "valid": _sha256(validation),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return {
        "train": train,
        "validation": validation,
        "test": test,
        "images": images,
        "prior": prior,
    }


def _run(
    root: Path,
    *,
    locked_patient: str = "locked-only",
    duplicate_pixels: bool = False,
    duplicate_bytes: bool = False,
    missing_file: bool = False,
    missing_view: bool = False,
    missing_patient: bool = False,
    run_name: str = "run-1",
):
    paths = _fixture(
        root,
        locked_patient=locked_patient,
        duplicate_pixels=duplicate_pixels,
        duplicate_bytes=duplicate_bytes,
        missing_file=missing_file,
        missing_view=missing_view,
        missing_patient=missing_patient,
    )
    private_root = root / "private"
    public_report = root / "public.json"
    result = audit_rsna_readiness(
        train_manifest=paths["train"],
        validation_manifest=paths["validation"],
        locked_test_manifest=paths["test"],
        image_root=paths["images"],
        prior_inventory=paths["prior"],
        private_root=private_root,
        public_report=public_report,
        run_name=run_name,
        requested_counts=dict(zip(NON_TEST_ROLES, (1, 1, 1, 1))),
    )
    return result, paths, private_root, public_report


def _audit_fixture(
    paths: dict[str, Path],
    *,
    private_root: Path,
    public_report: Path,
    run_name: str,
    resume: bool = False,
    re_audit_changed: bool = False,
    stop_after_new_records: int | None = None,
):
    return audit_rsna_readiness(
        train_manifest=paths["train"],
        validation_manifest=paths["validation"],
        locked_test_manifest=paths["test"],
        image_root=paths["images"],
        prior_inventory=paths["prior"],
        private_root=private_root,
        public_report=public_report,
        run_name=run_name,
        requested_counts=dict(zip(NON_TEST_ROLES, (1, 1, 1, 1))),
        image_workers=1,
        resume=resume,
        re_audit_changed=re_audit_changed,
        _stop_after_new_records=stop_after_new_records,
    )


def test_locked_projection_consults_only_patient_identity_column() -> None:
    rows = (_IdentityOnlyRow("locked-a"), _IdentityOnlyRow("locked-b"))

    projected = _project_identity_rows(("patient_id", "density", *VIEWS), rows)

    assert projected == ("locked-a", "locked-b")


@pytest.mark.parametrize(
    ("flag", "expected"),
    (("missing_view", "missing_view_identifier_count"), ("missing_patient", "missing_patient_id_count")),
)
def test_malformed_non_test_rows_block_before_image_or_role_promotion(
    tmp_path: Path, flag: str, expected: str
) -> None:
    result, _, private_root, public_report = _run(tmp_path, **{flag: True})

    assert result["status"] == "blocked"
    assert result["observations"]["metadata"][expected] == 1
    assert result["audits"]["production_dicom_decode"] == "not_performed"
    assert not (private_root / "run-1" / "role-manifest.json").exists()
    assert json.loads(public_report.read_text(encoding="utf-8")) == result


def test_locked_patient_overlap_refuses_assignment_without_exposing_identity(tmp_path: Path) -> None:
    result, _, private_root, public_report = _run(
        tmp_path, locked_patient="development-0"
    )

    encoded = public_report.read_text(encoding="utf-8")
    assert result["status"] == "blocked"
    assert result["observations"]["locked_test_isolation"]["overlap_count"] == 1
    assert result["audits"]["role_assignment"] == "not_performed"
    assert not (private_root / "run-1" / "role-manifest.json").exists()
    assert "development-0" not in encoded


def test_decoded_pixel_duplicate_blocks_manifest_and_readiness(tmp_path: Path) -> None:
    result, _, private_root, _ = _run(tmp_path, duplicate_pixels=True)

    duplicates = result["observations"]["image_audit"]["duplicates"]
    assert result["status"] == "blocked"
    assert duplicates["decoded_pixel_group_count"] == 1
    assert duplicates["decoded_pixel_extra_count"] == 1
    assert result["audits"]["role_assignment"] == "not_performed"
    assert not (private_root / "run-1" / "readiness.json").exists()


@pytest.mark.parametrize(
    ("fixture_flag", "observation", "expected"),
    (
        ("duplicate_bytes", ("duplicates", "identical_file_byte_extra_count"), 1),
        ("missing_file", ("missing_file_count",), 1),
    ),
)
def test_file_byte_duplicate_and_missing_file_block_promotion(
    tmp_path: Path, fixture_flag: str, observation: tuple[str, ...], expected: int
) -> None:
    result, _, private_root, _ = _run(tmp_path, **{fixture_flag: True})
    value = result["observations"]["image_audit"]
    for key in observation:
        value = value[key]

    assert result["status"] == "blocked"
    assert value == expected
    assert result["audits"]["role_assignment"] == "not_performed"
    assert not (private_root / "run-1" / "role-manifest.json").exists()


def test_valid_synthetic_audit_uses_accepted_role_and_readiness_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_sha256_file = rsna_audit_module.sha256_file
    locked_path = (tmp_path / "source" / "test.csv").resolve()

    def identity_safe_sha256(path) -> str:
        if Path(path).resolve() == locked_path:
            raise AssertionError("full locked test CSV was hashed outside identity projection")
        return original_sha256_file(path)

    monkeypatch.setattr(rsna_audit_module, "sha256_file", identity_safe_sha256)
    result, paths, private_root, public_report = _run(tmp_path)
    run_dir = private_root / "run-1"

    assert result["status"] == "ready"
    assert result["data_readiness"] == "verified"
    assert result["weight_compute_readiness"] == "not_assessed"
    assert result["scientific_performance"] == "not_assessed"
    assert result["observations"]["assignment"]["actual_exam_counts"] == {
        role.value: 1 for role in NON_TEST_ROLES
    }
    assert result["observations"]["inventory"]["exam_count"] == 4
    assert result["observations"]["inventory"]["locked_identity_count"] == 1

    binding_value = json.loads((run_dir / "role-manifest-binding.json").read_text())
    binding = ManifestBinding(**binding_value)
    manifest = load_private_manifest(
        run_dir / "role-manifest.json", expected=binding, private_root=private_root
    )
    readiness = load_verified_readiness_audit(run_dir / "readiness.json")
    assert len(manifest.records) == 4
    assert readiness.patient_ready is True
    assert set(readiness.role_exam_counts.values()) == {1}

    denylist = json.loads((run_dir / "locked-test-identity-denylist.json").read_text())
    assert set(denylist) == {
        "schema_version",
        "dataset_namespace",
        "patient_identity_digests",
    }
    assert "density" not in denylist
    public_text = public_report.read_text(encoding="utf-8")
    for forbidden in (
        "development-0",
        "locked-only",
        str(paths["images"]),
        str(run_dir),
    ):
        assert forbidden not in public_text


def test_private_destination_restriction_and_no_clobber(tmp_path: Path) -> None:
    paths = _fixture(tmp_path / "fixture")
    repository = Path(__file__).resolve().parents[2]
    with pytest.raises(ValueError, match="private.*ignored|external"):
        audit_rsna_readiness(
            train_manifest=paths["train"],
            validation_manifest=paths["validation"],
            locked_test_manifest=paths["test"],
            image_root=paths["images"],
            prior_inventory=paths["prior"],
            private_root=repository / "evidence" / "unsafe-private",
            public_report=tmp_path / "unsafe-public.json",
            run_name="run-1",
            requested_counts=dict(zip(NON_TEST_ROLES, (1, 1, 1, 1))),
        )

    result, _, private_root, public_report = _run(tmp_path / "first")
    before = public_report.read_bytes()
    with pytest.raises(FileExistsError, match="clobber"):
        audit_rsna_readiness(
            train_manifest=(tmp_path / "first/source/train.csv"),
            validation_manifest=(tmp_path / "first/source/validation.csv"),
            locked_test_manifest=(tmp_path / "first/source/test.csv"),
            image_root=(tmp_path / "first/source/train_images"),
            prior_inventory=(tmp_path / "first/prior.json"),
            private_root=private_root,
            public_report=public_report,
            run_name="run-1",
            requested_counts=dict(zip(NON_TEST_ROLES, (1, 1, 1, 1))),
        )
    assert result["status"] == "ready"
    assert public_report.read_bytes() == before


def test_interrupted_resume_matches_fresh_image_audit_and_partial_is_not_ready(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path / "fixture")
    resume_private = tmp_path / "resume-private"
    resume_report = tmp_path / "resume-public.json"
    with pytest.raises(AuditInterrupted, match="intentional bounded interruption"):
        _audit_fixture(
            paths,
            private_root=resume_private,
            public_report=resume_report,
            run_name="resume-run",
            stop_after_new_records=2,
        )

    run_dir = resume_private / "resume-run"
    progress = json.loads((tmp_path / "resume-public-progress.json").read_text())
    assert progress["status"] == "in_progress"
    assert progress["completed_image_count"] == 8
    assert progress["readiness_promoted"] is False
    assert not resume_report.exists()
    assert not (run_dir / "role-manifest.json").exists()
    assert not (run_dir / "readiness.json").exists()

    resumed = _audit_fixture(
        paths,
        private_root=resume_private,
        public_report=resume_report,
        run_name="resume-run",
        resume=True,
    )
    fresh = _audit_fixture(
        paths,
        private_root=tmp_path / "fresh-private",
        public_report=tmp_path / "fresh-public.json",
        run_name="fresh-run",
    )

    assert resumed["status"] == fresh["status"] == "ready"
    assert resumed["observations"]["image_audit"] == fresh["observations"]["image_audit"]
    assert resumed["observations"]["assignment"] == fresh["observations"]["assignment"]
    assert resumed["observations"]["inventory"] == fresh["observations"]["inventory"]
    assert resumed["observations"]["resume"]["reused_image_count"] == 8
    assert resumed["aggregate_artifact_hashes"]["dicom_byte_collection"] == fresh[
        "aggregate_artifact_hashes"
    ]["dicom_byte_collection"]
    assert resumed["aggregate_artifact_hashes"]["decoded_pixel_collection"] == fresh[
        "aggregate_artifact_hashes"
    ]["decoded_pixel_collection"]


def test_resume_rejects_changed_image_unless_explicitly_reaudited(tmp_path: Path) -> None:
    paths = _fixture(tmp_path / "fixture")
    private_root = tmp_path / "private"
    public_report = tmp_path / "public.json"
    with pytest.raises(AuditInterrupted):
        _audit_fixture(
            paths,
            private_root=private_root,
            public_report=public_report,
            run_name="run",
            stop_after_new_records=1,
        )
    _write_dicom(paths["images"] / "development-0" / "image-0-0.dcm", 99)

    with pytest.raises(ValueError, match="changed completed image.*re-audit"):
        _audit_fixture(
            paths,
            private_root=private_root,
            public_report=public_report,
            run_name="run",
            resume=True,
        )
    result = _audit_fixture(
        paths,
        private_root=private_root,
        public_report=public_report,
        run_name="run",
        resume=True,
        re_audit_changed=True,
    )

    assert result["status"] == "ready"
    assert result["observations"]["resume"]["changed_image_count"] == 1
    assert result["observations"]["resume"]["reaudited_image_count"] == 1
    journal_lines = (
        private_root / "run" / "image-audit-journal.jsonl"
    ).read_text(encoding="utf-8").splitlines()[1:]
    entries = [json.loads(line) for line in journal_lines]
    assert [entry["sequence"] for entry in entries] == list(range(1, len(entries) + 1))
    assert max(entry["revision"] for entry in entries) == 2


def test_resume_rejects_changed_source_binding(tmp_path: Path) -> None:
    paths = _fixture(tmp_path / "fixture")
    private_root = tmp_path / "private"
    public_report = tmp_path / "public.json"
    with pytest.raises(AuditInterrupted):
        _audit_fixture(
            paths,
            private_root=private_root,
            public_report=public_report,
            run_name="run",
            stop_after_new_records=1,
        )
    with paths["train"].open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(ValueError, match="journal binding.*source"):
        _audit_fixture(
            paths,
            private_root=private_root,
            public_report=public_report,
            run_name="run",
            resume=True,
        )


def test_resume_rejects_corrupt_journal_but_discards_incomplete_final_append(
    tmp_path: Path,
) -> None:
    corrupt_paths = _fixture(tmp_path / "corrupt-fixture")
    corrupt_private = tmp_path / "corrupt-private"
    with pytest.raises(AuditInterrupted):
        _audit_fixture(
            corrupt_paths,
            private_root=corrupt_private,
            public_report=tmp_path / "corrupt-public.json",
            run_name="run",
            stop_after_new_records=1,
        )
    corrupt_journal = corrupt_private / "run" / "image-audit-journal.jsonl"
    with corrupt_journal.open("ab") as handle:
        handle.write(b"not-json\n")
    with pytest.raises(ValueError, match="journal is corrupt"):
        _audit_fixture(
            corrupt_paths,
            private_root=corrupt_private,
            public_report=tmp_path / "corrupt-public.json",
            run_name="run",
            resume=True,
        )

    partial_paths = _fixture(tmp_path / "partial-fixture")
    partial_private = tmp_path / "partial-private"
    partial_report = tmp_path / "partial-public.json"
    with pytest.raises(AuditInterrupted):
        _audit_fixture(
            partial_paths,
            private_root=partial_private,
            public_report=partial_report,
            run_name="run",
            stop_after_new_records=1,
        )
    partial_journal = partial_private / "run" / "image-audit-journal.jsonl"
    with partial_journal.open("ab") as handle:
        handle.write(b'{"kind":"image"')

    result = _audit_fixture(
        partial_paths,
        private_root=partial_private,
        public_report=partial_report,
        run_name="run",
        resume=True,
    )
    assert result["status"] == "ready"
    assert result["observations"]["resume"]["discarded_incomplete_final_append_count"] == 1
