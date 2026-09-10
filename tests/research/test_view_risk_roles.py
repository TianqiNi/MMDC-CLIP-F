from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from mmdc_clip_f.research.view_risk.roles import (
    DEFAULT_ROLE_COUNTS,
    LockedPatientIdentityDenylist,
    ManifestBinding,
    Operation,
    PatientMappingDeclaration,
    PrivateExamRecord,
    Role,
    RoleManifest,
    ViewReference,
    assign_patient_roles,
    load_private_manifest,
    patient_identity_digest,
    run_with_role_access,
    save_private_manifest,
    validate_inventory,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
NON_TEST_ROLES = (
    Role.CLASSIFIER_FIT,
    Role.CONFIDENCE_FIT,
    Role.TUNE,
    Role.PILOT,
)


def mapping(namespace: str = "synthetic") -> PatientMappingDeclaration:
    return PatientMappingDeclaration(
        dataset_namespace=namespace,
        mapping_source_sha256=HASH_A,
        verification_method="custodian_verified_exam_to_patient",
        verification_authority="synthetic fixture custodian",
    )


def exam(
    exam_key: str,
    patient_key: str,
    density: int,
    *,
    namespace: str = "synthetic",
    source_manifest: str = "development",
    role: Role | None = None,
    image_prefix: str | None = None,
    content_hash: str | None = None,
) -> PrivateExamRecord:
    prefix = image_prefix or exam_key
    return PrivateExamRecord(
        dataset_namespace=namespace,
        exam_key=exam_key,
        patient_key=patient_key,
        density=density,
        source_manifest=source_manifest,
        views={
            view: ViewReference(
                image_id=f"{prefix}-{view}",
                path=f"private/{prefix}/{view}.img",
                content_sha256=content_hash if view == "L_CC" else None,
            )
            for view in ("L_CC", "L_MLO", "R_CC", "R_MLO")
        },
        role=role,
    )


def assigned_manifest(*records: PrivateExamRecord, namespace: str = "synthetic") -> RoleManifest:
    return RoleManifest(
        dataset_namespace=namespace,
        source_hashes={"development": HASH_B},
        patient_mapping=mapping(namespace),
        records=records,
    )


def roles_by_exam(result) -> dict[str, Role]:
    return {record.exam_key: record.role for record in result.manifest.records}


def test_protocol_default_targets_and_feasible_single_exam_counts_are_exact() -> None:
    assert DEFAULT_ROLE_COUNTS["RSNA"] == {
        Role.CLASSIFIER_FIT: 3209,
        Role.CONFIDENCE_FIT: 987,
        Role.TUNE: 370,
        Role.PILOT: 371,
    }
    assert DEFAULT_ROLE_COUNTS["DDSM"] == {
        Role.CLASSIFIER_FIT: 879,
        Role.CONFIDENCE_FIT: 270,
        Role.TUNE: 102,
        Role.PILOT: 102,
    }

    records = [exam(f"e{i:02}", f"p{i:02}", i % 4) for i in range(20)]
    targets = dict(zip(NON_TEST_ROLES, (9, 5, 3, 3)))

    result = assign_patient_roles(
        records,
        patient_mapping=mapping(),
        source_hashes={"development": HASH_B},
        counts=targets,
    )

    assert result.summary.requested_exam_counts == targets
    assert result.summary.actual_exam_counts == targets
    assert result.summary.group_constraints.targets_exact is True
    assert sum(result.summary.actual_patient_counts.values()) == 20


def test_synthetic_proportions_use_deterministic_largest_remainder_counts() -> None:
    records = [exam(f"e{i}", f"p{i}", i % 4) for i in range(11)]

    result = assign_patient_roles(
        records,
        patient_mapping=mapping(),
        source_hashes={"development": HASH_B},
        proportions=dict(zip(NON_TEST_ROLES, (0.5, 0.25, 0.125, 0.125))),
    )

    assert result.summary.requested_exam_counts == dict(zip(NON_TEST_ROLES, (6, 3, 1, 1)))
    assert result.summary.actual_exam_counts == result.summary.requested_exam_counts


def test_assignment_groups_repeated_exams_and_reports_conflicting_patient_labels() -> None:
    records = [
        exam("e0", "p-shared", 1),
        exam("e1", "p-shared", 3),
        exam("e2", "p2", 0),
        exam("e3", "p3", 2),
    ]

    result = assign_patient_roles(
        records,
        patient_mapping=mapping(),
        source_hashes={"development": HASH_B},
        counts=dict(zip(NON_TEST_ROLES, (2, 2, 0, 0))),
    )
    assigned = roles_by_exam(result)

    assert assigned["e0"] == assigned["e1"]
    assert result.summary.conflicting_patient_count == 1
    assert result.summary.patient_stratum_counts[3] >= 1
    assert result.summary.stratification_rule == "majority density; highest density on ties"
    assert sum(sum(counts.values()) for counts in result.summary.class_exam_counts.values()) == 4


def test_assignment_is_seeded_and_independent_of_input_row_order() -> None:
    records = [exam(f"e{i:02}", f"p{i:02}", i % 4) for i in range(32)]
    targets = dict(zip(NON_TEST_ROLES, (12, 8, 6, 6)))

    forward = assign_patient_roles(
        records,
        patient_mapping=mapping(),
        source_hashes={"development": HASH_B},
        counts=targets,
        seed=42,
    )
    reversed_rows = assign_patient_roles(
        list(reversed(records)),
        patient_mapping=mapping(),
        source_hashes={"development": HASH_B},
        counts=targets,
        seed=42,
    )
    another_seed = assign_patient_roles(
        records,
        patient_mapping=mapping(),
        source_hashes={"development": HASH_B},
        counts=targets,
        seed=43,
    )

    assert roles_by_exam(forward) == roles_by_exam(reversed_rows)
    assert roles_by_exam(forward) != roles_by_exam(another_seed)


def test_indivisible_multi_exam_groups_report_requested_and_actual_counts() -> None:
    records = [
        *(exam(f"a{i}", "patient-a", i % 2) for i in range(3)),
        *(exam(f"b{i}", "patient-b", 2) for i in range(2)),
    ]
    requested = dict(zip(NON_TEST_ROLES, (2, 1, 1, 1)))

    result = assign_patient_roles(
        records,
        patient_mapping=mapping(),
        source_hashes={"development": HASH_B},
        counts=requested,
    )

    by_patient: dict[str, set[Role]] = {}
    for record in result.manifest.records:
        by_patient.setdefault(record.patient_key, set()).add(record.role)
    assert all(len(patient_roles) == 1 for patient_roles in by_patient.values())
    assert result.summary.requested_exam_counts == requested
    assert sum(result.summary.actual_exam_counts.values()) == 5
    assert result.summary.actual_exam_counts != requested
    assert result.summary.group_constraints.targets_exact is False
    assert result.summary.group_constraints.multi_exam_patient_count == 2
    assert result.summary.group_constraints.max_exams_per_patient == 3


def test_absent_patient_mapping_and_invalid_private_records_fail_closed() -> None:
    with pytest.raises(ValueError, match="mapping declaration"):
        assign_patient_roles(
            [exam("e0", "p0", 0)],
            patient_mapping=None,
            source_hashes={"development": HASH_B},
            counts=dict(zip(NON_TEST_ROLES, (1, 0, 0, 0))),
        )

    with pytest.raises(ValueError, match="density"):
        exam("private-exam", "private-patient", 4)
    with pytest.raises(ValueError, match="canonical"):
        replace(exam("private-exam", "private-patient", 0), views={"L_CC": ViewReference("x", "x")})


def test_manifest_rejects_duplicate_exams_patient_overlap_and_bad_provenance_without_ids() -> None:
    first = exam("private-exam", "private-patient", 0, role=Role.CLASSIFIER_FIT)

    cases = [
        (first, replace(first, patient_key="another-patient")),
        (first, replace(exam("other-exam", "private-patient", 1), role=Role.PILOT)),
        (
            first,
            replace(
                exam("other-exam", "other-patient", 1), source_manifest="unknown", role=Role.PILOT
            ),
        ),
    ]
    for records in cases:
        with pytest.raises(ValueError) as caught:
            assigned_manifest(*records)
        assert "private-exam" not in str(caught.value)
        assert "private-patient" not in str(caught.value)


@pytest.mark.parametrize("collision_field", ["image_id", "content_sha256"])
def test_inventory_rejects_cross_role_image_or_content_collisions(collision_field: str) -> None:
    first = exam(
        "e0",
        "p0",
        0,
        role=Role.CLASSIFIER_FIT,
        image_prefix="shared" if collision_field == "image_id" else "left",
        content_hash=HASH_A if collision_field == "content_sha256" else None,
    )
    second = exam(
        "e1",
        "p1",
        1,
        role=Role.PILOT,
        image_prefix="shared" if collision_field == "image_id" else "right",
        content_hash=HASH_A if collision_field == "content_sha256" else None,
    )

    with pytest.raises(ValueError, match="cross-role") as caught:
        assigned_manifest(first, second)
    assert "shared" not in str(caught.value)


def test_inventory_rejects_conflicting_provenance_within_one_dataset_namespace() -> None:
    first = assigned_manifest(exam("e0", "p0", 0, role=Role.CLASSIFIER_FIT))
    second = RoleManifest(
        dataset_namespace="synthetic",
        source_hashes={"development": HASH_A},
        patient_mapping=mapping(),
        records=(exam("e1", "p1", 1, role=Role.PILOT),),
    )

    with pytest.raises(ValueError, match="provenance"):
        validate_inventory((first, second))


def test_inventory_rejects_namespaced_path_reuse_across_manifests_and_roles() -> None:
    first_record = exam("e0", "p0", 0, role=Role.CLASSIFIER_FIT)
    second_record = exam("e1", "p1", 1, role=Role.PILOT)
    second_views = dict(second_record.views)
    second_views["L_CC"] = replace(second_views["L_CC"], path=first_record.views["L_CC"].path)
    second_record = replace(second_record, views=second_views)

    first = assigned_manifest(first_record)
    second = assigned_manifest(second_record)

    with pytest.raises(ValueError, match="cross-role.*path") as caught:
        validate_inventory((first, second))
    assert first_record.views["L_CC"].path not in str(caught.value)


def test_dataset_namespaces_scope_identifiers_but_content_hashes_remain_global() -> None:
    left = assigned_manifest(
        exam("same-exam", "same-patient", 0, namespace="A", role=Role.CLASSIFIER_FIT),
        namespace="A",
    )
    right = assigned_manifest(
        exam("same-exam", "same-patient", 1, namespace="B", role=Role.PILOT),
        namespace="B",
    )

    summary = validate_inventory((left, right))

    assert summary.dataset_count == 2
    assert summary.exam_count == 2
    assert summary.image_id_collision_count == 0


def test_identity_only_locked_denylist_detects_patient_overlap_without_outcomes() -> None:
    manifest = assigned_manifest(exam("e0", "p0", 0, role=Role.PILOT))
    overlapping = LockedPatientIdentityDenylist(
        dataset_namespace="synthetic",
        source_sha256=HASH_B,
        patient_identity_digests=frozenset({patient_identity_digest("synthetic", "p0")}),
    )

    with pytest.raises(ValueError, match="locked patient overlap") as caught:
        validate_inventory((manifest,), locked_patient_denylists=(overlapping,))
    assert "p0" not in str(caught.value)

    disjoint = replace(
        overlapping,
        patient_identity_digests=frozenset(
            {patient_identity_digest("synthetic", "another-patient")}
        ),
    )
    summary = validate_inventory((manifest,), locked_patient_denylists=(disjoint,))
    assert summary.locked_denylist_dataset_count == 1
    assert summary.locked_identity_count == 1


def test_role_guard_refuses_forbidden_and_locked_roles_before_loader_runs() -> None:
    manifest = assigned_manifest(
        exam("fit", "p-fit", 0, role=Role.CLASSIFIER_FIT),
        exam("confidence", "p-confidence", 1, role=Role.CONFIDENCE_FIT),
        exam("pilot", "p-pilot", 2, role=Role.PILOT),
    )
    calls = 0

    def loader(records):
        nonlocal calls
        calls += 1
        return len(records)

    with pytest.raises(PermissionError, match="not permitted"):
        run_with_role_access(
            manifest,
            operation=Operation.CLASSIFIER_FITTING,
            roles=(Role.CONFIDENCE_FIT,),
            loader=loader,
        )
    with pytest.raises(PermissionError, match="locked_test"):
        run_with_role_access(
            manifest,
            operation=Operation.PILOT_EVALUATION,
            roles=(Role.LOCKED_TEST,),
            loader=loader,
        )
    assert calls == 0

    assert (
        run_with_role_access(
            manifest,
            operation=Operation.PILOT_EVALUATION,
            roles=(Role.PILOT,),
            loader=loader,
        )
        == 1
    )
    assert calls == 1


def test_outcome_bearing_locked_record_is_rejected_during_ordinary_construction() -> None:
    with pytest.raises(PermissionError, match="locked_test.*outcome"):
        exam("locked", "p-locked", 3, role=Role.LOCKED_TEST)


def test_ordinary_save_revalidates_and_refuses_a_locked_manifest(tmp_path) -> None:
    manifest = assigned_manifest(exam("e0", "p0", 0, role=Role.CLASSIFIER_FIT))
    object.__setattr__(manifest.records[0], "role", Role.LOCKED_TEST)
    path = tmp_path / "private" / "roles.json"

    with pytest.raises(PermissionError, match="locked_test.*outcome"):
        save_private_manifest(manifest, path, private_root=tmp_path)
    assert not path.exists()


def test_self_consistently_hashed_locked_manifest_is_rejected_on_load(tmp_path) -> None:
    manifest = assigned_manifest(exam("e0", "p0", 0, role=Role.CLASSIFIER_FIT))
    path = tmp_path / "private" / "roles.json"
    binding = save_private_manifest(manifest, path, private_root=tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["manifest"]["records"][0]["role"] = Role.LOCKED_TEST.value
    encoded = json.dumps(
        document["manifest"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    document["manifest_sha256"] = digest
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(PermissionError, match="locked_test.*outcome"):
        load_private_manifest(
            path,
            expected=replace(binding, manifest_sha256=digest),
            private_root=tmp_path,
        )


def test_private_manifest_round_trip_and_tamper_or_stale_binding_refusal(tmp_path) -> None:
    result = assign_patient_roles(
        [exam(f"e{i}", f"p{i}", i % 4) for i in range(8)],
        patient_mapping=mapping(),
        source_hashes={"development": HASH_B},
        counts=dict(zip(NON_TEST_ROLES, (3, 2, 2, 1))),
    )
    path = tmp_path / "private" / "roles.json"

    binding = save_private_manifest(result.manifest, path, private_root=tmp_path)
    loaded = load_private_manifest(path, expected=binding, private_root=tmp_path)

    assert [(r.exam_key, r.role) for r in loaded.records] == [
        (r.exam_key, r.role) for r in result.manifest.records
    ]

    stale = replace(binding, source_hashes={"development": HASH_A})
    with pytest.raises(ValueError, match="provenance"):
        load_private_manifest(path, expected=stale, private_root=tmp_path)

    document = json.loads(path.read_text(encoding="utf-8"))
    document["manifest"]["records"][0]["role"] = Role.LOCKED_TEST.value
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        load_private_manifest(path, expected=binding, private_root=tmp_path)


def test_private_manifest_cannot_be_written_outside_declared_private_root(tmp_path) -> None:
    manifest = assigned_manifest(exam("e0", "p0", 0, role=Role.CLASSIFIER_FIT))

    with pytest.raises(ValueError, match="private root"):
        save_private_manifest(manifest, tmp_path / ".." / "roles.json", private_root=tmp_path)


def test_manifest_binding_is_not_a_generic_locked_test_release_switch() -> None:
    assert "allow_test" not in ManifestBinding.__dataclass_fields__
    assert set(Operation) == {
        Operation.CLASSIFIER_FITTING,
        Operation.CONFIDENCE_FITTING,
        Operation.TUNE_SELECTION,
        Operation.PILOT_EVALUATION,
    }
