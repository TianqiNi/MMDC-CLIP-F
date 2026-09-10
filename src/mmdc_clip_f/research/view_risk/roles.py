"""Private patient-role manifests and fail-closed research data access.

The record types in this module contain linkable identifiers and paths.  They
belong in private artifacts, never in aggregate evidence or repository files.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

from .inputs import CANONICAL_VIEWS


SCHEMA_VERSION = "view-risk-role-manifest/v1"
_ENVELOPE_VERSION = "view-risk-private-manifest-envelope/v1"


class Role(str, Enum):
    CLASSIFIER_FIT = "classifier_fit"
    CONFIDENCE_FIT = "confidence_fit"
    TUNE = "tune"
    PILOT = "pilot"
    LOCKED_TEST = "locked_test"


class Operation(str, Enum):
    CLASSIFIER_FITTING = "classifier_fitting"
    CONFIDENCE_FITTING = "confidence_fitting"
    TUNE_SELECTION = "tune_selection"
    PILOT_EVALUATION = "pilot_evaluation"


NON_TEST_ROLES = (
    Role.CLASSIFIER_FIT,
    Role.CONFIDENCE_FIT,
    Role.TUNE,
    Role.PILOT,
)

DEFAULT_ROLE_COUNTS: Mapping[str, Mapping[Role, int]] = MappingProxyType(
    {
        "RSNA": MappingProxyType(
            {
                Role.CLASSIFIER_FIT: 3209,
                Role.CONFIDENCE_FIT: 987,
                Role.TUNE: 370,
                Role.PILOT: 371,
            }
        ),
        "DDSM": MappingProxyType(
            {
                Role.CLASSIFIER_FIT: 879,
                Role.CONFIDENCE_FIT: 270,
                Role.TUNE: 102,
                Role.PILOT: 102,
            }
        ),
    }
)

OPERATION_ROLE_ALLOWLIST: Mapping[Operation, frozenset[Role]] = MappingProxyType(
    {
        Operation.CLASSIFIER_FITTING: frozenset({Role.CLASSIFIER_FIT}),
        Operation.CONFIDENCE_FITTING: frozenset({Role.CONFIDENCE_FIT}),
        Operation.TUNE_SELECTION: frozenset({Role.TUNE}),
        Operation.PILOT_EVALUATION: frozenset({Role.PILOT}),
    }
)


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _required_private_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"private record has missing {field}")
    return value.strip()


def _snapshot_source_hashes(source_hashes: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError("manifest provenance requires source hashes")
    snapshot: dict[str, str] = {}
    for name, digest in source_hashes.items():
        key = _required_private_string(name, "source manifest name")
        if key in snapshot:
            raise ValueError("manifest provenance contains duplicate source names")
        if not _is_sha256(digest):
            raise ValueError("manifest provenance contains an invalid SHA-256 digest")
        snapshot[key] = digest
    return MappingProxyType(dict(sorted(snapshot.items())))


@dataclass(frozen=True)
class ViewReference:
    """Private identity and path for one canonical view."""

    image_id: str
    path: str
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_id", _required_private_string(self.image_id, "image ID"))
        object.__setattr__(self, "path", _required_private_string(self.path, "image path"))
        if self.content_sha256 is not None and not _is_sha256(self.content_sha256):
            raise ValueError("private view has an invalid content SHA-256 digest")


@dataclass(frozen=True)
class PrivateExamRecord:
    """One private exam, with an explicitly verified patient key.

    ``patient_key`` is meaningful only under the manifest's
    :class:`PatientMappingDeclaration`; this class never treats ``exam_key`` as
    proof of patient identity.
    """

    dataset_namespace: str
    exam_key: str
    patient_key: str
    density: int
    source_manifest: str
    views: Mapping[str, ViewReference]
    role: Role | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dataset_namespace",
            _required_private_string(self.dataset_namespace, "dataset namespace"),
        )
        object.__setattr__(self, "exam_key", _required_private_string(self.exam_key, "exam key"))
        object.__setattr__(
            self, "patient_key", _required_private_string(self.patient_key, "verified patient key")
        )
        object.__setattr__(
            self,
            "source_manifest",
            _required_private_string(self.source_manifest, "source manifest binding"),
        )
        if isinstance(self.density, bool) or not isinstance(self.density, int):
            raise ValueError("density must be an integer in [0, 3]")
        if self.density not in range(4):
            raise ValueError("density must be an integer in [0, 3]")
        if not isinstance(self.views, Mapping) or set(self.views) != set(CANONICAL_VIEWS):
            raise ValueError("each exam must contain every canonical named view exactly once")
        ordered: dict[str, ViewReference] = {}
        for name in CANONICAL_VIEWS:
            reference = self.views[name]
            if not isinstance(reference, ViewReference):
                raise TypeError("canonical views must contain ViewReference values")
            ordered[name] = reference
        if len({reference.image_id for reference in ordered.values()}) != len(CANONICAL_VIEWS):
            raise ValueError("view image IDs must be unique within each exam")
        if len({reference.path for reference in ordered.values()}) != len(CANONICAL_VIEWS):
            raise ValueError("view paths must be unique within each exam")
        object.__setattr__(self, "views", MappingProxyType(ordered))
        if self.role is not None:
            try:
                object.__setattr__(self, "role", Role(self.role))
            except (TypeError, ValueError) as exc:
                raise ValueError("private exam has an invalid role") from exc


@dataclass(frozen=True)
class PatientMappingDeclaration:
    """Attestation that exam keys were mapped to patient keys outside this module."""

    dataset_namespace: str
    mapping_source_sha256: str
    verification_method: str
    verification_authority: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dataset_namespace",
            _required_private_string(self.dataset_namespace, "mapping dataset namespace"),
        )
        if not _is_sha256(self.mapping_source_sha256):
            raise ValueError("patient mapping declaration requires a valid source SHA-256")
        method = _required_private_string(self.verification_method, "patient mapping method")
        authority = _required_private_string(
            self.verification_authority, "patient mapping verification authority"
        )
        if method.casefold() in {"exam_id", "exam_id_assumed", "assumed_exam_identity"}:
            raise ValueError("patient mapping cannot be an unverified exam-ID assumption")
        object.__setattr__(self, "verification_method", method)
        object.__setattr__(self, "verification_authority", authority)


@dataclass(frozen=True)
class RoleManifest:
    """Validated private manifest whose records are bound to roles and provenance."""

    dataset_namespace: str
    source_hashes: Mapping[str, str]
    patient_mapping: PatientMappingDeclaration
    records: Sequence[PrivateExamRecord]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        namespace = _required_private_string(self.dataset_namespace, "dataset namespace")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported role manifest schema version")
        if not isinstance(self.patient_mapping, PatientMappingDeclaration):
            raise ValueError("manifest requires an explicit patient mapping declaration")
        if self.patient_mapping.dataset_namespace != namespace:
            raise ValueError("patient mapping and manifest dataset namespaces disagree")
        sources = _snapshot_source_hashes(self.source_hashes)
        materialized = tuple(self.records)
        if not materialized:
            raise ValueError("role manifest cannot be empty")

        seen_exams: set[tuple[str, str]] = set()
        patient_roles: dict[tuple[str, str], Role] = {}
        image_roles: dict[tuple[str, str], Role] = {}
        path_roles: dict[tuple[str, str], Role] = {}
        content_roles: dict[str, Role] = {}
        for record in materialized:
            if not isinstance(record, PrivateExamRecord):
                raise TypeError("role manifest records must be PrivateExamRecord values")
            if record.dataset_namespace != namespace:
                raise ValueError("record and manifest dataset namespaces disagree")
            if record.source_manifest not in sources:
                raise ValueError("record has inconsistent manifest provenance")
            if record.role is None:
                raise ValueError("every manifest record requires an explicit role binding")
            exam_identity = (namespace, record.exam_key)
            if exam_identity in seen_exams:
                raise ValueError("manifest contains duplicate exam IDs (count >= 1)")
            seen_exams.add(exam_identity)
            patient_identity = (namespace, record.patient_key)
            previous_patient_role = patient_roles.setdefault(patient_identity, record.role)
            if previous_patient_role != record.role:
                raise ValueError("manifest contains held-out patient overlap across roles")
            for reference in record.views.values():
                _check_duplicate_across_roles(
                    image_roles,
                    (namespace, reference.image_id),
                    record.role,
                    "image ID",
                )
                _check_duplicate_across_roles(
                    path_roles,
                    (namespace, reference.path),
                    record.role,
                    "image path",
                )
                if reference.content_sha256 is not None:
                    _check_duplicate_across_roles(
                        content_roles,
                        reference.content_sha256,
                        record.role,
                        "content hash",
                    )

        ordered = tuple(sorted(materialized, key=lambda record: record.exam_key))
        object.__setattr__(self, "dataset_namespace", namespace)
        object.__setattr__(self, "source_hashes", sources)
        object.__setattr__(self, "records", ordered)

    @property
    def manifest_sha256(self) -> str:
        return _hash_payload(_manifest_payload(self))


def _check_duplicate_across_roles(
    seen: dict[Any, Role], identity: Any, role: Role, kind: str
) -> None:
    if identity not in seen:
        seen[identity] = role
        return
    prior = seen[identity]
    if prior != role:
        raise ValueError(f"manifest contains a repeated {kind} across-role collision")
    raise ValueError(f"manifest contains a repeated {kind}")


@dataclass(frozen=True)
class GroupConstraintSummary:
    targets_exact: bool
    multi_exam_patient_count: int
    max_exams_per_patient: int


@dataclass(frozen=True)
class AssignmentSummary:
    requested_exam_counts: Mapping[Role, int]
    actual_exam_counts: Mapping[Role, int]
    actual_patient_counts: Mapping[Role, int]
    class_exam_counts: Mapping[Role, Mapping[int, int]]
    patient_stratum_counts: Mapping[int, int]
    conflicting_patient_count: int
    stratification_rule: str
    group_constraints: GroupConstraintSummary
    exclusion_counts: Mapping[str, int]


@dataclass(frozen=True)
class RoleAssignment:
    manifest: RoleManifest
    summary: AssignmentSummary


def _role_mapping(values: Mapping[Role | str, int], field: str) -> dict[Role, int]:
    normalized: dict[Role, int] = {}
    for raw_role, value in values.items():
        try:
            role = Role(raw_role)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} contains an unknown role") from exc
        if role == Role.LOCKED_TEST:
            raise ValueError(f"{field} cannot assign locked_test records")
        if role in normalized:
            raise ValueError(f"{field} contains duplicate roles")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} values must be nonnegative integers")
        normalized[role] = value
    if set(normalized) != set(NON_TEST_ROLES):
        raise ValueError(f"{field} must specify all four non-test roles")
    return {role: normalized[role] for role in NON_TEST_ROLES}


def _apportion(total: int, weights: Mapping[Role, int | float]) -> dict[Role, int]:
    weight_sum = sum(weights.values())
    if total == 0:
        return {role: 0 for role in NON_TEST_ROLES}
    if weight_sum <= 0:
        raise ValueError("role proportions must have positive total weight")
    ideals = {role: total * weights[role] / weight_sum for role in NON_TEST_ROLES}
    result = {role: int(ideals[role]) for role in NON_TEST_ROLES}
    remaining = total - sum(result.values())
    order = sorted(
        NON_TEST_ROLES,
        key=lambda role: (-(ideals[role] - result[role]), NON_TEST_ROLES.index(role)),
    )
    for role in order[:remaining]:
        result[role] += 1
    return result


def _resolve_targets(
    total: int,
    namespace: str,
    counts: Mapping[Role | str, int] | None,
    proportions: Mapping[Role | str, float] | None,
) -> dict[Role, int]:
    if counts is not None and proportions is not None:
        raise ValueError("specify role counts or proportions, not both")
    if counts is not None:
        result = _role_mapping(counts, "role counts")
    elif proportions is not None:
        normalized: dict[Role, float] = {}
        for raw_role, value in proportions.items():
            role = Role(raw_role)
            if role == Role.LOCKED_TEST or role in normalized:
                raise ValueError("role proportions contain an invalid role")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError("role proportions must be nonnegative numbers")
            normalized[role] = float(value)
        if set(normalized) != set(NON_TEST_ROLES):
            raise ValueError("role proportions must specify all four non-test roles")
        result = _apportion(total, normalized)
    else:
        key = namespace.upper()
        aliases = {"RSNA-SMBC": "RSNA", "MINI-DDSM": "DDSM"}
        key = aliases.get(key, key)
        if key not in DEFAULT_ROLE_COUNTS:
            raise ValueError("role counts or proportions are required for this dataset namespace")
        result = dict(DEFAULT_ROLE_COUNTS[key])
    if sum(result.values()) != total:
        raise ValueError("requested role counts must equal the number of eligible exams")
    return result


def _patient_order(seed: int, namespace: str, patient_key: str) -> bytes:
    material = f"{SCHEMA_VERSION}\0{seed}\0{namespace}\0{patient_key}".encode("utf-8")
    return hashlib.sha256(material).digest()


def _patient_stratum(records: Sequence[PrivateExamRecord]) -> tuple[int, bool]:
    counts = Counter(record.density for record in records)
    largest = max(counts.values())
    winners = [density for density, count in counts.items() if count == largest]
    return max(winners), len(counts) > 1


def assign_patient_roles(
    records: Iterable[PrivateExamRecord],
    *,
    patient_mapping: PatientMappingDeclaration | None,
    source_hashes: Mapping[str, str],
    counts: Mapping[Role | str, int] | None = None,
    proportions: Mapping[Role | str, float] | None = None,
    seed: int = 42,
) -> RoleAssignment:
    """Assign non-test exams by verified patient and patient-level density stratum.

    A patient's stratum is its majority exam density, with the highest density
    breaking a tie.  Assignment is deterministic for a seed and independent of
    input row order.  Indivisible multi-exam groups may prevent exact targets;
    achieved counts are always reported and patients are never split.
    """

    materialized = tuple(records)
    if patient_mapping is None:
        raise ValueError("assignment requires an explicit patient mapping declaration")
    if not materialized:
        raise ValueError("cannot assign an empty exam inventory")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    namespace = materialized[0].dataset_namespace
    if patient_mapping.dataset_namespace != namespace:
        raise ValueError("patient mapping and record dataset namespaces disagree")
    sources = _snapshot_source_hashes(source_hashes)

    seen_exams: set[tuple[str, str]] = set()
    patients: dict[str, list[PrivateExamRecord]] = defaultdict(list)
    for record in materialized:
        if not isinstance(record, PrivateExamRecord):
            raise TypeError("assignment records must be PrivateExamRecord values")
        if record.dataset_namespace != namespace:
            raise ValueError("assignment cannot mix dataset namespaces")
        if record.role is not None:
            raise ValueError("assignment inputs must not have pre-existing role bindings")
        if record.source_manifest not in sources:
            raise ValueError("record has inconsistent manifest provenance")
        identity = (namespace, record.exam_key)
        if identity in seen_exams:
            raise ValueError("assignment input contains duplicate exam IDs (count >= 1)")
        seen_exams.add(identity)
        patients[record.patient_key].append(record)

    requested = _resolve_targets(len(materialized), namespace, counts, proportions)
    grouped: dict[int, list[tuple[str, tuple[PrivateExamRecord, ...]]]] = defaultdict(list)
    conflicting = 0
    for patient_key, patient_records in patients.items():
        ordered_records = tuple(sorted(patient_records, key=lambda record: record.exam_key))
        stratum, has_conflict = _patient_stratum(ordered_records)
        conflicting += int(has_conflict)
        grouped[stratum].append((patient_key, ordered_records))

    actual = {role: 0 for role in NON_TEST_ROLES}
    patient_counts = {role: 0 for role in NON_TEST_ROLES}
    class_counts = {role: {density: 0 for density in range(4)} for role in NON_TEST_ROLES}
    patient_strata = {density: len(grouped[density]) for density in range(4)}
    assigned: list[PrivateExamRecord] = []

    for density in range(4):
        groups = sorted(grouped[density], key=lambda item: _patient_order(seed, namespace, item[0]))
        stratum_exams = sum(len(patient_records) for _, patient_records in groups)
        remaining_capacity = {
            role: max(requested[role] - actual[role], 0) for role in NON_TEST_ROLES
        }
        if stratum_exams and sum(remaining_capacity.values()) == 0:
            class_target = {role: 0 for role in NON_TEST_ROLES}
        else:
            class_target = _apportion(stratum_exams, remaining_capacity)

        class_assigned = {role: 0 for role in NON_TEST_ROLES}
        for _, patient_records in groups:
            size = len(patient_records)
            if size == 1:
                candidates = [role for role in NON_TEST_ROLES if actual[role] < requested[role]]
                if not candidates:
                    candidates = list(NON_TEST_ROLES)
                role = max(
                    candidates,
                    key=lambda candidate: (
                        class_target[candidate] - class_assigned[candidate],
                        requested[candidate] - actual[candidate],
                        -NON_TEST_ROLES.index(candidate),
                    ),
                )
            else:
                role = min(
                    NON_TEST_ROLES,
                    key=lambda candidate: (
                        abs(actual[candidate] + size - requested[candidate])
                        + abs(class_assigned[candidate] + size - class_target[candidate]),
                        max(actual[candidate] + size - requested[candidate], 0),
                        NON_TEST_ROLES.index(candidate),
                    ),
                )
            actual[role] += size
            patient_counts[role] += 1
            class_assigned[role] += size
            for record in patient_records:
                class_counts[role][record.density] += 1
                assigned.append(replace(record, role=role))

    manifest = RoleManifest(
        dataset_namespace=namespace,
        source_hashes=sources,
        patient_mapping=patient_mapping,
        records=assigned,
    )
    group_sizes = [len(patient_records) for patient_records in patients.values()]
    summary = AssignmentSummary(
        requested_exam_counts=MappingProxyType(dict(requested)),
        actual_exam_counts=MappingProxyType(dict(actual)),
        actual_patient_counts=MappingProxyType(dict(patient_counts)),
        class_exam_counts=MappingProxyType(
            {
                role: MappingProxyType(dict(counts_by_class))
                for role, counts_by_class in class_counts.items()
            }
        ),
        patient_stratum_counts=MappingProxyType(patient_strata),
        conflicting_patient_count=conflicting,
        stratification_rule="majority density; highest density on ties",
        group_constraints=GroupConstraintSummary(
            targets_exact=actual == requested,
            multi_exam_patient_count=sum(size > 1 for size in group_sizes),
            max_exams_per_patient=max(group_sizes),
        ),
        exclusion_counts=MappingProxyType({}),
    )
    return RoleAssignment(manifest=manifest, summary=summary)


@dataclass(frozen=True)
class InventorySummary:
    dataset_count: int
    exam_count: int
    patient_count: int
    role_exam_counts: Mapping[Role, int]
    class_exam_counts: Mapping[int, int]
    image_count: int
    content_hash_count: int
    image_id_collision_count: int
    content_collision_count: int


def validate_inventory(manifests: Iterable[RoleManifest]) -> InventorySummary:
    """Validate one or more private manifests and return only aggregate counts."""

    materialized = tuple(manifests)
    if not materialized:
        raise ValueError("inventory requires at least one manifest")
    exams: set[tuple[str, str]] = set()
    patients: set[tuple[str, str]] = set()
    patient_roles: dict[tuple[str, str], Role] = {}
    image_roles: dict[tuple[str, str], Role] = {}
    content_roles: dict[str, Role] = {}
    roles: Counter[Role] = Counter()
    classes: Counter[int] = Counter()
    image_count = 0
    content_count = 0
    namespaces: set[str] = set()
    namespace_provenance: dict[
        str, tuple[tuple[tuple[str, str], ...], PatientMappingDeclaration]
    ] = {}
    for manifest in materialized:
        if not isinstance(manifest, RoleManifest):
            raise TypeError("inventory entries must be RoleManifest values")
        namespaces.add(manifest.dataset_namespace)
        provenance = (tuple(manifest.source_hashes.items()), manifest.patient_mapping)
        prior_provenance = namespace_provenance.setdefault(manifest.dataset_namespace, provenance)
        if prior_provenance != provenance:
            raise ValueError("inventory has inconsistent provenance for one dataset namespace")
        for record in manifest.records:
            exam_identity = (manifest.dataset_namespace, record.exam_key)
            if exam_identity in exams:
                raise ValueError("inventory contains duplicate namespaced exam IDs")
            exams.add(exam_identity)
            patient_identity = (manifest.dataset_namespace, record.patient_key)
            prior_role = patient_roles.setdefault(patient_identity, record.role)  # type: ignore[arg-type]
            if prior_role != record.role:
                raise ValueError("inventory contains held-out patient overlap across roles")
            patients.add(patient_identity)
            roles[record.role] += 1  # type: ignore[index]
            classes[record.density] += 1
            for reference in record.views.values():
                image_count += 1
                image_identity = (manifest.dataset_namespace, reference.image_id)
                if image_identity in image_roles:
                    prior = image_roles[image_identity]
                    kind = "cross-role " if prior != record.role else ""
                    raise ValueError(f"inventory contains a {kind}image ID collision")
                image_roles[image_identity] = record.role  # type: ignore[assignment]
                if reference.content_sha256 is not None:
                    content_count += 1
                    if reference.content_sha256 in content_roles:
                        prior = content_roles[reference.content_sha256]
                        kind = "cross-role " if prior != record.role else ""
                        raise ValueError(f"inventory contains a {kind}content hash collision")
                    content_roles[reference.content_sha256] = record.role  # type: ignore[assignment]
    return InventorySummary(
        dataset_count=len(namespaces),
        exam_count=len(exams),
        patient_count=len(patients),
        role_exam_counts=MappingProxyType({role: roles[role] for role in Role}),
        class_exam_counts=MappingProxyType({density: classes[density] for density in range(4)}),
        image_count=image_count,
        content_hash_count=content_count,
        image_id_collision_count=0,
        content_collision_count=0,
    )


def _view_payload(reference: ViewReference) -> dict[str, Any]:
    return {
        "image_id": reference.image_id,
        "path": reference.path,
        "content_sha256": reference.content_sha256,
    }


def _record_payload(record: PrivateExamRecord) -> dict[str, Any]:
    return {
        "dataset_namespace": record.dataset_namespace,
        "exam_key": record.exam_key,
        "patient_key": record.patient_key,
        "density": record.density,
        "source_manifest": record.source_manifest,
        "views": {name: _view_payload(record.views[name]) for name in CANONICAL_VIEWS},
        "role": record.role.value if record.role is not None else None,
    }


def _manifest_payload(manifest: RoleManifest) -> dict[str, Any]:
    declaration = manifest.patient_mapping
    return {
        "schema_version": manifest.schema_version,
        "dataset_namespace": manifest.dataset_namespace,
        "source_hashes": dict(manifest.source_hashes),
        "patient_mapping": {
            "dataset_namespace": declaration.dataset_namespace,
            "mapping_source_sha256": declaration.mapping_source_sha256,
            "verification_method": declaration.verification_method,
            "verification_authority": declaration.verification_authority,
        },
        "records": [_record_payload(record) for record in manifest.records],
    }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _hash_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


@dataclass(frozen=True)
class ManifestBinding:
    schema_version: str
    dataset_namespace: str
    source_hashes: Mapping[str, str]
    manifest_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported manifest binding schema")
        object.__setattr__(
            self,
            "dataset_namespace",
            _required_private_string(self.dataset_namespace, "binding dataset namespace"),
        )
        object.__setattr__(self, "source_hashes", _snapshot_source_hashes(self.source_hashes))
        if not _is_sha256(self.manifest_sha256):
            raise ValueError("manifest binding has an invalid SHA-256")


def _private_path(path: str | Path, private_root: str | Path) -> Path:
    root = Path(private_root).expanduser().resolve()
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "private manifest path must remain under the declared private root"
        ) from exc
    if resolved == root:
        raise ValueError("private manifest path must be a file under the declared private root")
    return resolved


def save_private_manifest(
    manifest: RoleManifest,
    path: str | Path,
    *,
    private_root: str | Path,
) -> ManifestBinding:
    """Write a private manifest and return the binding safe for downstream checks."""

    if not isinstance(manifest, RoleManifest):
        raise TypeError("manifest must be a RoleManifest")
    target = _private_path(path, private_root)
    payload = _manifest_payload(manifest)
    digest = _hash_payload(payload)
    document = {
        "envelope_version": _ENVELOPE_VERSION,
        "manifest_sha256": digest,
        "manifest": payload,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_canonical_json(document) + b"\n")
    target.chmod(0o600)
    return ManifestBinding(
        schema_version=manifest.schema_version,
        dataset_namespace=manifest.dataset_namespace,
        source_hashes=manifest.source_hashes,
        manifest_sha256=digest,
    )


def _manifest_from_payload(payload: object) -> RoleManifest:
    if not isinstance(payload, dict):
        raise ValueError("private manifest payload must be an object")
    expected_fields = {
        "schema_version",
        "dataset_namespace",
        "source_hashes",
        "patient_mapping",
        "records",
    }
    if set(payload) != expected_fields:
        raise ValueError("private manifest payload has an invalid schema")
    raw_records = payload["records"]
    if not isinstance(raw_records, list):
        raise ValueError("private manifest records must be an ordered list")
    ordering: list[tuple[str, str]] = []
    records: list[PrivateExamRecord] = []
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            raise ValueError("private manifest contains an invalid record")
        try:
            raw_views = raw_record["views"]
            if not isinstance(raw_views, dict):
                raise ValueError("private manifest contains invalid named views")
            views = {
                name: ViewReference(**raw_views[name])
                for name in CANONICAL_VIEWS
                if name in raw_views
            }
            record = PrivateExamRecord(
                dataset_namespace=raw_record["dataset_namespace"],
                exam_key=raw_record["exam_key"],
                patient_key=raw_record["patient_key"],
                density=raw_record["density"],
                source_manifest=raw_record["source_manifest"],
                views=views,
                role=raw_record["role"],
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("private manifest contains an invalid record schema") from exc
        records.append(record)
        ordering.append((record.dataset_namespace, record.exam_key))
    if ordering != sorted(ordering):
        raise ValueError("private manifest record order is not canonical")
    try:
        mapping = PatientMappingDeclaration(**payload["patient_mapping"])
        return RoleManifest(
            schema_version=payload["schema_version"],
            dataset_namespace=payload["dataset_namespace"],
            source_hashes=payload["source_hashes"],
            patient_mapping=mapping,
            records=records,
        )
    except TypeError as exc:
        raise ValueError("private manifest contains invalid provenance schema") from exc


def load_private_manifest(
    path: str | Path,
    *,
    expected: ManifestBinding,
    private_root: str | Path,
) -> RoleManifest:
    """Load only a canonically ordered manifest matching an external binding."""

    if not isinstance(expected, ManifestBinding):
        raise TypeError("expected must be a ManifestBinding")
    source = _private_path(path, private_root)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("private manifest could not be read as canonical JSON") from exc
    if not isinstance(document, dict) or document.get("envelope_version") != _ENVELOPE_VERSION:
        raise ValueError("private manifest envelope version is invalid")
    payload = document.get("manifest")
    if not isinstance(payload, dict):
        raise ValueError("private manifest payload is invalid")
    digest = _hash_payload(payload)
    if document.get("manifest_sha256") != digest or expected.manifest_sha256 != digest:
        raise ValueError("private manifest integrity check failed")
    if payload.get("schema_version") != expected.schema_version:
        raise ValueError("private manifest schema binding is stale")
    if payload.get("dataset_namespace") != expected.dataset_namespace:
        raise ValueError("private manifest dataset provenance binding is stale")
    if payload.get("source_hashes") != dict(expected.source_hashes):
        raise ValueError("private manifest source provenance binding is stale")
    manifest = _manifest_from_payload(payload)
    if manifest.manifest_sha256 != expected.manifest_sha256:
        raise ValueError("private manifest role binding is stale")
    return manifest


T = TypeVar("T")


def run_with_role_access(
    manifest: RoleManifest,
    *,
    operation: Operation,
    roles: Iterable[Role],
    loader: Callable[[tuple[PrivateExamRecord, ...]], T],
) -> T:
    """Authorize requested roles before invoking a loader or callback.

    There is deliberately no locked-test release flag.  A verified progression
    record and release path belong to the later evaluation phase.
    """

    try:
        normalized_operation = Operation(operation)
    except (TypeError, ValueError) as exc:
        raise PermissionError("unknown research operation") from exc
    requested: list[Role] = []
    for raw_role in roles:
        try:
            role = Role(raw_role)
        except (TypeError, ValueError) as exc:
            raise PermissionError("unknown research role") from exc
        if role in requested:
            raise PermissionError("requested roles must be unique")
        requested.append(role)
    if not requested:
        raise PermissionError("at least one role must be requested")
    if Role.LOCKED_TEST in requested:
        raise PermissionError("locked_test is inaccessible before verified progression release")
    allowed = OPERATION_ROLE_ALLOWLIST[normalized_operation]
    if not set(requested).issubset(allowed):
        raise PermissionError("requested role is not permitted for this operation")
    selected = tuple(record for record in manifest.records if record.role in requested)
    return loader(selected)
