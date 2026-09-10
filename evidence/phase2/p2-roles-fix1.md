P2A review fixes are complete in the four assigned files. No nested agents, commits, dependency changes, training, or patient outcome access occurred.

### Accepted findings fixed

1. Locked-test exposure

- `PrivateExamRecord(..., role=Role.LOCKED_TEST)` now raises `PermissionError`.
- `save_private_manifest()` defensively rejects locked records before writing.
- `load_private_manifest()` examines raw role fields and rejects locked payloads before constructing usable records or view objects.
- No boolean bypass or release mechanism was introduced; P4 retains release responsibility.
- Identity-only overlap checking is supported without outcomes or image paths.

2. Image-path collisions

- `validate_inventory()` now rejects reuse of the same image path across manifests/roles within the same dataset namespace, even with different image IDs and no hashes.
- Identical paths in different dataset namespaces remain independent.

3. RSNA evidence

- External custodian sign-off is no longer listed as required.
- The official RSNA schema supports `patient_id`.
- Evidence states that an authorized researcher may record schema-derived provenance, while correctly noting that the local binding has not been executed.

### New identity-only APIs

Defined in [roles.py](/home/tianqini/research/.ivr-worktrees/p2-roles/src/mmdc_clip_f/research/view_risk/roles.py:562):

```python
LockedPatientIdentityDenylist(
    dataset_namespace: str,
    source_sha256: str,
    patient_identity_digests: frozenset[str],
    schema_version: str = "view-risk-locked-patient-denylist/v1",
)

patient_identity_digest(
    dataset_namespace: str,
    patient_key: str,
) -> str

validate_locked_patient_isolation(
    manifest: RoleManifest,
    denylist: LockedPatientIdentityDenylist,
) -> LockedPatientIsolationSummary

validate_inventory(
    manifests: Iterable[RoleManifest],
    *,
    locked_patient_denylists: Iterable[LockedPatientIdentityDenylist] = (),
) -> InventorySummary
```

The denylist remains private, grants no test access, and contains only namespaced identity digests.

### RED/GREEN results

- RED: **3 failed, 15 passed**
  - Locked record construction was accepted.
  - A rehashed locked payload loaded successfully.
  - Same-namespace path reuse passed inventory validation.
- GREEN: **20 passed in 0.56s** scoped.
- Full research suite: **60 passed in 0.62s**.
- Ruff lint and format, JSON validation, and `git diff --check` passed.

### Remaining readiness blockers

- RSNA schema-derived local provenance binding has not been recorded.
- Locked-test identity overlap has not been audited for either dataset.
- Content-duplication auditing remains unexecuted.
- DDSM exam-to-patient mapping remains unverified.
- No patient split was generated.

Updated files:

- [roles.py](/home/tianqini/research/.ivr-worktrees/p2-roles/src/mmdc_clip_f/research/view_risk/roles.py)
- [test_view_risk_roles.py](/home/tianqini/research/.ivr-worktrees/p2-roles/tests/research/test_view_risk_roles.py)
- [p2-roles-tdd.md](/home/tianqini/research/.ivr-worktrees/p2-roles/evidence/p2-roles-tdd.md)
- [p2-inventory.json](/home/tianqini/research/.ivr-worktrees/p2-roles/evidence/p2-inventory.json)

Independent follow-up review remains with the orchestrator.
