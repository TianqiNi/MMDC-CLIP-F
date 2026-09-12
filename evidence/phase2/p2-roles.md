Implemented P2A using a fresh GPT-5.6-Sol xhigh agent with no nested subagents. Changes remain uncommitted and limited to the four authorized files.

### Acceptance mapping

- Fixed roles: `classifier_fit`, `confidence_fit`, `tune`, `pilot`, `locked_test`.
- Validated private exam, view, patient-mapping, manifest, assignment-summary, and inventory-summary dataclasses.
- Deterministic seed-42, input-order-independent, patient-grouped density stratification.
- Majority-density stratification with highest-density tie-break.
- Exact targets for feasible single-exam groups; requested versus achieved counts and grouping constraints reported otherwise.
- Dataset namespace isolation and provenance binding.
- Refusals for invalid views/labels/mappings, duplicate exams, patient overlap, cross-role image/content collisions, and inconsistent provenance.
- Operation-specific role access checked before loaders execute.
- `locked_test` remains inaccessible; no generic boolean bypass exists.
- Canonical private-manifest serialization with schema, source, role, order, and SHA-256 verification.

### Callable interfaces

Defined in [roles.py](/home/tianqini/research/.ivr-worktrees/p2-roles/src/mmdc_clip_f/research/view_risk/roles.py:25):

```python
assign_patient_roles(
    records,
    *,
    patient_mapping,
    source_hashes,
    counts=None,
    proportions=None,
    seed=42,
) -> RoleAssignment

validate_inventory(manifests) -> InventorySummary

save_private_manifest(
    manifest,
    path,
    *,
    private_root,
) -> ManifestBinding

load_private_manifest(
    path,
    *,
    expected,
    private_root,
) -> RoleManifest

run_with_role_access(
    manifest,
    *,
    operation,
    roles,
    loader,
)
```

Primary records are `ViewReference`, `PrivateExamRecord`, `PatientMappingDeclaration`, `RoleManifest`, and `ManifestBinding`.

### RED/GREEN evidence

- Initial RED: collection failed because `roles.py` did not exist.
- Additional RED: conflicting same-namespace source provenance was initially accepted.
- Final scoped suite: **15 passed**.
- Full research suite: **55 passed in 0.62s**.
- Synthetic exact-target property check: **76 fixture sizes passed**.
- Ruff lint and format checks passed.
- JSON parsing and `git diff --check` passed.

Evidence: [p2-roles-tdd.md](/home/tianqini/research/.ivr-worktrees/p2-roles/evidence/p2-roles-tdd.md)

### Aggregate preflight

- RSNA non-test exams: **4,937**, matching the protocol total.
- DDSM non-test exams: **1,353**, matching the protocol total.
- All expected non-test view paths were present.
- No test manifests, outcomes, or images were inspected.
- No image was decoded and no content-hash audit was run.
- No real patient split was generated.

Remaining real-study blockers:

- DDSM lacks a verified exam-to-patient mapping.
- Both datasets lack an identity-only locked-test denylist/digest.
- RSNA lacks the required custodian mapping declaration.
- Content-level duplication remains unaudited.
- Locked-test release remains intentionally deferred to P4.
- Independent review remains for the orchestrator.

Aggregate evidence: [p2-inventory.json](/home/tianqini/research/.ivr-worktrees/p2-roles/evidence/p2-inventory.json)

Changed files:

- [roles.py](/home/tianqini/research/.ivr-worktrees/p2-roles/src/mmdc_clip_f/research/view_risk/roles.py)
- [test_view_risk_roles.py](/home/tianqini/research/.ivr-worktrees/p2-roles/tests/research/test_view_risk_roles.py)
- [p2-roles-tdd.md](/home/tianqini/research/.ivr-worktrees/p2-roles/evidence/p2-roles-tdd.md)
- [p2-inventory.json](/home/tianqini/research/.ivr-worktrees/p2-roles/evidence/p2-inventory.json)
