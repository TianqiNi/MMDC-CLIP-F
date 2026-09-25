Reviewed commit `2b63cf50016a418d1cca256da3a9965d3a890556` against base `3541fc0d8749bda97c5acbe4ba632dbf2b37da3d`.

## Findings

1. **High — locked-test outcomes can be loaded before the access guard**

   [roles.py](/home/tianqini/research/.ivr-worktrees/p2-roles-review1/src/mmdc_clip_f/research/view_risk/roles.py:123) requires every record—including `locked_test`—to contain `density`, while [load_private_manifest()](/home/tianqini/research/.ivr-worktrees/p2-roles-review1/src/mmdc_clip_f/research/view_risk/roles.py:783) unconditionally deserializes every record. The allowlist in [run_with_role_access()](/home/tianqini/research/.ivr-worktrees/p2-roles-review1/src/mmdc_clip_f/research/view_risk/roles.py:821) is therefore an optional, later boundary.

   Public-API reproduction successfully saved and loaded a `locked_test` manifest and directly returned:

   ```text
   unguarded_loaded_role= locked_test
   unguarded_loaded_density= 3
   unguarded_loaded_path_present= True
   ```

   The existing test at [test_view_risk_roles.py:299](/home/tianqini/research/.ivr-worktrees/p2-roles-review1/tests/research/test_view_risk_roles.py:299) only proves that the final callback is not invoked; it has already materialized the locked density and paths.

   Fix by keeping locked-test data identity-only in P2—such as a patient denylist/digest without outcomes or image paths—and rejecting locked records from ordinary manifest loading until P4’s verified release. Test the actual manifest/outcome loader, not only a downstream callback.

2. **Medium — cross-manifest reuse of the same image path is accepted**

   [RoleManifest](/home/tianqini/research/.ivr-worktrees/p2-roles-review1/src/mmdc_clip_f/research/view_risk/roles.py:229) tracks paths within one manifest, but [validate_inventory()](/home/tianqini/research/.ivr-worktrees/p2-roles-review1/src/mmdc_clip_f/research/view_risk/roles.py:550) tracks only exam, patient, image-ID, and content-hash identities across manifests.

   Reproduction used two same-namespace manifests with matching provenance, different patients/image IDs, classifier-fit versus pilot roles, and the same `L_CC` path. Validation returned successfully:

   ```text
   cross_manifest_shared_path=ACCEPTED 2
   ```

   Because content hashes are optional and were not computed in the preflight, this permits ordinary cross-role image reuse. Add a namespaced path registry to `validate_inventory()` and a cross-manifest regression test.

3. **Medium — evidence invents an RSNA custodian-signoff prerequisite**

   [p2-inventory.json:45](/home/tianqini/research/.ivr-worktrees/p2-roles-review1/evidence/p2-inventory.json:45) and [p2-roles-tdd.md:60](/home/tianqini/research/.ivr-worktrees/p2-roles-review1/evidence/p2-roles-tdd.md:60) classify the missing RSNA “custodian verification declaration” as a real-split blocker. The implementation report repeats this as a required declaration.

   The accepted protocol requires an external exam-to-patient mapping specifically for DDSM; RSNA’s official manifest schema already identifies `patient_id`. The authorized researcher can record the schema-derived identity/provenance declaration. A synthetic probe confirmed the API accepts `verification_method="official_rsna_patient_id_schema"` and `verification_authority="authorized researcher"`.

   Revise the evidence/report. RSNA remains unready because locked-test identity overlap and content duplication were not audited, but external custodian approval is not required for its non-test patient mapping. DDSM’s mapping remains separately unverified.

## Checks run

- Exact commit/parent and clean worktree verified.
- Focused tests: `15 passed`.
- Full research tests: `55 passed`.
- Ruff: passed.
- Compile check and `git diff --check`: passed.
- 348 independent single-exam count/order cases: exact and row-order independent.
- Cross-namespace content collision: rejected.
- Self-consistently rehashed but reordered manifest: rejected.
- Locked-manifest and cross-manifest shared-path reproductions above.

No real metadata, patient outcomes, test manifests, images, or pixel content were opened. The aggregate count/file-presence report was treated as readiness evidence, not a completed real-data audit. Manifest hashes correctly provide integrity/staleness binding, not signed authentication.

**Verdict: REQUEST CHANGES.** Synthetic grouping, deterministic counts, provenance binding, canonical-order validation, and callback allowlists largely work, but the locked-outcome exposure is high severity and the two medium findings are actionable.
