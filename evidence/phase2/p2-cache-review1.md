Reviewed `0883109f84171d88ace8debc34f14cd2f9f1f6a9` against base `c689eb9ddb147abb68452461a5e130ee3257a626`. The candidate’s parent is exactly that base, the worktree is clean, and only the five declared files changed.

## Findings

1. **High — Loaded labels are not checked against the manifest’s ordered densities.**  
   [cache.py:923](/home/tianqini/research/.ivr-worktrees/p2-cache-review1/src/mmdc_clip_f/research/view_risk/cache.py:923) validates exam and patient keys, then loads the artifact without comparing `targets.labels` to each corresponding `PrivateExamRecord.density`.

   Reproduction: changed label `2 → 3`, regenerated every label-dependent target from the stored logits, recomputed all tensor/file/record hashes, and loaded against the original manifest. Loading succeeded with `manifest_density=2` and `loaded_label=3`.

   Fix: inside the authorized callback, compare loaded labels exactly—including dtype, shape, value, and order—to densities resolved from `expected_provenance.exam_keys`.

2. **High — Direct `RealizedParent` metadata can disguise evaluation-only severity as fit-safe.**  
   [cache.py:533](/home/tianqini/research/.ivr-worktrees/p2-cache-review1/src/mmdc_clip_f/research/view_risk/cache.py:533) validates only `metadata.spec`; [cache.py:389](/home/tianqini/research/.ivr-worktrees/p2-cache-review1/src/mmdc_clip_f/research/view_risk/cache.py:389) then persists caller-provided realized parameters without checking that they match the spec.

   Reproduction: generated an actual strong Gaussian-noise image (`sigma=0.1`), wrapped it in a direct `RealizedParent` whose spec said `mild`, and passed it to confidence fitting. The bundle was accepted and recorded mild severity with strong parameters.

   Fix: validate realized parameters against `resolve_parameters(spec, resolution)` before encoding, including family-specific extra fields/bounds.

3. **Medium — Bound text inputs remain mutable while their provenance hash stays fixed.**  
   [features.py:435](/home/tianqini/research/.ivr-worktrees/p2-cache-review1/src/mmdc_clip_f/research/view_risk/features.py:435) uses `detach().to(device)`, which aliases the caller tensor when it is already on that device.

   Reproduction: mutating the original `input_ids` after factory construction changed extracted logits, while `text_input_sha256` remained unchanged. The probe reported `shared_storage=True`, `digest_unchanged=True`, and `logits_changed=True`.

   Fix: retain an independent private clone and verify its content binding before extraction.

4. **Medium — The unchanged-weight guard relies only on PyTorch version counters.**  
   [features.py:238](/home/tianqini/research/.ivr-worktrees/p2-cache-review1/src/mmdc_clip_f/research/view_risk/features.py:238) records `_version`, which is not a content identity. A direct `.data` mutation left the counter unchanged, changed logits, and was accepted under the original checkpoint hash.

   Fix: verify state tensor contents against a checkpoint-derived snapshot/content manifest, rather than relying solely on `_version`.

## Checks run

- Focused tests: `52 passed`.
- Full research regression: `150 passed`.
- Ruff: passed.
- `py_compile`: passed.
- `git diff --check`: passed.
- Exact synthetic legacy parity probe: `torch.equal=True`, maximum absolute difference `0.0`.
- Target-label, severity-metadata, input-binding, and weight-mutation probes as described above.

The successful checks establish substantial synthetic behavior, including observed-only encoding, omission reuse, exact legacy parity, role rejection, and ordinary tamper detection. No pretrained checkpoint, real patient data, inventory, or real-data readiness was exercised.

**Verdict: REQUEST CHANGES — not accepted.**