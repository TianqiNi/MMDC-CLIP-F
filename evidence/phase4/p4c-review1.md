# P4C review — REQUEST CHANGES

Reviewed exact commit **`0eea2089647b71b590ffd8d970bff4b86dfd4b04`** against accepted base **`1a4a0f927f167d9f46cef1ee8dc0bf8073a4e418`**.

## Finding

1. **Medium — Reported coverage is not end-to-end across the required trees, controls, stresses, and metrics.**

   DDSM bundles are created for all masks at [smoke.py:754](/home/tianqini/research/.ivr-p4c-worktrees/p4c-review1/src/mmdc_clip_f/research/view_risk/smoke.py:754), but downstream head/target verification explicitly selects only RSNA bundles at [smoke.py:958](/home/tianqini/research/.ivr-p4c-worktrees/p4c-review1/src/mmdc_clip_f/research/view_risk/smoke.py:958). MSP and DS are evaluated only on one clean, full-view RSNA example at [smoke.py:1040](/home/tianqini/research/.ivr-p4c-worktrees/p4c-review1/src/mmdc_clip_f/research/view_risk/smoke.py:1040), while the sole confidence-metric call receives candidate scores only at [smoke.py:1062](/home/tianqini/research/.ivr-p4c-worktrees/p4c-review1/src/mmdc_clip_f/research/view_risk/smoke.py:1062).

   Nevertheless, the report independently lists both fusion trees, four exercised methods, and all representative panels at [smoke.py:1210](/home/tianqini/research/.ivr-p4c-worktrees/p4c-review1/src/mmdc_clip_f/research/view_risk/smoke.py:1210). The test merely checks those reported flags/lists at [test_view_risk_smoke.py:89](/home/tianqini/research/.ivr-p4c-worktrees/p4c-review1/tests/research/test_view_risk_smoke.py:89).

   Controlled instrumentation reproduced:

   ```text
   learned scoring fusion trees: RSNA only
   control calls: MSP and DS once for output, once for timing
   confidence metric calls: [(36, 36, "probability")]
   ```

   Impact: CI can pass while DDSM head integration, control scoring on permitted/held-out/common-mode stresses, or control-to-P4A metric integration is broken. This misses a core P4C acceptance condition.

   Minimal fix: pass DDSM mask bundles through checked targets, the accepted candidate head, and a P4A metric; feed each claimed representative control’s clean/stress scores into P4A metrics; report a method/tree/panel coverage matrix. Add a behavioral assertion over those outputs or observed calls. This need not become the full mandatory-control study.

## Independent validation

- Public CPU smoke: passed in 2.2s after creating the required ignored `.cache` parent. The initial invocation correctly failed because that parent did not exist.
- Timing-independent digest matched committed evidence: `3e9eebfb72cd8e8a9386b19506aea1b8cbfe49ee530381f3a9673df2b3bb194c`.
- Targeted P4C tests: `7 passed` in 2.77s.
- Full research suite: `268 passed` in 23.24s.
- Ruff: passed.
- Compile, CLI help, preflight, diff check, and workflow YAML parsing: passed.
- Only warning: unavailable NVML.
- Measured CPU medians, 2 warmups/5 calls: encoder 3.407 ms, cache reload 2.343 ms, candidate 3.171 ms, MSP 1.302 ms, DS 1.265 ms, synthetic end-to-end 13.360 ms.
- Parameter counts and cache bytes were accurate; RSS was correctly labeled process-lifetime high-water memory.
- No production lock weakening, test removal, test-module runtime import, pytest shell-out, network/download path, or real/private-data access found.

Real CLIP costs, accelerator memory, complete panels, mandatory controls/ablations, patient readiness, empirical gains, pilot decisions, P5, and G4 remain unmeasured. No approval was rejected or bypassed. No source edits, commits, or pushes were made; review caches were removed and the worktree remains clean.