Reviewed candidate `b0508fdcbbb1db4326125002c8a448dcc199e8d4` against dependency base `c689eb9ddb147abb68452461a5e130ee3257a626`. The worktree remained clean.

## Finding

1. **High — Saving can create an unignored patient-bearing tensor artifact.**

   [cache.py:771](/home/tianqini/research/.ivr-worktrees/p2-cache-review2/src/mmdc_clip_f/research/view_risk/cache.py:771) checks only whether the requested JSON metadata path is external or Git-ignored. [cache.py:772](/home/tianqini/research/.ivr-worktrees/p2-cache-review2/src/mmdc_clip_f/research/view_risk/cache.py:772) then derives and writes a sibling `.safetensors` path without applying the same check.

   Reproduction in a temporary Git repository:

   ```text
   .gitignore: artifact.json
   save destination: artifact.json

   metadata_exists=True
   tensor_exists=True
   tensor_ignored=False
   git status:
   ?? .gitignore
   ?? artifact.safetensors
   ```

   The unignored tensor contains labels, predictions, targets, and per-exam features. This violates the explicit requirement that patient-bearing artifacts be persisted only externally or at ignored destinations.

   Fix: resolve both final paths and validate both with `_require_external_or_ignored_destination` before writing either file. Add a regression where only the JSON is ignored and assert refusal with neither artifact created.

## Checks run

- Focused feature/cache tests: `61 passed`.
- Full research suite: `159 passed`.
- Ruff: passed.
- `py_compile`: passed.
- `git diff --check`: passed.
- Independent exact legacy-parity probe:
  - RSNA tree: `torch.equal=True`, maximum absolute difference `0.0`.
  - DDSM tree: `torch.equal=True`, maximum absolute difference `0.0`.
- Confirmed all four previous-review corrections are present and their regression tests pass.
- Candidate and review completed before the effective cutoff.

These checks establish synthetic software behavior only. No pretrained checkpoint parity, real patient data, inventory, or real-data readiness was exercised or established.

**Verdict: REQUEST CHANGES — not accepted due to one high-severity actionable finding.**