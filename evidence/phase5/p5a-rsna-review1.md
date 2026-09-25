## REQUEST CHANGES — `69f9ee8984449d4c59e780cd31115c74579ff2c1`

1. **High: interruption after image scanning can permanently strand the completed journal.**  
   [rsna_audit.py:1343](/home/tianqini/research/MMDC-CLIP-IVR/src/mmdc_clip_f/research/view_risk/rsna_audit.py:1343) refuses resume when any final readiness artifact already exists, although those artifacts are written sequentially starting at line 1350.

   Reproduction: interrupt `save_readiness_audit` after the role manifest and binding are written. The run is left `(manifest=True, binding=True, readiness=False)`; `--resume` then raises `FileExistsError: refusing to clobber incomplete final readiness artifacts`, with no public report. This can waste the completed 19,748-image scan.

   Fix by making finalization recoverable: verify and reuse consistent existing artifacts, finish missing ones, or use adapter-owned staging plus safe cleanup. Add interruption tests after each final artifact boundary.

2. **Medium: a blocked audit exits successfully.**  
   [cli.py:106](/home/tianqini/research/MMDC-CLIP-IVR/src/mmdc_clip_f/cli.py:106) prints every research result and returns `0` at line 109. An actual CLI probe produced `status="blocked"` with a count-mismatch blocker and no role/readiness artifacts, but exit code `0`.

   This can make automation treat duplicate, overlap, invalid, or incomplete data as accepted. Return a nonzero audit exit code when status is not `ready`, while retaining the sanitized report.

## Verified

- Real non-test manifests match the prior inventory hashes and official schema: 4,647 train + 290 validation patients, 19,748 named views, expected density aggregates, no missing fields or duplicate patient/view identities.
- The private bounded probe is mode `0600`, header-hash consistent, contains eight contiguous passed image records, expects 19,748 images, and has no readiness/role artifacts.
- Identity-only locked-test handling is structurally enforced; no locked-test outcomes or images were used or displayed.
- Mid-scan resume, completed-byte rehash, changed-byte refusal, explicit re-audit, corrupt-journal refusal, partial-append recovery, duplicate/missing refusal, and no premature promotion passed focused tests.
- Focused: `13 passed`.
- Full research suite: `282 passed`, one benign NVML warning.
- Ruff, compile, CLI help, and `git diff --check`: passed.
- Worktree remains clean at the exact reviewed commit.

Real RSNA data readiness remains **pending**: only eight real images have journal evidence, not the full 19,748. Weight/compute readiness, MINI-DDSM, cross-dataset cleanliness, and scientific performance remain unaudited. No fitting is permitted, and the classifier schedule decision remains pending.
