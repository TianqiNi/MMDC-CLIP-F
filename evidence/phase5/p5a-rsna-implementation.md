Implemented and validated the resumable RSNA audit adapter. The full real-data audit remains pending and was not restarted.

Changes:

- Added [rsna_audit.py](../../src/mmdc_clip_f/research/view_risk/rsna_audit.py):
  - Identity-only locked-test projection.
  - Production DICOM byte and decoded-pixel auditing.
  - Append-only private per-image journal with `fsync`.
  - Exact source, mapping, identity, schema, configuration, semantics, and image-root bindings.
  - Resume-time re-hashing of every completed image.
  - Changed-image refusal unless `--re-audit-changed` is explicit.
  - Changed-source and corrupt-journal refusal.
  - Safe truncation only for an unterminated final journal append.
  - Partial progress cannot create role/readiness artifacts.
  - Sanitized aggregate public progress.
- Extended [cli.py](../../src/mmdc_clip_f/research/view_risk/cli.py) with `view-risk-audit-rsna`, `--resume`, and `--re-audit-changed`.
- Added 13 focused tests in [test_view_risk_rsna_audit.py](../../tests/research/test_view_risk_rsna_audit.py).

Validation:

- Preserved original RED: missing adapter module.
- New resumability RED: missing `AuditInterrupted`.
- Focused GREEN: `13 passed`.
- Full research suite: `282 passed`, one benign NVML warning.
- Ruff: passed.
- Compile: passed.
- CLI help and `git diff --check`: passed.
- No dependencies added, commits, pushes, training, model loading, or config changes.

Bounded real probe:

- First execution durably checkpointed 4 images.
- Resume re-hashed/reused those 4 and checkpointed 4 more.
- Aggregate result: 8 passed, 0 missing, 0 unreadable, 0 decode failures, 19,740 pending.
- Readiness promotion: false.
- Historical `p5a-rsna-20260923T1630Z` remains untouched with only its original denylist and mapping artifacts.
- Probe journal is external and mode `0600`.

### Full audit command

Use a fresh run name in the separate long-running shell:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python \
-m mmdc_clip_f.cli view-risk-audit-rsna \
--train-manifest /home/tianqini/research/MMDC-CLIP_original/MMDC-CLIP/RSNA/RSNA_train.csv \
--validation-manifest /home/tianqini/research/MMDC-CLIP_original/MMDC-CLIP/RSNA/RSNA_val.csv \
--locked-test-manifest /home/tianqini/research/MMDC-CLIP_original/MMDC-CLIP/RSNA/RSNA_test.csv \
--image-root /home/tianqini/research/MMDC-CLIP_original/MMDC-CLIP/RSNA/train_images \
--prior-inventory /home/tianqini/research/.ivr-p5-worktrees/p5a-audit/evidence/p2-inventory.json \
--private-root /home/tianqini/research/mmdc_ivr_private/p5a/rsna-audit \
--run-name p5a-rsna-full-20260924-v1 \
--public-report /home/tianqini/research/.ivr-p5-worktrees/p5a-audit/evidence/phase5/p5a-rsna-audit.json \
--image-workers 4
```

After interruption, run the exact command again with `--resume`. Do not add `--re-audit-changed` automatically; a changed-image refusal requires an explicit policy decision.

Sanitized progress will appear at:

`evidence/phase5/p5a-rsna-audit-progress.json`

Expected full runtime remains approximately 70–80 minutes plus the accepted readiness API’s final byte re-hash. Until it completes, RSNA data readiness remains pending/blocked; weight/compute readiness and scientific performance are outside this audit.
