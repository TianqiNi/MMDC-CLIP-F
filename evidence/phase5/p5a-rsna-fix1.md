Implemented both requested fixes at unchanged HEAD `69f9ee8984449d4c59e780cd31115c74579ff2c1`; changes remain uncommitted.

Changes:

- [rsna_audit.py](/home/tianqini/research/MMDC-CLIP-IVR/src/mmdc_clip_f/research/view_risk/rsna_audit.py:136)
  - Recovers across all six finalization boundaries.
  - Reuses only verified role manifests, bindings, readiness evidence, private run manifests, public reports, and progress.
  - Rehashes production bytes through the accepted readiness loader.
  - Rejects tampered, malformed, out-of-order, or blocked/ready-conflicting artifacts.
  - Ignores only validated runtime/resume telemetry when comparing resumed artifacts.
- [cli.py](/home/tianqini/research/MMDC-CLIP-IVR/src/mmdc_clip_f/cli.py:109)
  - Blocked RSNA audits still emit their sanitized report but return exit code `2`.
  - Ready audits and other research commands retain exit code `0`.
- [test_view_risk_rsna_audit.py](/home/tianqini/research/MMDC-CLIP-IVR/tests/research/test_view_risk_rsna_audit.py:531)
  - Covers interruption after every finalization write, tamper refusal, and blocked CLI behavior.

Validation:

- Preserved RED: `10 failed, 13 deselected`.
- Final GREEN selector: `10 passed, 13 deselected`.
- Complete adapter: `23 passed`.
- Full research suite: `292 passed, 1 benign NVML warning`.
- Ruff: passed.
- Compile-all: passed.
- Audit CLI help: passed.
- `git diff --check`: passed.

The initially attempted `python -m mmdc_clip_f ...` help invocation failed because the package has no `__main__`; the declared entrypoint `python -m mmdc_clip_f.cli view-risk-audit-rsna --help` passed.

No full DICOM scan, training, configuration change, dependency, commit, push, or real artifact mutation occurred. Real RSNA readiness remains pending: only the prior eight-image bounded journal evidence exists versus 19,748 expected images. Weight/compute readiness, MINI-DDSM, cross-dataset cleanliness, and scientific performance remain unaudited. Root can now run the long exact-bound audit separately with durable resume support.
Scope clarification: resource readiness was separately reviewed and published at `2695f0424e386535b33343df9522cf68262151ed`; this implementation task did not reassess it. This report records the pre-commit implementer result; independent review is required before a real full scan.
