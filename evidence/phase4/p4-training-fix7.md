Implemented the focused optimizer-freeze fix.

- [training.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/src/mmdc_clip_f/research/view_risk/training.py:243): `ResearchRunConfig` now requires exact `OptimizerConfig()` protocol defaults: Adam, learning rate `1e-4`, weight decay `0`, batch size `6`.
- [test_view_risk_training.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/tests/research/test_view_risk_training.py:138): added public loading regressions for altered learning rate, weight decay, and batch size. Default round-trip remains accepted.
- Low-level `OptimizerConfig` and synthetic fitting helpers remain flexible.

Validation:

- RED: new regression produced `3 failed`; every altered optimizer field was previously accepted.
- GREEN targeted: `5 passed`, including default configuration and non-default low-level synthetic resume.
- Full research suite: `261 passed`.
- Ruff: passed.
- Compileall: passed.
- `git diff --check`: passed.
- One expected CUDA/NVML warning; no test failures.
- Review7 evidence remains preserved and untracked.
- No dependencies, commits, real-data access, downloads, P4C, G4, or P5 work.

This candidate is ready for fresh independent review; no acceptance claim is made.