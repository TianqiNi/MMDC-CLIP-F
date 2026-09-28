# REQUEST CHANGES — `dce28edb077608c3312adbe9f08d1f07221c7005`

## Finding

- **Medium — classifier-fit artifacts do not bind the recorded seed to the frozen schedule seed.** The save and reload validations in [production.py](/home/tianqini/research/.ivr-p5-worktrees/classifier-schedule-review1/src/mmdc_clip_f/research/view_risk/production.py:748) and [production.py](/home/tianqini/research/.ivr-p5-worktrees/classifier-schedule-review1/src/mmdc_clip_f/research/view_risk/production.py:883) only require `result.seed == binding.seed`; neither requires `binding.seed == schedule.seed`. The selected-artifact saver has the same omission at [production.py](/home/tianqini/research/.ivr-p5-worktrees/classifier-schedule-review1/src/mmdc_clip_f/research/view_risk/production.py:1039), although its loader eventually checks the seed.

  Reproduction: I changed both recorded seeds in a genuine synthetic 50-epoch fit artifact from 42 to 43, recomputed its canonical artifact hash, and reloaded it. It was accepted with:

  ```text
  {'accepted': True, 'recorded_seed': 43, 'frozen_seed': 42, 'completed_epoch': 50}
  ```

  Impact: a mismatched fit artifact is treated as valid and can enter tune selection despite contradicting the approved seed. Selection can consequently write/report success for an artifact that downstream selected-artifact reload later rejects.

  Required fix: enforce `binding.seed == schedule.seed` when saving and loading classifier-fit artifacts and when saving selected-classifier artifacts. Add a regression using a consistently self-rehashed `binding.seed`/`result.seed` mismatch.

## Validation

- Five focused schedule/production tests passed with tiny injected models.
- Full research suite: **293 passed**, one harmless unavailable-NVML warning.
- Ruff: passed.
- Compileall: passed.
- CLI help and checked-in config validation: passed.
- Explicit config v1, classifier-fit v1, and selected-classifier v2 probes were refused.
- Schedule/hash tampering tests passed.
- Diff check passed; worktree remains clean.

The classifier execution path itself correctly uses Adam `1e-7`, weight decay `1e-5`, batch 3, 50 epochs, seed 42, no AMP, and all 50 NLL candidates. Confidence settings remain Adam `1e-4`, weight decay 0, batch 6, RSNA 20/DDSM 50 epochs, seeds 42/43/44, with controls and search budgets unchanged.

**Approved first real fit may not run yet**; fix the artifact seed binding first. No real training, patient-data access, public-model loading, or scientific-performance evaluation occurred.