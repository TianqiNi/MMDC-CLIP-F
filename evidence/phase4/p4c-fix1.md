Implemented the narrow P4C coverage fix. The candidate remains uncommitted at HEAD `0eea2089647b71b590ffd8d970bff4b86dfd4b04`, ready for fresh review; no acceptance is claimed.

### Exact fix

- Both RSNA and DDSM fusion configurations now evaluate 36 checked inputs each:

  - 2 clean four-view samples
  - 28 clean-mask samples: 14 proper masks × 2 records, including singletons
  - 2 permitted-stress samples
  - 2 held-out-stress samples
  - 2 common-mode-stress samples

- Candidate, `same_input_mlp`, MSP, and DS score all 72 checked inputs.
- All 40 method/tree/panel combinations report actual counts and P4A confidence metrics.
- Candidate additionally reports P4A effect metrics for all 10 tree/panel combinations.
- All four methods run the 14-cell clean-mask AURC evaluation for both trees.
- MSP remains a ranking score with `brier: null`; candidate, same-input MLP, and DS remain probability outputs with Brier metrics.
- Each matrix entry reports `sample_count`, `checked_target_count`, `prediction_consistent_count`, metric outputs, and a digest of the actual scored input/results.
- Training scope is explicitly separated from scoring scope:

| Method | Synthetic training scope | Scoring scope |
|---|---|---|
| Candidate | Partial one-epoch RSNA-tree fit/reload | Both trees, all five panels |
| Same-input MLP | Partial one-epoch RSNA-tree fit/reload | Both trees, all five panels |
| MSP | No fit; raw scalar artifact reload | Both trees, all five panels |
| DS logistic | Role-authorized RSNA synthetic confidence-fit/tune artifact | Both trees, all five panels |

DDSM here means the configured DDSM fusion tree over synthetic inputs—not a DDSM patient dataset or cross-dataset efficacy experiment.

Implementation: [smoke.py](/home/tianqini/research/.ivr-p4c-worktrees/p4c-smoke/src/mmdc_clip_f/research/view_risk/smoke.py:935)  
Behavioral regression: [test_view_risk_smoke.py](/home/tianqini/research/.ivr-p4c-worktrees/p4c-smoke/tests/research/test_view_risk_smoke.py:119)

### RED/GREEN evidence

RED:

```text
PYTHONPATH=src .../.venv/bin/python -m pytest \
  tests/research/test_view_risk_smoke.py::test_smoke_scores_each_tree_method_and_panel_through_accepted_metrics \
  -q --basetemp=.cache/pytest-p4c-red

FAILED: KeyError: 'coverage_matrix'
1 failed
```

Final GREEN:

```text
Behavioral regression: 1 passed
Targeted P4C smoke tests: 8 passed
Full research suite: 269 passed
Ruff src tests/research: passed
compileall src tests/research: passed
CLI help: passed
git diff --check: passed
```

The behavioral test observes the accepted target/scoring/metric calls, including 36 checked inputs per tree, DDSM scoring by both learned and analytic controls, and exact correspondence between 40 reported confidence results and actual P4A metric outputs.

### Retained synthetic evidence

Updated report: [p4c-synthetic-smoke-report.json](/home/tianqini/research/.ivr-p4c-worktrees/p4c-smoke/evidence/phase4/p4c-synthetic-smoke-report.json)

- File SHA-256: `34d2dec34aa243e03c85f99e4c370e77dbaf77f3ee511eafeaf93fa0bbd38d4a`
- Timing-independent scientific digest: `705a196af57bee4d61afa40643e293cc4146a9fbfbfccb857c1ffca187f4700b`
- Status: `software_smoke_passed`
- Mode: `synthetic_software_only`
- Signed effect support: `-1: 32`, `0: 96`, `+1: 32`

Command:

```bash
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python \
  -m mmdc_clip_f.cli view-risk-smoke \
  --output-dir .cache/p4c-fix1-final-smoke \
  --device cpu --warmup 2 --repeats 5
```

Measured tiny-CPU medians:

- Encoder/feature validation: `3.268260 ms`
- Cache reload: `2.459633 ms`
- Candidate head: `4.072977 ms`
- MSP control: `1.329958 ms`
- DS control: `1.309518 ms`
- Synthetic end-to-end: `14.361706 ms`

Exact measured parameter counts remain 433,436 candidate and 433,651 same-input parameters; generated cache size was 34,406 bytes. CPU memory remains explicitly labeled process-lifetime high-water RSS.

CUDA was unavailable. Private data, patient mapping, and real patient readiness remain unavailable/not audited. No real backbone costs, empirical benefit, pilot decision, or efficiency improvement were measured.

Changed files:

- `src/mmdc_clip_f/research/view_risk/smoke.py`
- `tests/research/test_view_risk_smoke.py`
- `evidence/phase4/p4c-synthetic-smoke-report.json`

No dependency, CI, protocol, or production-configuration changes. The copied review, validation, and interruption evidence remains preserved and unmodified.