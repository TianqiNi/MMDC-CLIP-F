Implemented P4A only, without commits, pushes, dependency changes, legacy metric edits, or private-data access.

Files changed:

- [metrics.py](/home/tianqini/research/.ivr-p4-worktrees/p4-metrics/src/mmdc_clip_f/research/view_risk/metrics.py:1)
- [test_view_risk_metrics.py](/home/tianqini/research/.ivr-p4-worktrees/p4-metrics/tests/research/test_view_risk_metrics.py:1)

Key interfaces:

- `confidence_panel_metrics`: expected-tie AURC, 80/90% risk, grouped AP in both orientations, explicit probability-only Brier, accuracy/prevalence/counts.
- `effect_metrics`: constrained three-way effect metrics, validity masking, absent-class availability, clean/stressed strata, singleton counts, and majority-zero reference.
- `EvaluationPrediction` plus `evaluate_aurc_panel` / `evaluate_primary_endpoint`: strict enumerator-driven completeness and nested variant/cell averaging.
- `patient_paired_bootstrap_aurc_difference`: default 2,000 draws/seed 2026, aligned patient-cluster resampling across methods/seeds/variants, percentile intervals, undefined counts, and separate seed spread.

Verification evidence:

- RED: focused pytest failed collection with `ModuleNotFoundError` before `metrics.py` existed.
- Final focused GREEN: `12 passed in 0.62s`.
- Integration: `223 passed in 2.08s`.
- Ruff and `git diff --check`: passed.
- Synthetic 2,000-draw bootstrap probe: completed deterministically with zero undefined replicates.
- Weighted bootstrap AURC matched literal duplicated-exam expansion.

Limitations: this is generic software validated on synthetic fixtures. No data readiness, model training, checkpoint evaluation, measured improvement, pilot outcome, P4B, or P5 work is claimed. The bootstrap interface is intentionally limited to protocol-panel AURC differences.