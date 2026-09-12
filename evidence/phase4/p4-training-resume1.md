Implemented P4B in the isolated worktree. No commits, pushes, dependency changes, private-data access, downloads, or real training were performed.

### Main changes

- [training.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/src/mmdc_clip_f/research/view_risk/training.py:214)
  - Strict frozen JSON/YAML configuration and search budgets.
  - Public pinned CLIP provenance and injected tiny-model initialization.
  - Synthetic versus real-data readiness audits.
  - Role-authorized fresh classifier fitting and tune-selected provenance.
  - Deterministic matched schedules, current-target regeneration, all learned controls/ablations, bounded mixed-mask cache streaming, and Adam optimization.
  - Verified frozen-encoder binding and unchanged-state checks.
  - Resume checkpoints binding model, optimizer, RNG, schedule, epoch, method, manifests, protocol, classifier, and config.
  - External/git-ignored final and staging path enforcement.

- [evaluation.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/src/mmdc_clip_f/research/view_risk/evaluation.py:198)
  - Tune-role/manifest-bound checkpoint results.
  - Correctness-MVACN reference selection.
  - Balanced 16-cell noise/blur selection with the 0.005 clean guardrail and deterministic ties.
  - Explicit no-eligible-selection result.
  - Immutable, exclusively published pilot plans requiring all 23 methods × 3 seeds.
  - Revalidation of the on-disk plan before every pilot outcome read.
  - Locked-test refusal and aggregate-only P4A evaluation.

- [view-risk CLI](/home/tianqini/research/.ivr-p4-worktrees/p4-training/src/mmdc_clip_f/research/view_risk/cli.py:58), registered in [cli.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/src/mmdc_clip_f/cli.py:51):
  - `view-risk-validate-config`
  - `view-risk-select`
  - `view-risk-freeze-pilot`
  - `view-risk-evaluate`
  - `view-risk-train-confidence`

Example:

```bash
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python \
  -m mmdc_clip_f.cli view-risk-train-confidence \
  --config /external/run.json \
  --manifest /external/confidence-fit.json \
  --manifest-binding /external/confidence-fit-binding.json \
  --private-root /external/images \
  --cache-index /external/cache-index.json \
  --method candidate --seed 42 \
  --checkpoint /external/checkpoints/candidate-42.pt
```

There is no research `--allow-test-selection` or test-unlock option.

### Test evidence

RED:

- Initial focused run: 2 collection errors because the P4B training/evaluation modules did not exist.
- Strict-type follow-up: 1 failure showing `epochs: 20.0` was accepted; fixed to require integer-only frozen budgets.

GREEN:

```text
Focused P4B suite:
20 passed in 1.12s

Full research suite:
244 passed in 2.18s

Ruff:
All checks passed!

py_compile:
exit 0

Top-level CLI help:
exit 0; all five view-risk commands registered

git diff --check:
exit 0
```

Tests are in:

- [test_view_risk_training.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/tests/research/test_view_risk_training.py:125)
- [test_view_risk_evaluation.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/tests/research/test_view_risk_evaluation.py:102)
- [test_view_risk_cache.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/tests/research/test_view_risk_cache.py:194)

They exercise refusal-before-reader behavior, diagnostic/fresh identity separation, held-out stress rejection, regenerated targets, actual optimizer updates, singleton/mixed masks, frozen classifiers, resume equivalence/tampering, selection guardrails/ties, complete pilot freezing, on-disk plan tampering, locked outcomes, and real P4A aggregation.

### Remaining prerequisites

- Synthetic fixtures remain explicitly `patient_ready=False`; no patient independence, inventory count, or pretrained-corpus independence is claimed.
- Analytic controls continue through accepted P3 APIs: tune-only temperature NLL, confidence-fit DS logistic with tune-selected regularization, scalar rankings, and tune-only monotone calibration. The learned-method CLI intentionally handles only epoch-trained Torch methods.
- P4C still owns full end-to-end synthetic command smoke, costs/latency, and CI wiring.
- P5 must supply audited external manifests, real cache indexes, public checkpoint loading, actual method artifacts, all real method/seed runs, and the real pilot plan.
- No locked-test release or automatic gate was implemented.
- No claims are made about pretrained parity, runtime, pilot outcome, uncertainty improvement, or real-data readiness.