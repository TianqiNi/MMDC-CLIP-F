Implemented all three review4 fixes on top of `2711fbf0a73365633090eedeaa6133004c140071`. No commit or push was made.

## Finding disposition

1. Learned-selection cache verification — fixed

- Learned-selection evidence now records every referenced tune-cache metadata and tensor path/digest.
- Reload reparses the tune index and rehashes each current cache pair, in addition to the existing fit artifact, checkpoints, budget, AURC, reference, and guardrail replay.
- Missing, modified, truncated, duplicated, or stale cache files reject selection reload and therefore pilot-plan reload.
- Main interfaces: [cache.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/src/mmdc_clip_f/research/view_risk/cache.py:824), [evaluation.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/src/mmdc_clip_f/research/view_risk/evaluation.py:1017).

2. Authoritative evaluation provenance — fixed

- `view-risk-freeze-classifier-predictions` now accepts only `--cache-index`; free-standing prediction JSON is gone.
- Actual cache metadata is role-authorized and validated before any target-bearing tensor is hashed or opened.
- Validation binds the pilot manifest, complete classifier configuration, checkpoint, prompts, preprocessing, dataset fusion tree, observed mask, panel, family, severity, target/common-mode behavior, variant, deterministic seed, realized parameters, parent identity, and realization identity.
- Every represented cell must contain exactly the authorized cohort before tensor access.
- Scoring repeats these validations and requires predictions to match the plan’s single immutable authoritative registry.
- Main interfaces: [cache.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/src/mmdc_clip_f/research/view_risk/cache.py:994), [evaluation.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/src/mmdc_clip_f/research/view_risk/evaluation.py:1944), [cli.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/src/mmdc_clip_f/research/view_risk/cli.py:698).

3. Analytic-control fitting provenance — fixed

- The public command no longer accepts standalone tune/confidence feature JSON.
- TS derives classifier scores from verified tune caches and fits tune NLL.
- DS derives accepted P3 vacuity/conflict features and classifier predictions from verified confidence-fit and tune caches; scaling/weights use confidence-fit and regularization selection uses tune.
- The five raw controls remain zero-fit rankings. Optional scalar calibration is explicitly separate, tune-fitted, and probability-labeled.
- Control artifacts bind and rehash indexes, manifests, metadata, and tensor bytes. A foreign-classifier index is rejected.
- Residual low-level production standalone-path parameters were removed; synthetic injected row readers remain available but cannot qualify a production plan without the required cache evidence.
- Main interfaces: [artifacts.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/src/mmdc_clip_f/research/view_risk/artifacts.py:408), [cli.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/src/mmdc_clip_f/research/view_risk/cli.py:460).

## RED/GREEN evidence

The immutable candidate’s RED baseline was the independent review’s three concrete probes:

- Corrupted `tune-cache-0.safetensors` still allowed selection reload.
- A clean cache relabeled as `primary/contrast/mild/L_CC` was accepted.
- Arbitrary standalone DS JSON produced `status: control_fitted`.

An additional outcome-ordering regression was run locally:

```text
pytest ...::test_one_plan_rejects_two_conflicting_authoritative_registrations
1 failed
cache tensor file integrity check failed
```

This demonstrated that tensor bytes were accessed before rejecting relabeled metadata. After metadata-first validation:

```text
1 passed in 0.75s
```

Final validation:

```text
Focused cache/evaluation/artifact/production suite:
68 passed in 11.11s

Full research suite:
258 passed in 10.82s

Ruff:
All checks passed!

compileall:
exit 0

git diff --check:
exit 0
```

CLI help checks passed for:

- `view-risk-select-confidence`
- `view-risk-fit-control`
- `view-risk-freeze-classifier-predictions`
- `view-risk-score-confidence`

The connected tiny-model test at [test_view_risk_production.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/tests/research/test_view_risk_production.py:444) mocks only the external Transformers loading boundary and exercises image loading, classifier fit/tune, cache generation, confidence fitting, inference-derived checkpoint selection, TS/DS/raw controls, artifact reload in a fresh process, and confidence scoring. The evaluation regression at [test_view_risk_evaluation.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/tests/research/test_view_risk_evaluation.py:721) exercises cache-derived authority, exact cohorts, conflicting registry refusal, relabeling, masks, wrong seeds/variants, and cache-byte reload checks.

## Public command flow

Representative flow:

```bash
mmdc-clip-f view-risk-select-confidence \
  --config run.json \
  --training-artifact confidence-fit.json \
  --manifest tune-manifest.json \
  --manifest-binding tune-binding.json \
  --private-root /private \
  --tune-cache-index tune-cache-index.json \
  --output correctness-selection.json
```

```bash
mmdc-clip-f view-risk-fit-control \
  --config run.json \
  --classifier-artifact selected-classifier.json \
  --method ds_logistic --seed 42 \
  --confidence-manifest confidence-manifest.json \
  --confidence-manifest-binding confidence-binding.json \
  --confidence-private-root /private \
  --confidence-cache-index confidence-control-index.json \
  --tune-manifest tune-manifest.json \
  --tune-manifest-binding tune-binding.json \
  --tune-private-root /private \
  --tune-cache-index tune-control-index.json \
  --output ds-control.json
```

After freezing the plan:

```bash
mmdc-clip-f view-risk-freeze-classifier-predictions \
  --config run.json --plan pilot-plan.json \
  --manifest pilot-manifest.json \
  --manifest-binding pilot-binding.json \
  --private-root /private \
  --cache-index evaluation-cache-index.json \
  --output <plan-owned-authoritative-path>
```

All method/seed scoring then uses that registry:

```bash
mmdc-clip-f view-risk-score-confidence \
  --config run.json --plan pilot-plan.json \
  --manifest pilot-manifest.json \
  --manifest-binding pilot-binding.json \
  --private-root /private \
  --cache-index evaluation-cache-index.json \
  --panel primary --method ds_logistic --seed 42 \
  --output /private/ds-primary-predictions.json
```

## Modified files

- [artifacts.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/src/mmdc_clip_f/research/view_risk/artifacts.py)
- [cache.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/src/mmdc_clip_f/research/view_risk/cache.py)
- [cli.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/src/mmdc_clip_f/research/view_risk/cli.py)
- [evaluation.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/src/mmdc_clip_f/research/view_risk/evaluation.py)
- [test_view_risk_evaluation.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/tests/research/test_view_risk_evaluation.py)
- [test_view_risk_production.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/tests/research/test_view_risk_production.py)

The copied review4 evidence remains untracked and untouched. No dependency changes were made.

## Remaining limitations

No real medical data, public weights, full training, pilot outcomes, or locked-test data were accessed. The tests establish software behavior only, not patient readiness, pretrained parity, runtime cost, or empirical improvement. P4C still owns comprehensive smoke/cost/CI work; P5 still owns real resources and execution. This candidate still requires fresh independent review before any P4B acceptance claim.