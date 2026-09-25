Implemented a complete P4B candidate in the preserved worktree. It is ready for fresh independent review; I am not claiming acceptance.

## Review findings

1. Production provenance reload — fixed

- `_restore_frozen_record()` refuses public production provenance.
- Public initialization persists and rehashes pinned model state, tokenizer IDs, prompts, preprocessing, revision, and dataset-specific fusion order.
- Selected-classifier reload reconstructs provenance from initialization, readiness, classifier-fit, tune-manifest, checkpoint bytes, and full tune history.
- Reload now derives the frozen budget from the persisted fit configuration even when the caller supplies no expected config.
- Real readiness requires and rehashes source, patient-mapping, denylist, and referenced image bytes.
- Mocked public loading remains `not_verified` and cannot freeze a real pilot plan.

2. One authoritative prediction identity — fixed

- Each immutable plan binds one absolute authoritative-prediction registry path.
- Registration is exclusive and occurs only after plan freeze.
- Copying the plan cannot create another registry path.
- All methods/seeds reload that same artifact and must match its classifier decisions and exact authorized cohort in every represented cell.
- Real outcomes can only enter through the authoritative P4A prediction-panel workflow.
- Locked-test access remains unavailable; there is no CLI unlock flag.

3. Verified selections and controls — fixed

- Learned selection records persist and replay completed training, all epoch checkpoints, tune evidence, reference selection, guardrail, budget, config, classifier, and manifests.
- Reload works from a fresh Python process.
- Missing, altered, incomplete, or no-eligible records cannot enter a plan.
- Temperature and DS artifacts now bind and rehash their actual role-manifest and fitting-input files. Raw controls retain correct zero-fit evidence.
- The optional monotone scalar calibration has the same tune-evidence binding.
- Control artifact schema is now `view-risk-control-artifact/v2`; stale v1 artifacts are rejected.

4. Production workflow connections — fixed generically

- Role-bound image batching and public CLIP classifier fit/resume/checkpointing.
- Authorized tune inference and NLL-based classifier selection.
- Streamed learned confidence fitting and actual per-epoch tune inference/AURC selection.
- Dispatch for all seven analytic controls.
- Learned/control reload into label-free confidence scoring.
- Verified selections feed plan freeze, authoritative prediction registration, confidence-row generation, and P4A evaluation.

## RED/GREEN evidence

Meaningful failures reproduced before fixes:

- Missing stress-table binding: `1 failed`; afterward immutable plan/authority regressions: `2 passed`.
- Self-rehashed partial classifier tune history: `DID NOT RAISE`; afterward: `1 passed`.
- Changed temperature fitting input remained accepted: `DID NOT RAISE`; afterward connected workflow test: `1 passed`.
- Conflicting complete authoritative files initially accepted; afterward the public CLI path rejects the second registration.
- Earlier connected workflow RED states included missing classifier-fit and tune-operation dispatch; the completed integration now exercises classifier fit/select, seven controls, learned fit/select, subprocess reload, and confidence scoring.

Final validation:

```text
Focused P4B set:                         81 passed
Final affected evaluation/artifact/prod: 19 passed
Full research suite:                    258 passed in 11.39s
Ruff src + tests/research:              All checks passed
compileall src + tests/research:         exit 0
All 11 P4B CLI help/dispatch checks:     exit 0
git diff --check:                        passed
```

## Public workflow

Representative command flow:

```bash
mmdc-clip-f view-risk-fit-classifier \
  --config run.json --manifest classifier-fit.json \
  --manifest-binding classifier-fit-binding.json --private-root PRIVATE \
  --image-root IMAGES --readiness readiness.json \
  --initialization public-init.json --checkpoint-directory CHECKPOINTS \
  --resume-checkpoint classifier-resume.pt --output classifier-fit-artifact.json

mmdc-clip-f view-risk-select-classifier \
  --config run.json --fit-artifact classifier-fit-artifact.json \
  --manifest tune.json --manifest-binding tune-binding.json \
  --private-root PRIVATE --image-root IMAGES \
  --output selected-classifier.json

mmdc-clip-f view-risk-train-confidence ...
mmdc-clip-f view-risk-select-confidence ...
mmdc-clip-f view-risk-fit-control ...

mmdc-clip-f view-risk-freeze-pilot \
  --config run.json --bindings bindings.json \
  --selections selection-record-paths.json --output pilot-plan.json

mmdc-clip-f view-risk-freeze-classifier-predictions ...
mmdc-clip-f view-risk-score-confidence ...
mmdc-clip-f view-risk-evaluate ...
```

The older aggregate `view-risk-select` helper remains available, but it is no longer the only selection path.

## Changed files

- [training.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/src/mmdc_clip_f/research/view_risk/training.py)
- [production.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/src/mmdc_clip_f/research/view_risk/production.py)
- [inference.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/src/mmdc_clip_f/research/view_risk/inference.py)
- [evaluation.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/src/mmdc_clip_f/research/view_risk/evaluation.py)
- [artifacts.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/src/mmdc_clip_f/research/view_risk/artifacts.py)
- [cli.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/src/mmdc_clip_f/research/view_risk/cli.py)
- [test_view_risk_evaluation.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/tests/research/test_view_risk_evaluation.py)
- [test_view_risk_production.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/tests/research/test_view_risk_production.py)

No dependency changes, commits, pushes, private-data reads, downloads, or real training occurred. Accepted metric, fusion, and head numerics were not changed.

Remaining work is limited to P4C’s comprehensive smoke/cost/CI coverage and P5’s provision of actual audited datasets, public resources, and run execution. No real-data readiness, pretrained parity, speed, pilot result, or uncertainty improvement is claimed.