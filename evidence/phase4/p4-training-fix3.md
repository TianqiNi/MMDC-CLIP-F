P4B remains unaccepted: findings 1–2 are substantially addressed, but findings 3–4 still have production workflow gaps requiring another implementation cycle and independent review.

## Review findings

1. **Synthetic versus real readiness — core invariant fixed**

   - Injected factories now create only `synthetic_injected` provenance and cannot become pilot eligible.
   - Added a pinned production loader using `BACKBONES`, `PROMPTS`, `CLIPModel`, `CLIPTokenizer`, and `MultiViewCLIPClassifier`.
   - Production initialization records actual model-state content, tokenizer inputs, fusion tree, revision, and prompts.
   - Real readiness now verifies source, mapping, denylist, and every referenced image byte.
   - Readiness artifacts are rehashed when loaded.
   - Raw public classifier JSON cannot acquire production provenance through the CLI.
   - Remaining gap: no persistent production initialization/selected-classifier artifact workflow or end-to-end classifier CLI.

2. **Incomplete/differing pilot predictions — fixed at evaluation API/CLI boundary**

   - Added an authoritative, label-free classifier prediction artifact bound to the already-frozen plan, classifier, and pilot manifest.
   - Every evaluation cell must contain the exact authorized exam cohort.
   - Every confidence method and seed must reuse exactly the authoritative row keys and classifier decisions.
   - Missing, duplicate, changed, or stale prediction rows are rejected.
   - Real evaluation additionally rehashes the selected model artifact.
   - Plan freezing still occurs before pilot outcomes; authoritative predictions are created afterward without outcomes.

3. **Arbitrary selection hashes — partially fixed**

   - Learned selections can now be created from typed `TrainingResult` and complete per-epoch tune results.
   - The factory replays the frozen epoch budget, clean-reference selection, guardrail, tie ordering, and no-eligible behavior, and verifies selected checkpoint bytes.
   - Analytic controls produce content-bound artifacts containing method-specific fitting, parameter, update, exposure, scaler, classifier, config, and manifest evidence.
   - Pilot freezing rehashes both model and workflow evidence and checks confidence-fit/reference bindings.
   - Arbitrary caller-declared production selection construction now fails closed.
   - Remaining gap: learned selection evidence is not yet reloadable through a verified production CLI record. Consequently, the real `view-risk-freeze-pilot` path is not end-to-end usable.

4. **Disconnected production workflows — incomplete**

   Implemented generic pieces include:

   - pinned production CLIP loading;
   - role-bound fresh classifier fitting API;
   - tune-only classifier selection API;
   - authorized per-checkpoint tune metric generation;
   - role-bound temperature, DS-logistic, raw-scalar, and monotone-calibration artifacts;
   - analytic confidence inference;
   - learned confidence training from streamed caches;
   - authoritative prediction freezing and confidence-row generation for P4A.

   Still required in P4B—not P4C/P5:

   - production classifier image/batch adapter and CLI fit/resume/checkpoint path;
   - authorized per-epoch classifier/confidence inference connected directly to selection CLI;
   - analytic-control CLI dispatch and reloadable selection records;
   - learned/control model loading connected directly to confidence prediction generation;
   - an end-to-end production freeze path using verified classifier/readiness/selection artifacts.

## Files changed

- [training.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/src/mmdc_clip_f/research/view_risk/training.py)
- [evaluation.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/src/mmdc_clip_f/research/view_risk/evaluation.py)
- [artifacts.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/src/mmdc_clip_f/research/view_risk/artifacts.py)
- [cli.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/src/mmdc_clip_f/research/view_risk/cli.py)
- [test_view_risk_training.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/tests/research/test_view_risk_training.py)
- [test_view_risk_evaluation.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/tests/research/test_view_risk_evaluation.py)
- [test_view_risk_artifacts.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/tests/research/test_view_risk_artifacts.py)
- [test_view_risk_cache.py](/home/tianqini/research/.ivr-p4-worktrees/p4-training/tests/research/test_view_risk_cache.py)

The cache test change only reclassifies its injected fixture as synthetic; its integrity expectations were not weakened. Review evidence files remain untracked and untouched.

## RED/GREEN evidence

Observed RED states included:

- Focused preserved draft: `6 failed, 77 passed` from invalid selection/tune constructors.
- New analytic-artifact test: collection failed before `artifacts.py` existed.
- Role-order regression: `1 failed, 1 passed` before exact-role rejection was moved ahead of the reader.
- Learned workflow regression initially rejected a deliberately stale classifier binding.

Final checks:

- Focused P4B set: `86 passed`.
- Current final focused training/evaluation/artifact set: `24 passed`.
- Full research suite: `250 passed`.
- Ruff: `All checks passed!`
- Compileall: passed.
- CLI help/dispatch: passed.
- `git diff --check`: passed.

No private data, locked-test outcomes, external downloads, or real training were accessed. No commits or pushes were made. No empirical readiness, parity, speed, pilot performance, or uncertainty improvement is claimed.