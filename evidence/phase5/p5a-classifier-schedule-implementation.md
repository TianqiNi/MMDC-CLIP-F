Implemented the classifier/confidence schedule separation without commits or pushes.

### Implementation

- Added an immutable, serialized `FrozenClassifierSchedule`:
  - 50 epochs
  - Adam, learning rate `1e-7`, weight decay `1e-5`
  - batch size 3
  - seed 42
  - AMP disabled
  - RandAugment `3/9/31`
  - checkpoint candidates 1–50
- Preserved RSNA confidence training unchanged at 20 epochs, batch size 6, Adam `1e-4`, weight decay 0, seeds 42/43/44, and the existing method/search tables.
- Classifier bindings now use `classifier_schedule_sha256`; confidence bindings continue to use `search_table_sha256`. They cannot be interchanged.
- Resume metadata, classifier-fit artifacts, and selected-classifier artifacts verify the schedule and hash.
- Bumped strict schemas:
  - run config v2
  - classifier-fit artifact v2
  - selected-classifier artifact v3
- Explicit v1 configurations and older classifier artifacts are refused rather than reinterpreted.
- Classifier fit, batching, checkpoint enumeration, tune NLL selection, reload, and CLI status now all use the separate 50-epoch schedule.
- Preserved tune-only multiclass NLL selection and all readiness/role/public-initialization protections.
- Added the reproducible RSNA configuration: [rsna_vit_b_32_p5a.yaml](/home/tianqini/research/MMDC-CLIP-IVR/configs/research/rsna_vit_b_32_p5a.yaml).

Key implementation paths:

- [training.py](/home/tianqini/research/MMDC-CLIP-IVR/src/mmdc_clip_f/research/view_risk/training.py:170)
- [production.py](/home/tianqini/research/MMDC-CLIP-IVR/src/mmdc_clip_f/research/view_risk/production.py:721)
- [cli.py](/home/tianqini/research/MMDC-CLIP-IVR/src/mmdc_clip_f/research/view_risk/cli.py:1439)
- [test_view_risk_training.py](/home/tianqini/research/MMDC-CLIP-IVR/tests/research/test_view_risk_training.py:637)
- [test_view_risk_production.py](/home/tianqini/research/MMDC-CLIP-IVR/tests/research/test_view_risk_production.py:86)
- [test_view_risk_evaluation.py](/home/tianqini/research/MMDC-CLIP-IVR/tests/research/test_view_risk_evaluation.py:280)

### RED/GREEN and validation

- RED: focused regressions initially failed during collection because `FrozenClassifierSchedule` and its production flow did not exist.
- GREEN focused regressions: 3 passed.
- Affected training/production/evaluation suites: 34 passed.
- Full research suite, including smoke coverage: 293 passed.
- Ruff: all checks passed.
- Compileall: passed.
- CLI help and checked-in configuration validation: passed.
- Validated hashes:
  - classifier schedule: `0b94cef276cf26903a6bafd1a1ee89d5823fd2d9decc5f5f941421a97e72358b`
  - confidence search table: `d72cb3b80ea524d79fa957a7f173fe72e1baf2268c49f076eb2cbd2b1457fe3b`
  - expanded run config: `d29d7a373ecb6739af0f5923588314da451d8b69c39a66e43b83138b373a3d8a`

All tests were software-only and synthetic. No patient data, pretrained downloads, real training, or scientific performance outcomes were accessed.

### Herdr run guidance

Use a new external run directory:

```bash
WORKTREE=/home/tianqini/research/.ivr-p5-worktrees/classifier-schedule
PYTHON_BIN=/home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python
CONFIG="$WORKTREE/configs/research/rsna_vit_b_32_p5a.yaml"

PYTHONPATH="$WORKTREE/src" "$PYTHON_BIN" -m mmdc_clip_f.cli \
  view-risk-fit-classifier \
  --config "$CONFIG" \
  --manifest "$CLASSIFIER_FIT_MANIFEST" \
  --manifest-binding "$CLASSIFIER_FIT_BINDING" \
  --private-root "$PRIVATE_ROOT" \
  --image-root "$IMAGE_ROOT" \
  --readiness "$READINESS_ARTIFACT" \
  --initialization "$RUN_ROOT/public-initialization.json" \
  --checkpoint-directory "$RUN_ROOT/classifier-checkpoints" \
  --resume-checkpoint "$RUN_ROOT/classifier-resume.pt" \
  --output "$RUN_ROOT/classifier-fit.json" \
  --device cuda:0
```

Resume with every binding argument unchanged:

```bash
PYTHONPATH="$WORKTREE/src" "$PYTHON_BIN" -m mmdc_clip_f.cli \
  view-risk-fit-classifier \
  --config "$CONFIG" \
  --manifest "$CLASSIFIER_FIT_MANIFEST" \
  --manifest-binding "$CLASSIFIER_FIT_BINDING" \
  --private-root "$PRIVATE_ROOT" \
  --image-root "$IMAGE_ROOT" \
  --readiness "$READINESS_ARTIFACT" \
  --initialization "$RUN_ROOT/public-initialization.json" \
  --checkpoint-directory "$RUN_ROOT/classifier-checkpoints" \
  --resume-checkpoint "$RUN_ROOT/classifier-resume.pt" \
  --output "$RUN_ROOT/classifier-fit-resumed.json" \
  --resume \
  --device cuda:0
```

If the interrupted run never created its output artifact, the original output path may be reused. If `--stop-after-epoch` produced a partial artifact, use a new output filename as above.

After completion, perform NLL selection:

```bash
PYTHONPATH="$WORKTREE/src" "$PYTHON_BIN" -m mmdc_clip_f.cli \
  view-risk-select-classifier \
  --config "$CONFIG" \
  --fit-artifact "$RUN_ROOT/classifier-fit-resumed.json" \
  --manifest "$TUNE_MANIFEST" \
  --manifest-binding "$TUNE_BINDING" \
  --private-root "$PRIVATE_ROOT" \
  --image-root "$IMAGE_ROOT" \
  --output "$RUN_ROOT/selected-classifier.json" \
  --device cuda:0
```
