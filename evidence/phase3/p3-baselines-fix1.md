Both confirmed target-binding issues are fixed by a fresh GPT-5.6-Sol xhigh agent. No nested agents, commits, dependency changes, training, accepted-source edits, or reviewer-worktree changes occurred.

### Fixes

- All four learned controls now bind outputs to current parent scores and masks. Their losses reject stale `CachedTargets` even when clean and stressed scores share the same argmax.
- Current observed/removal masks are validated for consistency, including singleton semantics.
- `TemperatureScaler.fit` now compares supplied labels exactly with authoritative densities from selected tune `RoleManifest` rows before optimization.
- Inference features remain label-free.
- Protection is scoped to accidental stale binding; complete hostile-caller/artifact provenance remains later orchestration work.

### API changes

- `MaskedMVACNAdapter.forward(..., *, classifier_prediction [B], current_scores [B,4])`
- `MaskedMVACNAdapter.output_from_logit(..., *, current_scores [B,4], observed_mask [B,4])`
- `ViLUFailureAdapter.forward(..., *, classifier_prediction [B], current_scores [B,4])`
- `MVACNOutput`, `ViLUOutput`, `SameInputOutput`, and `DensityControlOutput` now carry:
  - `current_scores [B,4]`
  - `observed_mask [B,4]`
  - `removal_valid_mask [B,4]`
- Loss signatures and `TemperatureScaler.fit` signature remain unchanged; validation is stronger.

### TDD and verification

- RED: **2 intended failures**
  - Manifest-density mismatch was accepted.
  - Learned outputs lacked current-score binding.
- Focused GREEN: **17 passed**
- Full research suite: **208 passed**
- Ruff lint/format and `git diff --check`: passed
- MV-ACN parity and B/L parameter-count tests remain green; counts are unchanged.

### Changed files

- [baselines.py](/home/tianqini/research/.ivr-worktrees/p3-baselines/src/mmdc_clip_f/research/view_risk/baselines.py:130)
- [test_view_risk_baselines.py](/home/tianqini/research/.ivr-worktrees/p3-baselines/tests/research/test_view_risk_baselines.py:198)
- [p3-baselines-tdd.md](/home/tianqini/research/.ivr-worktrees/p3-baselines/evidence/p3-baselines-tdd.md:58)

No scoped implementation issues remain. No commit was created.