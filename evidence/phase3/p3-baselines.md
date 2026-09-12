Implemented P3B using one fresh GPT-5.6-Sol xhigh agent with no nested agents. No commits, dependency changes, training, or accepted-source modifications were made.

### Acceptance mapping

1. Added all scalar controls with explicit orientation/probability status, tune-only positive temperature fitting, frozen-argmax checks, omission singleton handling, and monotone logistic probability adaptation.
2. Added masked DS features with semantic branch-1/branch-2/root conflict placement, RoleManifest-bound fit access, confidence-fit training, and tune-only regularization selection.
3. Added owned masked MV-ACN adapters preserving B/L dropout and CLS conventions, exact full-view legacy parity, subset support, and current-cache TCP/MSE and correctness/BCE objectives.
4. Added ViLU adaptation with learned Q/K/V attention, observed-view pooling, frozen-prediction text selection, and explicitly documented unweighted BCE adaptation.
5. Added independently projected/scaled same-input binary and four-class controls using canonical `[B,550]` features. Four-class confidence gathers probability at the frozen classifier prediction.
6. Exposed clean/augmented input and objective metadata, finite selection budgets, and reusable P3A ablation definitions without claiming achieved exposure matching.
7. Added scientific behavioral and refusal tests covering every requested area.

### Verification

- RED: expected `ModuleNotFoundError` before implementation.
- Focused GREEN: **15 passed**
- Full research suite: **206 passed**
- Ruff lint and formatting: passed
- `git diff --check` and untracked-file whitespace checks: passed

Parameter counts:

| Backbone | Candidate | Binary MLP | Four-class | MV-ACN |
|---|---:|---:|---:|---:|
| ViT-B/32 | 433,436 | 433,651 | 433,243 | 2,471,425 |
| ViT-L/14-336 | 466,204 | 466,419 | 466,011 | 4,340,993 |

Both same-input controls are within approximately 0.05% of the corresponding candidate budget. The declared finite budget is three seeds, one architecture trial per learned method, at most 20 RSNA or 50 DDSM checkpoints per seed, and four DS regularization candidates.

### Main interfaces

- `scalar_baseline_scores(scores [B,4], temperature=...)`
- `TemperatureScaler.fit(scores, labels, manifest, exam_keys)`
- `MonotoneLogisticProbabilityAdapter.fit(score, error, manifest, exam_keys)`
- `absolute_omission_sensitivity(differences [B,4,4], valid [B,4])`
- `build_ds_features(...) -> DSBaselineFeatures(values [B,8], valid [B,8], ...)`
- `DSLogisticErrorControl.fit_with_tune_selection(...)`
- `MaskedMVACNAdapter(...) -> MVACNOutput`
- `compute_mvacn_objective(output, CachedTargets, objective)`
- `ViLUFailureAdapter(...) -> ViLUOutput`
- `compute_vilu_failure_loss(output, CachedTargets)`
- `SameInputMLP(raw: RawHeadInputs)`
- `SameInputDensityControl(raw: RawHeadInputs)`
- `compute_same_input_error_loss(...)`
- `compute_density_control_loss(...)`
- `p3a_ablation_definitions(base: RelationAwareHeadConfig)`

### Changed files

- [baselines.py](/home/tianqini/research/.ivr-worktrees/p3-baselines/src/mmdc_clip_f/research/view_risk/baselines.py)
- [test_view_risk_baselines.py](/home/tianqini/research/.ivr-worktrees/p3-baselines/tests/research/test_view_risk_baselines.py)
- [p3-baselines-tdd.md](/home/tianqini/research/.ivr-worktrees/p3-baselines/evidence/p3-baselines-tdd.md)

No scoped implementation issues remain. Calibration, exposure equality, cache/provenance audits, real-data readiness, clinical validity, and training outcomes remain deliberately unclaimed pending later orchestration and independent review.