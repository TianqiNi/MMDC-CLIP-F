# P3B fair-baseline TDD evidence

Date: 2026-09-10. Scope is generic software on synthetic fixtures. No model
training, patient data, checkpoint download, metric run, pilot access, or test
access occurred. `implementation_status.md` still describes P3A as pending in a
historical table; the accepted source contract and orchestrator-supplied P3A
review at `c3e7a675f51085bd8648d7a2f1a29cd503d7eed8` governed this task.

## RED / GREEN

Interpreter and import contract for every pytest run:

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python
```

RED, after writing behavioral/refusal tests and before creating the module:

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research/test_view_risk_baselines.py
```

Result: collection error, `ModuleNotFoundError` for
`mmdc_clip_f.research.view_risk.baselines` (one error). This established that
the new controls were absent.

Focused GREEN:

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research/test_view_risk_baselines.py
```

Result: **15 passed in 1.03s**.

Compatibility GREEN:

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research
```

Result: **206 passed in 1.82s**.

Static/worktree checks:

```text
/home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m ruff check src/mmdc_clip_f/research/view_risk/baselines.py tests/research/test_view_risk_baselines.py
/home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m ruff format --check src/mmdc_clip_f/research/view_risk/baselines.py tests/research/test_view_risk_baselines.py
git diff --check
```

Results: Ruff passed; both files already formatted; `git diff --check` emitted
no errors. A no-index whitespace check is run after adding this untracked
evidence file because ordinary `git diff` does not include untracked files.
`git diff --no-index --check /dev/null <file>` emitted no whitespace errors for
each of the three new files (exit 1 is the expected no-index "files differ"
status).

## Review-fix RED / GREEN cycle

The two orchestrator reproductions were converted to behavioral/refusal tests
before changing the implementation. The focused RED command was:

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research/test_view_risk_baselines.py -k 'temperature_refuses_labels or learned_objectives_reject'
```

Result before the fix: **2 failed**. `TemperatureScaler.fit` did not raise when
supplied labels `[3,3]` contradicted the selected tune-manifest densities
`[0,1]`; the learned-control regression failed because `MVACNOutput` had no
current-score binding.

The same focused command after implementation produced **2 passed, 15
deselected in 0.56s**. The complete P3B file then produced **17 passed in
0.99s**, and the full research suite produced **208 passed in 1.84s**.

Post-fix static checks ran Ruff formatting, Ruff lint, Ruff format-check, and
`git diff --check` on the owned files. All passed; Ruff mechanically reformatted
one file before the final check.

## Legacy-order review-fix RED / GREEN cycle

This is a continuation of the original P3B implementation. Before changing the
adapter, the focused order command was:

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research/test_view_risk_baselines.py -k 'mvacn_is_exact_for_each_legacy_order or mvacn_filters_configured_order'
```

RED result: **5 failed** because the adapter accepted no explicit
`legacy_view_order`; consequently DDSM parity and configured subset order were
not expressible. GREEN result for the same command: **5 passed, 15 deselected in
0.70s**. The complete P3B file then produced **20 passed in 1.08s** and the full
research suite produced **211 passed in 1.88s**. Both B/L conventions now pass
bit-exact copied-weight parity for RSNA order `L_CC,L_MLO,R_CC,R_MLO` and DDSM
order `L_CC,R_CC,L_MLO,R_MLO`; a token-capture fixture verifies order filtering
for a proper subset and invalid missing/duplicate/unknown orders fail closed.
Final post-format verification produced **20 passed in 1.13s** and **211 passed
in 1.92s**; Ruff lint/format and `git diff --check` passed.

## Acceptance mapping

1. `scalar_baseline_scores` exposes MSP, tune-temperature MSP, margin, negative
   entropy, and fixed-T=1 Energy (`-logsumexp`) with explicit ranking orientation
   and probability status. `TemperatureScaler.fit` requires tune-manifest access,
   minimizes tune NLL with positive T, and checks frozen argmax invariance. Fitting rows are bound to exact ordered exam keys selected from the manifest.
   Supplied temperature labels must also equal the authoritative densities of
   those selected manifest records before optimizer construction.
   `absolute_omission_sensitivity` implements the protocol maximum over valid
   removal-wise mean absolute class-score differences; singleton output is zero
   with `no_removal=True`. `MonotoneLogisticProbabilityAdapter` refuses use
   before tune fitting and constrains the error-oriented slope positive.
2. `build_ds_features` returns fused `4/sum(alpha)`, four canonical per-view
   vacuities, semantic branch-1/branch-2/root conflict slots, and masks. Tests
   cover pruned RSNA and DDSM trees and compare semantic slots to the accepted
   active operation trace. `DSLogisticErrorControl` accepts only this masked
   feature type, fits scaler/weights on manifest-selected confidence-fit records,
   and selects a finite regularization grid on manifest-selected tune records.
3. `MaskedMVACNAdapter` owns `MVACNHead`, requires an explicit validated
   legacy view order, and filters that order when views are missing while retaining
   canonical named masks. Synthetic B and L tests copy weights and prove exact
   (`rtol=atol=0`) full-view equality for both RSNA and DDSM legacy orders plus
   finite 1/2/3-view outputs.
   B uses dropout 0.2/CLS excluded; L uses 0.3/CLS included.
   `compute_mvacn_objective` provides TCP/MSE and correctness/BCE. Adapter
   outputs now carry the actual input parent scores and observed/removal masks;
   the objective requires exact equality with `CachedTargets.scores` and its
   removal-valid mask, rejecting stale TCP, same-argmax clean scores, changed
   frozen predictions, and mismatched current masks.
4. `ViLUFailureAdapter` pools only observed frozen projected visual embeddings;
   learned Q/K/V use visual query and all class-text keys/values. Its MLP receives
   pooled visual, frozen predicted-class text, and attended text. Tests establish
   image-conditioned attention, nonzero Q/K gradients, absent-slot invariance,
   and frozen-prediction text selection. `compute_vilu_failure_loss` is explicit
   unweighted failure BCE. `VILU_PRIMARY_SOURCE_NOTE` links ViLU sections 4.2-4.3
   and the authors' repository, and discloses that this is not the source batch-
   weighted BCE. ViLU outputs carry the same current score/mask binding used by
   the matched objective.
5. `SameInputMLP` and `SameInputDensityControl` independently own accepted P3A
   preparation projections and their own learned affine scalers. Their vector is
   canonical view `[B,4,128]`, omission `[B,4,4]`, global `[B,13]`, then observed
   `[B,4]`, removal-valid `[B,4]`, and no-removal `[B,1]`: 541 continuous plus
   nine mask features, total `[B,550]`. No label, correctness, agreement, or
   prediction metadata is an inference feature. All parameters receive gradients.
   Density confidence gathers its probability at the frozen classifier prediction.
   Both outputs carry their source `RawHeadInputs` current scores and masks for
   checked target binding at objective evaluation.
6. `BASELINE_DEFINITIONS` distinguishes clean four-view, labeled masked-clean,
   and matched augmentation/mask exposure; all learned target consumers require
   current realized targets. `actual_exposure_verified=False` is deliberate:
   P4 must implement and run exposure/provenance matching. `p3a_ablation_definitions`
   reuses `RelationAwareHeadConfig` and loss weight rather than copying P3A.
7. The focused tests exercise hand-computed scalar signs, role refusals, both
   pruned trees, parity/subsets, distinct objectives, stale-target refusal,
   learned ViLU attention, feature order/projection independence, parameter
   matching, all-parameter use, and frozen-prediction probability gathering.

## Counts and finite selection declaration

Runtime counts include every learned projection/scaler:

| Backbone | P3A candidate | Same-input binary | Difference | Four-class | Difference | MV-ACN |
|---|---:|---:|---:|---:|---:|---:|
| ViT-B/32 | 433,436 | 433,651 | +0.0496% | 433,243 | -0.0445% | 2,471,425 |
| ViT-L/14-336 | 466,204 | 466,419 | +0.0461% | 466,011 | -0.0414% | 4,340,993 |

`CONTROL_SELECTION_BUDGET` exposes one architecture trial per learned method,
three fixed seeds, at most 20 RSNA or 50 DDSM checkpoint candidates per seed,
and four DS regularization candidates. Temperature/scalar logistic each have one
prespecified calibration form. P4 must apply the common checkpoint budget and
record actual updates/trials; no selection was run here.

## Callable interfaces and shapes

- `scalar_baseline_scores(scores [B,4], temperature=...) -> Mapping[str, ScalarScore[B]]`
- `TemperatureScaler.fit(scores [B,4], labels [B], manifest=RoleManifest, exam_keys=[B])`;
  `transform(scores) -> [B,4]`
- `MonotoneLogisticProbabilityAdapter.fit(score [B], error [B], manifest=..., exam_keys=[B])`;
  `predict_error_probability(score) -> [B]`
- `absolute_omission_sensitivity(differences [B,4,4], valid [B,4]) -> ScalarScore[B]`
- `build_ds_features(logits_by_view, observed, fusion_pairs=...) ->`
  `DSBaselineFeatures(values [B,8], valid [B,8], observed [B,4], conflicts [B,3])`;
  `.as_logistic_input() -> [B,16]`
- `DSLogisticErrorControl.fit_with_tune_selection(confidence_features, errors,
  tune_features, tune_errors, confidence_manifest=..., confidence_exam_keys=[Bfit], tune_manifest=..., tune_exam_keys=[Btune])`;
  `.predict_error_probability(features) -> [B]`
- `MaskedMVACNAdapter(config, *, legacy_view_order=[4 names])`,
  `.for_backbone(backbone, *, legacy_view_order=[4 names])`, and
  `.from_legacy(legacy, *, legacy_view_order=[4 names])` require the dataset's
  configured permutation of all four canonical views.
- `MaskedMVACNAdapter(hidden_by_view, observed, *, classifier_prediction [B],
  current_scores [B,4]) -> MVACNOutput(confidence_logit [B], confidence [B],
  classifier_prediction [B], current_scores [B,4], observed_mask [B,4],
  removal_valid_mask [B,4])`
- `MaskedMVACNAdapter.output_from_logit(logit [B], classifier_prediction [B], *,
  current_scores [B,4], observed_mask [B,4]) -> MVACNOutput`
- `compute_mvacn_objective(output, CachedTargets, MVACNObjective) -> scalar`
- `ViLUFailureAdapter(projected_visual_embeddings [B,4,Dv], observed_mask [B,4],
  *, classifier_prediction [B], current_scores [B,4]) -> ViLUOutput(error/confidence
  [B], attention [B,4], current_scores [B,4], observed/removal masks [B,4])`
- `compute_vilu_failure_loss(output, CachedTargets) -> scalar`
- `SameInputMLP(raw: RawHeadInputs) -> SameInputOutput(error/confidence [B],
  classifier_prediction [B], current_scores [B,4], observed/removal masks [B,4])`;
  `compute_same_input_error_loss(output, CachedTargets) -> scalar`
- `SameInputDensityControl(raw) -> DensityControlOutput(class_logits [B,4],
  frozen_prediction_probability [B], classifier_prediction [B], current_scores
  [B,4], observed/removal masks [B,4])`;
  `compute_density_control_loss(output, CachedTargets) -> scalar`
- `p3a_ablation_definitions(base: RelationAwareHeadConfig) -> immutable mapping`

## Limitations and required later checks

This phase establishes software contracts, not calibration, uncertainty validity,
fairness, clinical utility, novelty, real-data readiness, or achieved exposure
matching. Output-level score/mask checks protect against accidental clean-target
reuse for changed realized inputs, including unchanged-argmax cases; they are not
authentication against a hostile caller that fabricates every binding consistently.
DS values are diagnostic features, not validated uncertainty
decompositions. MSP is a classifier class probability, not a calibrated
correctness probability. Learned sigmoid outputs are probability estimates, not
demonstrated calibration. The ViLU loss is the protocol-permitted unweighted
adaptation, not an exact reproduction of source weighting. P4/P5 must verify
same patients, masks, perturbation draws, configured legacy MV-ACN view order,
optimizer/update/checkpoint budgets,
frozen classifier parameters and predictions, cache provenance/current-target
bindings, train/tune role access, realized parameter/update counts, and actual
clean/augmented exposure before pilot. Bounded/streamed ViT-L processing and
exact parent reuse remain mandatory; this task created no hidden-token artifacts.

Changed files are only:

- `src/mmdc_clip_f/research/view_risk/baselines.py`
- `tests/research/test_view_risk_baselines.py`
- `evidence/p3-baselines-tdd.md`

Unresolved implementation issues: none found in scoped synthetic verification.
Independent review and integration remain with the orchestrator.
