## P4B review — NOT ACCEPTED

Reviewed exact commit: **`2711fbf0a73365633090eedeaa6133004c140071`**

Accepted base: `8562a4b0b6fe5c3baf0e463c9ab213e00aae1dcd`  
Latest-fix parent: `fcdb2532eaaf9906644cfcff3dafe8f3087d6df0`

### Findings

1. **High — learned-selection reload does not verify the referenced tune caches**

   Location: [evaluation.py:1036](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review4/src/mmdc_clip_f/research/view_risk/evaluation.py:1036), [production.py:513](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review4/src/mmdc_clip_f/research/view_risk/production.py:513)

   Reproduction: after the connected workflow generated a valid correctness-MVACN selection, I truncated `tune-cache-0.safetensors` to one byte without changing the tune-index JSON, selection record, fit artifact, or checkpoints. A fresh-process `load_model_artifact_selection()` still succeeded:

   ```text
   before correctness_mvacn 20
   after_tune_cache_corruption correctness_mvacn 20
   ```

   The reload hashes the tune-index file but never reopens its referenced cache metadata/tensor artifacts or rederives the stored AURCs. `load_pilot_plan()` inherits this gap when replaying selection records.

   Impact: an unreadable, changed, or stale tune realization can remain accepted after plan freeze/reload, so the selected epoch and clean guardrail are no longer backed by current tune evidence.

   Minimal fix: bind every referenced tune cache metadata/tensor digest in the selection artifact and validate them on reload, or reload the bound tune index and each `CacheBundle` and deterministically recompute the per-epoch tune results and selection.

2. **High — authoritative/evaluation cells are not tied to the actual realized cache stresses**

   Location: [cli.py:507](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review4/src/mmdc_clip_f/research/view_risk/cli.py:507), [cli.py:1248](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review4/src/mmdc_clip_f/research/view_risk/cli.py:1248), [cli.py:1288](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review4/src/mmdc_clip_f/research/view_risk/cli.py:1288)

   Reproduction: I referenced a verified clean cache but declared its evaluation cell as `primary / contrast / mild / L_CC`. `_evaluation_cache_index()` accepted it:

   ```text
   accepted_cell ... family='contrast', severity='mild', target_view='L_CC'
   actual_L_CC_spec {'family': 'clean', 'severity': None, ...}
   ```

   The score path checks that the declared cell exists in the authoritative registry and that classifier/cohort identities match, but never compares the cell with the cache’s observed mask and perturbation provenance. The authoritative registration command also imports predictions directly from caller-provided `--input` JSON rather than producing them through verified classifier inference.

   Impact: clean or otherwise incorrect/reused inputs can populate the nominal 48-cell primary panel, strong panels, or held-out-family panels and support invalid aggregate conclusions.

   Minimal fix: generate authoritative predictions from verified evaluation bundles using the selected classifier, and validate every declared cell against the bundle’s mask and complete `PerturbationSpec`, including family, severity, variant, target/common-mode behavior, and seed-derived realization identity.

3. **High — fitted analytic controls accept features and predictions unbound to the selected classifier**

   Location: [cli.py:464](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review4/src/mmdc_clip_f/research/view_risk/cli.py:464), [cli.py:477](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review4/src/mmdc_clip_f/research/view_risk/cli.py:477), [cli.py:1070](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review4/src/mmdc_clip_f/research/view_risk/cli.py:1070)

   Reproduction: the public `view-risk-fit-control` command successfully fitted a DS logistic artifact from standalone JSON containing caller-authored DS values and classifier predictions:

   ```text
   status: control_fitted
   parameter_count: 17
   selection_trials: 4
   ```

   Those input schemas carry no selected-classifier checkpoint, cache, manifest, score, or DS-path provenance. Loading the selected classifier artifact does not connect it to the supplied rows.

   Impact: mandatory TS and DS controls can be fitted on predictions/features from another classifier or fabricated values, breaking the shared-frozen-classifier and matched-control requirements.

   Minimal fix: derive scalar scores, predictions, and DS features from verified role-bound cache bundles inside the fitting command, bind their cache/classifier/manifests, and revalidate those referenced artifacts when loading the control artifact.

### Previous-finding disposition

- **Critical synthetic provenance promotion:** fixed. Production plan reload reconstructs the classifier through persisted initialization, readiness, fit, tune, and checkpoint evidence.
- **Conflicting authoritative prediction identities:** fixed narrowly. A plan owns one absolute exclusive registry path, and copied plans cannot register a second file. The provenance of the registered predictions remains affected by finding 2.
- **Incomplete learned/control artifacts:** not fixed; findings 1 and 3 remain.
- **Disconnected production workflows:** partially fixed, but not accepted; classifier and learned training paths exist, while authoritative evaluation and analytic-control inputs remain disconnected as described above.

### Validation

- Prior-finding focused regressions: **5 passed**
- Connected synthetic production workflow: **1 passed**
- Full research suite: **258 passed in 11.41s**
- Ruff over `src` and `tests/research`: **passed**
- Latest-fix delta `git diff --check`: **passed**
- Worktree remained clean.
- Accepted-base diff still reports the two previously known trailing spaces in `p4-training-review1.md`; nonfunctional and not a review finding.

No real data, downloads, actual public-weight loading, or real training were performed. P4C smoke/cost/CI and P5 resource/run work were not assessed.