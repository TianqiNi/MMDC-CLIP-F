Reviewed commit `4ff5289d19212bb0d94b04eed18b761b82506b2e` against base `3541fc0d8749bda97c5acbe4ba632dbf2b37da3d`.

## Finding

- **High — The public training replay path bypasses the held-out-family and strong-severity restrictions.**
  [TrainingSampleSpec.from_dict](/home/tianqini/research/.ivr-worktrees/p2-perturbations-review1/src/mmdc_clip_f/research/view_risk/perturbations.py:199) accepts arbitrary combinations without validating training support. [realize_training_parent](/home/tianqini/research/.ivr-worktrees/p2-perturbations-review1/src/mmdc_clip_f/research/view_risk/perturbations.py:664) then applies the supplied perturbation without calling `validate_training_perturbation`. Thus a serialized training draw containing `contrast/mild` or `gaussian_noise/strong` is realized successfully, despite both being rejected by the standalone guard.

  Synthetic reproduction produced:

  ```text
  contrast rejected: contrast is held out from confidence-fit realized_changed=True
  gaussian_noise rejected: strong severity is evaluation-only ... realized_changed=True
  ```

  The same path also accepts inconsistent modes/strata because every mode other than `"common"` falls through to the per-view branch. Validate restored `TrainingSampleSpec` invariants and enforce the shared fit/tune perturbation restrictions at the training-realization boundary. This needs no dependency on P2A; operation/provenance binding may remain P2C work.

## Checks run

- Candidate-specific tests: `24 passed in 0.55s`
- Full synthetic research suite: `64 passed in 0.66s`
- Ruff lint: passed
- Ruff formatting check: passed
- `git diff --check`: passed
- Verified base ancestry, exact HEAD, three-file change scope, and clean worktree
- Manually audited numerical parameters, 224/336 scaling, shared random fields, exact tensor reuse, mask probabilities, and 48-cell nested panel structure; no additional medium-or-higher defects found

No real patient data, training, or outcome access occurred. The passing checks establish synthetic behavior only; real-data readiness remains unexecuted.

**Verdict: REQUEST CHANGES / NOT ACCEPTED.**
