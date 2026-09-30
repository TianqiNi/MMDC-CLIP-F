# P5A selector implementation and final-fit review

Reviewed implementation commit: `d3d6a4de43d327e0a8a0d3f0cee1b5dcf848a407`.
Reviewer: orchestrator. This is a self-review, not an independent agent review;
the user's later instruction prohibits further AI subagents.

## Implementation acceptance

- The same role guard authorizes tune records before preprocessing. Empty authorized
  cohorts fail before image reads; neither pilot nor locked test is selected.
- All 50 candidates remain mandatory. The model is in evaluation mode; the
  deterministic tensors, original batch size/order and summed multiclass NLL
  calculation are unchanged. Cache defaults to disabled.
- The optional budget is checked before image access. It bounds retained image
  and label tensors; model/worker/Python overhead is documented separately.
- A new regression failed before implementation and passes afterward. It proves
  exact equality of all candidate NLLs and the reduction from 200 image reads to
  four on its one-exam fixture. It tests insufficient-budget and forbidden-role
  failures. Existing production workflow coverage exercises multiprocessing.
- Full research suite: 299 passed. Ruff, compilation and diff checks passed.
- Remote CI passed on the exact reviewed commit: push run `36730188412` and PR
  run `36730194413`.

No actionable Critical, High or Medium findings in this review. The small
regression is software evidence; it does not estimate a scientific effect size.

## Final fit acceptance

The unchanged production artifact loader revalidated the referenced readiness,
initialization, configuration, all checkpoint file hashes and all state hashes.
All 50 epochs, 3,209 classifier-fit patients and 53,500 updates are accounted for.
The final model equals the resume model exactly, all 19,900 floating tensors
across the checkpoint history are finite, and all 1,194 floating optimizer
state tensors are finite. The original Adam learning rate and weight decay are
verified. See `fit-validation.json` for identities and measured runtime.

Validation was run on the host: the sandbox's synthetic ancestor `.git` directory
caused the fail-closed private-path check to reject reads. No guard was removed
or relaxed; a diagnostic wrapper only printed metadata on exceptions and
re-raised them. The successful run used all original integrity checks.

Tune selection and diagnostic results are accepted separately after their
actual execution; this review does not claim an uncertainty-method benefit.

## Executed selection and diagnostic acceptance

Selection exited zero after evaluating exactly epochs 1–50. Epoch 38 is the
minimum-NLL candidate, with earlier-epoch tie breaking preserved. The separately
loaded selected artifact passed its production provenance validator; its tune
NLL was reproduced exactly over all 370 unique authorized exams. Aggregate
confusion counts sum to 370, with 290 correct predictions. Both executions
exited zero with empty error logs. Results and limitations are in `results.md`.
P5A is accepted for this RSNA/backbone run. The proposed signed-effect signal
still requires a support audit and matched uncertainty benchmarks; acceptance
of P5A does not establish the scientific hypothesis.
