## Bounded partial-fix review

Reviewed commit: **`fcdb2532eaaf9906644cfcff3dafe8f3087d6df0`** (`fix(research): harden partial P4B provenance and prediction bindings`)
Compared against rejected candidate: `ed093038b9a5f7e297a8f55c3df821c7c72350be`
Decision: **NOT ACCEPTED — findings 1 and 2 remain incompletely fixed.**

### Critical — finding 1 remains exploitable through plan reload

The in-memory separation is materially improved:

- Injected factories now produce `synthetic_injected` and cannot become pilot eligible.
- The production loader uses the pinned model/revision and hashes loaded state.
- Real-readiness construction checks source, mapping, denylist, and referenced image bytes.

However, persisted plan reload bypasses those protections. [`_restore_frozen_record()`](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review2/src/mmdc_clip_f/research/view_risk/training.py:588) grants the private production token based solely on the JSON `"kind"`. [`_plan_from_payload()`](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review2/src/mmdc_clip_f/research/view_risk/evaluation.py:838) feeds it caller-controlled booleans and SHA-shaped strings, while [`load_pilot_plan()`](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review2/src/mmdc_clip_f/research/view_risk/evaluation.py:1051) verifies only a reproducible self-hash. It does not reload/re-hash the bound readiness audit or verify production initialization evidence.

Synthetic probe:

```text
kind= public_pretrained_fresh workflow_complete= True pilot_eligible= True
```

This was produced from a synthetic provenance dictionary by changing its kind, readiness boolean, and bindings to arbitrary valid-looking hashes, then invoking the same restoration path used by plan loading.

Required correction: production plan reload must verify independently persisted initialization/readiness artifacts and their current contents; it must not confer production provenance from self-hashed plan fields.

### High — finding 2 fixes cohort completeness but not shared prediction identity

[`_validate_complete_prediction_cohort()`](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review2/src/mmdc_clip_f/research/view_risk/evaluation.py:1224) correctly requires every represented cell to contain the exact authorized exam cohort. Evaluation also rejects omitted, duplicated, or changed rows at [`evaluation.py:1534`](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review2/src/mmdc_clip_f/research/view_risk/evaluation.py:1534).

But the plan does not bind one authoritative-prediction artifact SHA. Loading validates only the plan, classifier, and manifest identities at [`evaluation.py:1357`](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review2/src/mmdc_clip_f/research/view_risk/evaluation.py:1357), and the CLI accepts a prediction-artifact path independently for every evaluation invocation at [`cli.py:587`](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review2/src/mmdc_clip_f/research/view_risk/cli.py:587).

Independent two-exam probe:

```text
same_plan= True
different_authoritative_sha= True
accepted= candidate correctness_mvacn
```

Candidate used classifier prediction `0`; correctness-MVACN used `1`. Both complete cohorts were accepted against the same frozen plan.

Required correction: freeze or otherwise register exactly one authoritative prediction-set identity for the plan, then require every method/seed evaluation to match that identity.

### Known findings 3 and 4

- **Finding 3 remains partially fixed:** typed learned/analytic evidence is stronger, but verified learned-selection reload and a usable production freeze path remain incomplete.
- **Finding 4 remains open:** production classifier/batch CLI, direct per-epoch inference-to-selection wiring, analytic dispatch/reload, learned/control loading, and full production freeze remain disconnected.

No additional critical/high/medium issue was identified within the bounded subset.

### Checks performed

- Read both requested evidence reports.
- Verified the target is a direct child of the rejected candidate.
- Inspected the changed provenance, readiness, plan, selection/artifact, authoritative-prediction, evaluation, CLI, and focused-test boundaries.
- Focused pytest: **4 passed**.
- Two independent synthetic probes exposed the failures above.
- Worktree remained clean.
- `git diff --check` found trailing whitespace in two lines of `p4-training-review1.md`; this is non-functional.
- No real data, downloads, public-weight loading, training, commits, edits, or pushes were performed.

This was not a whole-system re-review and makes no claims about empirical readiness, model performance, or novelty.
