## P4B independent review

Reviewed commit: **`ed093038b9a5f7e297a8f55c3df821c7c72350be`**  
Accepted base: `8562a4b0b6fe5c3baf0e463c9ab213e00aae1dcd`  
Decision: **NOT ACCEPTED**

### Critical

1. Real-pilot eligibility can be asserted without verified data readiness or genuine public CLIP initialization.

[training.py:653](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review1/src/mmdc_clip_f/research/view_risk/training.py:653) labels any parameterized module returned by an injected callback—including `nn.Linear(1,1)`—as `public_pretrained_fresh`. [training.py:741](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review1/src/mmdc_clip_f/research/view_risk/training.py:741) marks arbitrary all-role manifests and a denylist as real-ready without binding a verified inventory/readiness artifact, dataset identity/count reconciliation, source existence, or an actual public-weight loader. The boolean is also accepted directly when classifier JSON is reconstructed at [cli.py:186](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review1/src/mmdc_clip_f/research/view_risk/cli.py:186), then trusted while freezing at [evaluation.py:548](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review1/src/mmdc_clip_f/research/view_risk/evaluation.py:548).

Reproduction: four one-row manifests with nonexistent image paths, free-form “verified” mapping strings, an empty denylist, and an injected `nn.Linear(1,1)` were passed through the public fit/tune APIs. Result:

```text
patient_ready=True
model_type=Linear
fresh_kind=public_pretrained_fresh
pilot_eligible=True
fit_software_only=False
```

Consequence: an unverified synthetic/non-CLIP setup can be labeled a verified real pilot and unlock pilot target/prediction processing. This does not unlock `locked_test`, but it defeats the mandatory real-data and fresh-initialization gate.

Required fix: make injected tiny initialization explicitly synthetic and non-promotable; require a verified production public-weight initialization artifact and an independently verified P2 readiness artifact. Do not deserialize real readiness from a boolean plus caller-supplied SHA-shaped strings.

### High

2. Pilot evaluation accepts incomplete cohorts and differing classifier predictions across methods.

The frozen payload at [evaluation.py:393](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review1/src/mmdc_clip_f/research/view_risk/evaluation.py:393) contains no frozen prediction artifact identity. Evaluation at [evaluation.py:696](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review1/src/mmdc_clip_f/research/view_risk/evaluation.py:696) verifies only that supplied rows are members of the manifest; it never requires every authorized exam. It evaluates one method/seed per invocation, so the P4A cross-method equality check is never exercised. The CLI accepts an arbitrary predictions path on every invocation at [cli.py:498](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review1/src/mmdc_clip_f/research/view_risk/cli.py:498).

Reproduction: with a two-exam pilot manifest, candidate and correctness-MVACN were each evaluated using only one exam, with predictions `0` and `1` respectively. Both succeeded:

```text
authorized_manifest_exams=2
candidate_evaluated_exams=1
baseline_evaluated_exams=1
different_frozen_predictions_both_accepted=True
```

Consequence: missing or favorable pilot cases can be omitted and methods can use different classifier predictions, invalidating paired comparisons.

Required fix: require exact manifest-exam completeness for every panel cell and verify all method/seed rows jointly against one frozen classifier-prediction identity, or bind and validate a single authoritative prediction artifact.

3. A complete pilot can be frozen from arbitrary hashes without evidence that mandatory controls were trained or selected.

`ModelArtifactSelection` validates only method, seed, and SHA syntax at [evaluation.py:312](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review1/src/mmdc_clip_f/research/view_risk/evaluation.py:312). [evaluation.py:526](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review1/src/mmdc_clip_f/research/view_risk/evaluation.py:526) checks only that the resulting keys are complete. The CLI consumes the free-form selection file at [cli.py:456](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review1/src/mmdc_clip_f/research/view_risk/cli.py:456). It does not bind training results, config/classifier/tune manifest, checkpoint budget, reference selection, guardrail eligibility, or an actual artifact.

The committed test itself constructs every “selected artifact” from formatted integers rather than training/selection evidence at [test_view_risk_evaluation.py:218](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review1/tests/research/test_view_risk_evaluation.py:218).

Consequence: unavailable mandatory controls and a no-eligible-selection result can still become a plan reported as `"frozen"`.

Required fix: freeze verified selection records carrying training-result, classifier, config, tune-manifest, budget, guardrail/reference, and artifact-integrity bindings; reject incomplete/no-selection results.

4. Required study-level training/selection workflows remain disconnected.

All seven analytic controls are included in `MANDATORY_METHODS` but excluded from the fitting route at [training.py:97](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review1/src/mmdc_clip_f/research/view_risk/training.py:97), and the only training CLI rejects them at [cli.py:288](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review1/src/mmdc_clip_f/research/view_risk/cli.py:288). The accepted P3 temperature, DS, scalar, and calibration APIs exist, but P4B does not connect them to its frozen config, caches, artifact bindings, selection records, or evaluation rows.

Similarly, tune selection consumes caller-precomputed aggregate JSON at [cli.py:425](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review1/src/mmdc_clip_f/research/view_risk/cli.py:425); the purported role-safe tune evaluator accepts already-created `TuneCellResult` values rather than producing them from authorized predictions at [evaluation.py:101](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review1/src/mmdc_clip_f/research/view_risk/evaluation.py:101). The command table at [cli.py:47](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review1/src/mmdc_clip_f/research/view_risk/cli.py:47) also has no role-safe fresh-classifier fit/selection path.

Consequence: P4C or P5 would have to implement generic classifier loading, checkpoint evaluation, analytic-control fitting/artifacts, and prediction generation—not merely smoke/cost work or real-data execution.

Required fix: provide executable production adapters with synthetic-injected counterparts clearly separated, and connect every analytic control plus per-epoch tune evaluation into bound artifacts consumed by selection and freeze.

### Verification

- Focused P4B suite: `67 passed`
- Full research suite: `244 passed`
- Focused Ruff: passed
- `git diff --check`: passed
- Independent forward/backward probe for all 16 learned Torch methods: passed
- Independent readiness/freshness probe: failed the required invariant
- Independent pilot completeness/frozen-prediction probe: failed the required invariant
- HEAD and merge-base verified; worktree remained clean
- No real patient data, locked-test data, downloads, or real training accessed
- No patient/secret leakage found in the added evidence

No additional medium or low findings recorded.