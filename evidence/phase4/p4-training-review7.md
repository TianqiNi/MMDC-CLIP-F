# P4B review — NOT ACCEPTED

Reviewed exact commit: **`1e3fc67534b5f606bc1ce2c4910c50a3faa3f043`**

Parent: `b91cc5c9404445d8c644136f404ed7e5d07de6c9`  
Accepted base: `8562a4b0b6fe5c3baf0e463c9ab213e00aae1dcd`

## Finding

1. **Medium — production configuration accepts non-protocol optimizer settings**

   Locations: [training.py:145](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review7/src/mmdc_clip_f/research/view_risk/training.py:145), [training.py:233](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review7/src/mmdc_clip_f/research/view_risk/training.py:233), [training.py:1706](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review7/src/mmdc_clip_f/research/view_risk/training.py:1706)

   `OptimizerConfig` checks only that learning rate is positive, weight decay nonnegative, and batch size positive. `ResearchRunConfig` does not require the frozen protocol values.

   Concrete reproduction:

   ```text
   ResearchRunConfig.from_dict({
       "dataset": "RSNA",
       "backbone": "vit_b_32",
       "optimizer": {
           "name": "adam",
           "learning_rate": 0.5,
           "weight_decay": 9.0,
           "batch_size": 1
       }
   })

   OptimizerConfig(name='adam', learning_rate=0.5, weight_decay=9.0, batch_size=1)
   ```

   These values flow directly into `torch.optim.Adam` and batch enforcement. The resulting configuration can be used by the public classifier/confidence fitting and pilot-plan paths despite violating the required Adam `lr=1e-4`, weight decay `0`, batch size `6` contract.

   Impact: a materially different experiment can complete, persist selections, and be represented as the protocol workflow. Defaults are correct, but prohibited configuration changes are not rejected.

   Minimal fix: enforce `config.optimizer == OptimizerConfig()` in `ResearchRunConfig.__post_init__` or an equivalent production-config validator. Add public config-loading tests rejecting changes to each frozen optimizer field while retaining flexible optimizer values only in lower-level synthetic tests.

## Review6 dispositions

- **Exact token/prompt/configuration identity:** fixed. `FrozenEncoderIdentity` is re-derived from the current verified initialization token bytes and selected checkpoint bytes. It is carried through learned fits, tune selections, controls, pilot plans, evaluation, and scoring. Independent probes rejected changed token IDs, prompt order, preprocessing, and fusion.
- **Prescribed tune/control realizations:** fixed. Learned selection requires clean plus all 16 full-view single-target noise/blur cells. TS and scalar calibration require clean full-view tune inputs. DS requires the deterministic epoch/exam confidence-fit schedule and the eligible tune regularization panel. Wrong mask, seed, realized parameters, stressed TS input, or unmatched DS draw is rejected.

Analytic controls record actual fitting semantics: raw controls have zero trials/updates, TS/scalar calibration record one fit, and DS records its regularization trials. No fabricated optimizer epochs were found.

## Earlier four-finding dispositions

- Synthetic provenance promotion: fixed.
- Conflicting authoritative prediction registries: fixed.
- Incomplete learned/control artifact verification: fixed, including current cache/checkpoint bytes and encoder identity on reload.
- Disconnected production workflows: fixed; the public tiny-model workflow exercised classifier fitting/selection, learned fitting, all seven analytic dispatches, reload, and label-free scoring.

Changes in the earlier P2/P3 modules do not alter accepted fusion/head/metric numerics: the feature changes add identity binding, baseline changes authorize repeated deterministic DS exposures, and the perturbation change centralizes replay of the existing seeded crop calculation.

## Validation

- Targeted public tiny-model integration plus exact resume test: **2 passed**.
- Independent identity/realization probe: altered tokens, prompts, preprocessing, fusion, and realization seed all rejected; 17-cell tune contract confirmed.
- Full research suite: **258 passed**.
- Ruff over `src` and `tests/research`: **passed**.
- Compileall over `src` and `tests/research`: **passed**.
- Source/test latest-fix `git diff --check`: **passed**.
- Full latest-fix diff has one nonfunctional trailing space in `evidence/phase4/p4-training-review6.md:5`.
- Worktree remained clean; no dependency changes.

The default pytest temporary directory was initially unusable because this environment contains a read-only `/tmp/.git`; rerunning with an ignored workspace basetemp passed. No approval request was rejected.

## Limitations

No real medical data, model downloads, real training, pilot outcomes, or locked-test access were used. External public model loading was mocked in the tiny integration. P4C, G4, and P5 were not assessed.