Reviewed exact commit `76cc677e38ccddaaaeb264e6e46f8f9e0e109cf3` against parent/base `b791722b78a3ba67ecad36d7635a5a7c80f3420a`.

## Findings

1. **[P1] The fusion tree is hardcoded to RSNA, violating the DDSM legacy contract and corrupting DDSM omission targets.**
   [fusion.py:86](/home/tianqini/research/.ivr-worktrees/p1-foundation-review1/src/mmdc_clip_f/research/view_risk/fusion.py:86) always uses `(L_CC,L_MLO)/(R_CC,R_MLO)`, while the repository explicitly configures DDSM as `(L_CC,R_CC)/(L_MLO,R_MLO)` in [ddsm_vit_b_32_legacy.yaml:26](/home/tianqini/research/.ivr-worktrees/p1-foundation-review1/configs/stage1/ddsm_vit_b_32_legacy.yaml:26). The legacy classifier accepts this configuration at [model.py:24](/home/tianqini/research/.ivr-worktrees/p1-foundation-review1/src/mmdc_clip_f/model.py:24).
   This is not harmless associativity: a fixed float32 synthetic example produced RSNA scores `[15169263, 15169263,…]` predicting class 0 versus DDSM `[15169256, 15169258,…]` predicting class 1. Pair-conflict features were respectively `[0.2071, 0.5154, 0.4962]` and `[0.4695, 0.1944, 0.5470]`.
   Moreover, [targets.py:92](/home/tianqini/research/.ivr-worktrees/p1-foundation-review1/src/mmdc_clip_f/research/view_risk/targets.py:92) invokes the same hardcoded fusion for every omission. On that example, two of four three-view omission predictions differed between trees, changing signed effects. Both APIs need the validated chosen tree propagated through the observed fusion and every omission.

2. **[P1] Finite extreme logits can silently generate NaN scores and fabricated targets.**
   Inputs are checked for finiteness, but DS arithmetic at [fusion.py:42](/home/tianqini/research/.ivr-worktrees/p1-foundation-review1/src/mmdc_clip_f/research/view_risk/fusion.py:42) can overflow, and [fusion.py:109](/home/tianqini/research/.ivr-worktrees/p1-foundation-review1/src/mmdc_clip_f/research/view_risk/fusion.py:109) returns without validating results. Four finite float32 tensors filled with `1e38` produced all-NaN `fused_alpha` and scores. Target generation then silently returned prediction 0 through `argmax`; with label 3 it emitted error 1 as if valid. The research API must fail clearly when legacy arithmetic yields nonfinite intermediates/outputs. This requires a fail-fast check, not an unreviewed formula or stabilization change.

## Checks

- `PYTHONPATH=src …/python -m pytest -q tests/research` — `27 passed`
- `PYTHONPATH=src …/python -m pytest -q` — `27 passed`
- Ruff on changed source/tests — passed
- `git diff --check` — passed
- Synthetic tree, omission, conflict, and extreme-logit reproductions — exposed both findings
- Worktree remained clean; no tracked files edited

The ordinary-range tests substantiate all 15 nonempty subsets, legacy `softplus(fused_alpha)` scoring for RSNA, signed omission semantics, singleton/empty handling, inference/label separation, and absent-view gradients.

**Verdict: reject for scientific acceptance pending both P1 fixes.**
