Reviewed commit `6c7c733aaa481ca3d31bd84617b7f34f367a8977` against base `b19fb9c5b411fb63ce165c9eb88eab8cfba6e219`; the candidate’s sole parent is the specified base.

## Findings

1. **Medium — corruption-mask coverage does not distinguish observed views from removable-effect slots.**  
   [test_view_risk_head.py:148](/home/tianqini/research/.ivr-worktrees/p3-head-review1/tests/research/test_view_risk_head.py:148) uses a two-view parent, for which `observed_mask == removal_valid_mask`. Consequently, the assertion at line 164 would still pass if corruption incorrectly supervised only removable slots. Add a singleton corruption case asserting that its sole observed view receives auxiliary loss and gradients despite having no valid removal. The current implementation at [head.py:212](/home/tianqini/research/.ivr-worktrees/p3-head-review1/src/mmdc_clip_f/research/view_risk/head.py:212) behaves correctly, but the mandatory regression would not catch this specific failure.

2. **Medium — the parent-normalization test cannot distinguish per-parent averaging from global removal averaging.**  
   [test_view_risk_head.py:171](/home/tianqini/research/.ivr-worktrees/p3-head-review1/tests/research/test_view_risk_head.py:171) gives both parents exactly two valid removals. Thus, mean-of-parent-means equals a single mean across all valid slots, and a scientifically incorrect global-normalization implementation would pass. Add parents with unequal valid counts, preferably including a singleton, and hand-calculate the mean of their individual auxiliary means. The current implementation at [head.py:345](/home/tianqini/research/.ivr-worktrees/p3-head-review1/src/mmdc_clip_f/research/view_risk/head.py:345) passed my independent unequal-mask probe.

## Checks run

- Targeted P3 and regression suites: `38 passed`.
- Full synthetic suite: `189 passed`.
- Independent mixed-mask probe:
  - Batch/single output equivalence passed.
  - Unequal-count parent-normalized CE passed.
  - λ=0 effect-to-risk gradient passed.
  - Singleton corruption supervision over observed slots passed.
- Ruff: passed.
- `git diff --check`: passed.
- Worktree remained clean.
- Static inspection confirmed label-free detached inputs, accepted fusion reuse, exact classifier-prediction preservation, absent-slot masking, deterministic signed/magnitude reporting, corruption-label isolation, identity scaling, and one-factor ablation paths.
- The tensor-level head/loss contract handles mixed parent masks once prepared records are collated; no unrelated framework additions appear necessary for P4.

**Verdict: REQUEST CHANGES.**

No current synthetic implementation failure was reproduced, but the two mandatory scientific invariants above lack tests capable of detecting their corresponding regressions. No real-data training, patient outcomes, pretrained weights, or empirical uncertainty improvement were evaluated; real-data readiness remains unexecuted, with fitted scaling appropriately deferred to P4.