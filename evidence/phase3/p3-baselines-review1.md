Verdict: **REQUEST CHANGES**

Reviewed candidate `7ead473ed9d874480afc23557645ddffe9764012` against base `c3e7a675f51085bd8648d7a2f1a29cd503d7eed8`.

### Findings

1. **High — Current-realization target binding is not enforced.**  
   [_validate_current_cached_targets](/home/tianqini/research/.ivr-worktrees/p3-baselines-review1/src/mmdc_clip_f/research/view_risk/baselines.py:668) verifies a `CachedTargets` object against its own stored scores and the output argmax, but never binds it to the scores, mask, or provenance of the input that produced the output. Consequently, an internally consistent clean target can be used for a stressed input sharing the same argmax. This violates the mandatory “current realized CachedTargets; no stale clean TCP” requirement.

   Synthetic reproduction:

   ```text
   clean scores:    [4.0, 1.0, 0.0, -1.0]
   stressed scores: [1.1, 1.0, 0.0, -1.0]
   same argmax: true
   clean TCP: 0.0463204
   stressed TCP: 0.3833788
   stale clean target accepted: true
   ```

   The test at [test_view_risk_baselines.py:308](/home/tianqini/research/.ivr-worktrees/p3-baselines-review1/tests/research/test_view_risk_baselines.py:308) only corrupts `tcp` while retaining its original scores, so it catches internal inconsistency but not substitution of a wholly consistent stale target. Require a validated `CacheBundle` or carry/check an exact input/provenance binding, and add a same-argmax clean-versus-stressed substitution test.

2. **Medium — Exact original MV-ACN parity fails for the DDSM legacy input order.**  
   The full-view branch always invokes the legacy head in `CANONICAL_VIEWS` order at [baselines.py:638](/home/tianqini/research/.ivr-worktrees/p3-baselines-review1/src/mmdc_clip_f/research/view_risk/baselines.py:638). DDSM’s original configuration uses `L_CC, R_CC, L_MLO, R_MLO` at [ddsm_vit_b_32_legacy.yaml:19](/home/tianqini/research/.ivr-worktrees/p3-baselines-review1/configs/confidence/ddsm_vit_b_32_legacy.yaml:19). The parity test only compares canonical ordering at [test_view_risk_baselines.py:292](/home/tianqini/research/.ivr-worktrees/p3-baselines-review1/tests/research/test_view_risk_baselines.py:292).

   Direct ViT-B synthetic reproduction after copying weights:

   ```text
   bitwise_equal: false
   max_abs_diff: 5.960464477539063e-08
   ```

   Although numerically tiny, the acceptance criterion explicitly requires exact four-view evaluation equivalence. Preserve the configured legacy input order for both datasets and test RSNA/DDSM with both B and L conventions.

### Checks run

- Exact commit/base and clean detached worktree verified.
- Focused P3B/P3A/cache compatibility suite: **88 passed**.
- Full synthetic research suite: **206 passed**.
- Ruff check and format check: passed.
- `git diff --check`: passed.
- Independent stale-target and DDSM exact-parity reproductions above.
- Worktree remained unchanged.

Scalar orientations, role enforcement, DS conflict-slot placement, subset finiteness, ViLU Q/K/V architecture, independent same-input projections/scalers, parameter budgets, and frozen-prediction probability gathering otherwise matched the acceptance contract.

This verdict concerns synthetic software correctness. No training, fitted exposure matching, real-data readiness, patient evaluation, or clinical validity was executed or inferred.