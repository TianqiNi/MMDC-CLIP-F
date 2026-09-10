Implemented both accepted review fixes without changing legacy code or DS arithmetic.

### Findings resolved

- Configurable fusion trees:
  - Added validated `RSNA_FUSION_PAIRS` and `DDSM_FUSION_PAIRS`.
  - Preserves branch, pair, and pruning order.
  - Propagates the selected tree through parent and every omission fusion.
  - Fusion results and targets expose `fusion_pairs` for provenance.
  - Pair conflicts follow the configured tree.
  - Invalid trees fail clearly.
- Numerical safety:
  - Added `InvalidDSFusionError`.
  - Nonfinite/invalid DS intermediates, alpha, statistics, and scores fail before prediction or target `argmax`.
  - No stabilization or formula changes.

### TDD results

Exact RED command:

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research
```

Result: exit 1; `13 failed, 27 passed in 0.61s`.

Exact GREEN command:

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research
```

Result: exit 0; `40 passed in 0.60s`.

Complete suite: `40 passed in 0.61s`. Ruff lint and formatting checks passed.

Changed files:

- [fusion.py](/home/tianqini/research/.ivr-worktrees/p1-foundation/src/mmdc_clip_f/research/view_risk/fusion.py)
- [targets.py](/home/tianqini/research/.ivr-worktrees/p1-foundation/src/mmdc_clip_f/research/view_risk/targets.py)
- [view_risk/__init__.py](/home/tianqini/research/.ivr-worktrees/p1-foundation/src/mmdc_clip_f/research/view_risk/__init__.py)
- [review regression tests](/home/tianqini/research/.ivr-worktrees/p1-foundation/tests/research/test_view_risk_review_regressions.py)
- [TDD evidence](/home/tianqini/research/.ivr-worktrees/p1-foundation/evidence/p1-foundation-tdd.md)

No unresolved issues, commits, dependency changes, training, or patient-data access.