Resumed the original P3B implementation thread and completed the accepted MV-ACN legacy-order fix. Prior score/mask and manifest-label bindings remain intact.

### Updated API requirement

These constructors now require an explicit validated order:

- `MaskedMVACNAdapter(config, *, legacy_view_order)`
- `MaskedMVACNAdapter.for_backbone(backbone, *, legacy_view_order)`
- `MaskedMVACNAdapter.from_legacy(legacy, *, legacy_view_order)`

P4 must supply:

- RSNA: `("L_CC", "L_MLO", "R_CC", "R_MLO")`
- DDSM: `("L_CC", "R_CC", "L_MLO", "R_MLO")`

Full-view calls preserve this exact order. Missing-view paths filter the configured order, while output masks remain in canonical `[L_CC, L_MLO, R_CC, R_MLO]` slots. Missing, duplicate, and unknown view orders fail closed.

### TDD results

- RED: **5 intended failures**
- Focused order GREEN: **5 passed**
- Complete P3B: **20 passed**
- Full research suite: **211 passed**
- Ruff lint/format and `git diff --check`: passed
- Exact copied-weight parity covers RSNA and DDSM under both ViT-B and ViT-L conventions.
- Parameter counts are unchanged.

### Changed files

- [baselines.py](/home/tianqini/research/.ivr-worktrees/p3-baselines/src/mmdc_clip_f/research/view_risk/baselines.py:581)
- [test_view_risk_baselines.py](/home/tianqini/research/.ivr-worktrees/p3-baselines/tests/research/test_view_risk_baselines.py:315)
- [p3-baselines-tdd.md](/home/tianqini/research/.ivr-worktrees/p3-baselines/evidence/p3-baselines-tdd.md:80)

No scoped unresolved issues remain. No commit, dependency change, training, accepted-source modification, or reviewer-worktree modification occurred.