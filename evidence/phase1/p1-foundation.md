Implemented the intervention-supervised MV-ACN foundation without modifying legacy classifier/confidence behavior, dependencies, or repository history.

### Acceptance coverage

- Canonical named-view `InferenceViewInputs` contract with optional frozen representations and no label channel.
- Legacy-compatible DS subset fusion:
  - all 15 nonempty subsets;
  - canonical pair/pair order;
  - `softplus(z)+1` alpha;
  - `softplus(fused_alpha)` scores;
  - evidence, strength, uncertainty, and conflict statistics;
  - dedicated `NoObservedViewsError`;
  - strict shape, dtype, device, name, and finiteness validation;
  - no computation or gradients for omitted views.
- Exact intervention targets:
  - observed predictions/errors;
  - omission predictions and effects;
  - `delta + 1` encoded labels;
  - canonical valid-removal mask;
  - repair, damage, unchanged, identical-wrong, and singleton behavior;
  - no classifier/image inputs or Stage-1 backpropagation.

### TDD evidence

Exact red command:

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research
```

Expected red: exit 2, 3 collection errors because the research API did not yet exist.

Exact green command:

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research
```

Green: exit 0, `27 passed in 0.59s`.

Complete available suite: `27 passed in 0.61s`. Ruff lint and format checks passed. Full evidence is in [p1-foundation-tdd.md](/home/tianqini/research/.ivr-worktrees/p1-foundation/evidence/p1-foundation-tdd.md).

### Changed files

- [inputs.py](/home/tianqini/research/.ivr-worktrees/p1-foundation/src/mmdc_clip_f/research/view_risk/inputs.py)
- [fusion.py](/home/tianqini/research/.ivr-worktrees/p1-foundation/src/mmdc_clip_f/research/view_risk/fusion.py)
- [targets.py](/home/tianqini/research/.ivr-worktrees/p1-foundation/src/mmdc_clip_f/research/view_risk/targets.py)
- Research package initializers
- Three behavioral test modules under [tests/research](/home/tianqini/research/.ivr-worktrees/p1-foundation/tests/research)
- TDD evidence file above

No unresolved issues. No commit, dependency changes, training, or patient data access performed.
