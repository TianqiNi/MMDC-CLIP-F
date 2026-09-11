Fixed the accepted review defect without changing perturbation numerics or existing valid behavior.

Fix mapping:

- `TrainingSampleSpec.from_dict()` now rejects held-out contrast, strong Gaussian noise, malformed strata/modes, and inconsistent stressed-view combinations.
- Directly constructed draws can no longer bypass restrictions: `realize_training_parent()` invokes the shared fit/tune validator.
- `sample_training_spec()` validates generated draws through the same path.
- Valid clean and allowed Gaussian replay remains deterministic for both operations.

Interface changes:

```python
TrainingSampleSpec.from_dict(
    value,
    *,
    operation: str = "confidence-fit",
) -> TrainingSampleSpec

validate_training_sample_spec(
    draw: TrainingSampleSpec,
    *,
    operation: str,
) -> None

realize_training_parent(
    images,
    draw,
    *,
    operation: str = "confidence-fit",
) -> RealizedParent
```

Existing callers remain compatible through the default operation.

Exact TDD checks:

```bash
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python \
  -m pytest -q tests/research/test_view_risk_perturbations.py \
  -k 'training_replay or training_realization'
```

- RED: `12 failed, 24 deselected in 0.61s`
- GREEN: `14 passed, 24 deselected in 0.53s`

```bash
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python \
  -m pytest -q tests/research/test_view_risk_perturbations.py
```

- Complete P2B suite: `38 passed in 0.57s`

```bash
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python \
  -m pytest -q tests/research
```

- Full research suite: `78 passed in 0.63s`
- Ruff lint, format check, and Python compilation also passed.

Changed only:

- [perturbations.py](/home/tianqini/research/.ivr-worktrees/p2-perturbations/src/mmdc_clip_f/research/view_risk/perturbations.py)
- [test_view_risk_perturbations.py](/home/tianqini/research/.ivr-worktrees/p2-perturbations/tests/research/test_view_risk_perturbations.py)
- [p2-perturbations-tdd.md](/home/tianqini/research/.ivr-worktrees/p2-perturbations/evidence/p2-perturbations-tdd.md)

No unresolved implementation issues. No commits, dependency changes, patient access, or training were performed.
