# P2B perturbation TDD evidence

All checks used synthetic CPU tensors only. No patient data, model weights, fitting,
tuning, or evaluation outcomes were accessed. Section 5 of the accepted protocol
was consistent with the requested configuration; no scientific definition was
changed.

## RED

Tests were written before the module existed.

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research/test_view_risk_perturbations.py
```

Expected outcome: pytest exit 2 with one collection error,
`ModuleNotFoundError: No module named 'mmdc_clip_f.research.view_risk.perturbations'`.
This demonstrated the missing behavior, not a missing dependency.

A strengthened evaluation-panel test was also run before exposing the clean
four-view panel: pytest exit 1 with `1 failed, 20 passed`; the failure was the
missing `EvaluationPanels.clean_four_view` contract. It passed after implementation.

## GREEN

Assigned scientific behavior/refusal suite:

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research/test_view_risk_perturbations.py
```

Outcome: pytest exit 0; `24 passed in 0.57s`.

Full research regression suite after final test strengthening:

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research
```

Outcome: pytest exit 0; `64 passed in 0.65s`.

Style checks:

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m ruff check src/mmdc_clip_f/research/view_risk/perturbations.py tests/research/test_view_risk_perturbations.py
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m ruff format --check src/mmdc_clip_f/research/view_risk/perturbations.py tests/research/test_view_risk_perturbations.py
```

Outcome: exit 0; `All checks passed!`; `2 files already formatted`.
