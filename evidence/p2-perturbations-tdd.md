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

## Review-1 regression fix

Accepted finding: deserialized or directly constructed training draws could
bypass fit/tune restrictions and malformed stratum/mode/view combinations were
not validated.

Focused RED command:

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research/test_view_risk_perturbations.py -k 'training_replay or training_realization'
```

Outcome before the fix: pytest exit 1; `12 failed, 24 deselected in 0.61s`.
Failures showed the missing operation arguments/guards and acceptance of every
malformed replay fixture.

Focused GREEN used the same command. Outcome: pytest exit 0;
`14 passed, 24 deselected in 0.53s`. Valid clean and allowed Gaussian replay
fixtures remained deterministic under both confidence-fit and tune guards.

Final full-suite GREEN used the full research-suite command above. Outcome:
pytest exit 0; `78 passed in 0.63s`. Ruff lint, Ruff format check, and Python
compilation also passed.
