# P1B foundation TDD evidence

All checks used synthetic tensors on CPU. No patient data or training was used. The dependency
readiness marker was present before the red run.

## Red

Command:

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research
```

Outcome: expected failure (pytest exit 2), with 3 collection errors because the new
`mmdc_clip_f.research.view_risk` interfaces did not exist. This was the intended behavioral red,
not a dependency import failure.

## Green

Command:

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research
```

Outcome: pytest exit 0; `27 passed in 0.59s`.

Style command:

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m ruff check src/mmdc_clip_f/research tests/research
```

Outcome: exit 0; `All checks passed!`.
