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

## Accepted-review fix cycle

Regression tests were added before source edits for configured RSNA/DDSM fusion trees and
nonfinite legacy DS states.

### Red

Command:

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research
```

Outcome: expected pytest exit 1; `13 failed, 27 passed in 0.61s`. Failures showed the absent
`fusion_pairs` API/provenance, absent invalid-DS error, and therefore missing configured-tree
support and target propagation.

### Green

Command:

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research
```

Outcome: pytest exit 0; `40 passed in 0.60s`.

The complete available suite also passed: `40 passed in 0.61s`. Ruff lint returned
`All checks passed!`; Ruff format check reported `9 files already formatted`.
