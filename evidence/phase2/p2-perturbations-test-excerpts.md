# Recorded test commands

Long traces are truncated; synthetic fixtures only.

## p2-perturbations-review1

`/bin/bash -lc 'PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research'`

Exit 0

```text
................................................................         [100%]
64 passed in 0.66s

```

## p2-perturbations-review1

`/bin/bash -lc 'PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research/test_view_risk_perturbations.py'`

Exit 0

```text
........................                                                 [100%]
24 passed in 0.55s

```

## p2-perturbations-review2

`/bin/bash -lc "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q -p no:cacheprovider tests/research/test_view_risk_perturbations.py -k 'training_replay or training_realization'"`

Exit 0

```text
..............                                                           [100%]
14 passed, 24 deselected in 0.56s

```

## p2-perturbations-review2

`/bin/bash -lc 'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q -p no:cacheprovider tests/research'`

Exit 0

```text
........................................................................ [ 92%]
......                                                                   [100%]
78 passed in 0.66s

```

## p2-perturbations-review2

`/bin/bash -lc 'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q -p no:cacheprovider tests/research/test_view_risk_perturbations.py'`

Exit 0

```text
......................................                                   [100%]
38 passed in 0.58s

```
