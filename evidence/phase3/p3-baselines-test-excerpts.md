# Recorded test commands

Long traces are truncated; synthetic fixtures only.

## p3-baselines-review1

`/bin/bash -lc 'env PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research/test_view_risk_baselines.py tests/research/test_view_risk_head_inputs.py tests/research/test_view_risk_head.py tests/research/test_view_risk_cache.py'`

Exit 0

```text
........................................................................ [ 81%]
................                                                         [100%]
88 passed in 1.75s

```

## p3-baselines-review1

`/bin/bash -lc 'env PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research && /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m ruff check src/mmdc_clip_f/research/view_risk/baselines.py tests/research/test_view_risk_baselines.py && /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m ruff format --check src/mmdc_clip_f/research/view_risk/baselines.py tests/research/test_view_risk_baselines.py'`

Exit 0

```text
........................................................................ [ 34%]
........................................................................ [ 69%]
..............................................................           [100%]
206 passed in 1.87s
All checks passed!
2 files already formatted

```

## p3-baselines-review2

`/bin/bash -lc 'PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research/test_view_risk_baselines.py'`

Exit 0

```text
....................                                                     [100%]
20 passed in 1.14s

```

## p3-baselines-review2

`/bin/bash -lc 'PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research && /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m ruff check src/mmdc_clip_f/research/view_risk/baselines.py tests/research/test_view_risk_baselines.py && /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m ruff format --check src/mmdc_clip_f/research/view_risk/baselines.py tests/research/test_view_risk_baselines.py'`

Exit 0

```text
........................................................................ [ 34%]
........................................................................ [ 68%]
...................................................................      [100%]
211 passed in 2.02s
All checks passed!
2 files already formatted

```
