# Recorded test commands

Long traces are truncated; synthetic fixtures only.

## p2-cache-review1

`/bin/bash -lc 'PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research/test_view_risk_features.py tests/research/test_view_risk_cache.py'`

Exit 0

```text
....................................................                     [100%]
52 passed in 0.84s

```

## p2-cache-review1

`/bin/bash -lc 'PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research && PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m ruff check src/mmdc_clip_f/research/view_risk/features.py src/mmdc_clip_f/research/view_risk/cache.py tests/research/test_view_risk_features.py tests/research/test_view_risk_cache.py && PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m py_compile src/mmdc_clip_f/research/view_risk/features.py src/mmdc_clip_f/research/view_risk/cache.py'`

Exit 0

```text
........................................................................ [ 48%]
........................................................................ [ 96%]
......                                                                   [100%]
150 passed in 0.98s
All checks passed!

```

## p2-cache-review2

`/bin/bash -lc 'PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research/test_view_risk_features.py tests/research/test_view_risk_cache.py'`

Exit 0

```text
.............................................................            [100%]
61 passed in 0.92s

```

## p2-cache-review2

`/bin/bash -lc 'PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research'`

Exit 0

```text
........................................................................ [ 45%]
........................................................................ [ 90%]
...............                                                          [100%]
159 passed in 1.02s

```

## p2-cache-review3

`/bin/bash -lc "env PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research/test_view_risk_cache.py -k 'ignored_metadata_with_unignored_tensor_sibling or fully_ignored_cache_siblings'"`

Exit 0

```text
..                                                                       [100%]
2 passed, 41 deselected in 0.60s

```

## p2-cache-review3

`/bin/bash -lc 'env PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research'`

Exit 0

```text
........................................................................ [ 44%]
........................................................................ [ 89%]
.................                                                        [100%]
161 passed in 1.07s

```

## p2-cache-review4

`/bin/bash -lc "PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research/test_view_risk_cache.py -k 'exact_filename_ignore_rules or fully_ignored_directory or atomic_replacement_cannot_strand'"`

Exit 0

```text
....                                                                     [100%]
4 passed, 42 deselected in 0.70s

```

## p2-cache-review4

`/bin/bash -lc 'PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research'`

Exit 0

```text
........................................................................ [ 43%]
........................................................................ [ 87%]
....................                                                     [100%]
164 passed in 1.18s

```
