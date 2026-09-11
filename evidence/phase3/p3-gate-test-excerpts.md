# Recorded test commands

Long traces are truncated; synthetic fixtures only.

## p3-gate

`/bin/bash -lc "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q -p no:cacheprovider tests/research/test_view_risk_head.py tests/research/test_view_risk_baselines.py -k 'singleton_corruption or mixed_masks or parent_normalized or lambda_zero or identical_predictions or stale_same_argmax or refuses_labels or exact_for_each_legacy_order or filters_configured_order or vilu_attention or conflicts_map or scalar_signs' && git diff --check b19fb9c5b411fb63ce165c9eb88eab8cfba6e219 HEAD -- src tests docs && git status --short && git rev-parse HEAD"`

Exit 0

```text
................                                                         [100%]
16 passed, 18 deselected in 0.80s
07df30ac114056f5f4d6ceaea8af6933ff7acd8c

```
