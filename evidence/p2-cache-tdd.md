# P2C frozen cache TDD evidence

Scope: synthetic software behavior only. Dependency candidate `c689eb9` was inspected; current accepted P1 fusion/targets and reviewed P2A/P2B role/perturbation APIs were consumed without modification. No model download, training, patient evaluation, or empirical result was produced.

## RED

Tests were written before `features.py` and `cache.py` existed.

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research/test_view_risk_features.py tests/research/test_view_risk_cache.py
```

Result: collection failed with 2 expected `ModuleNotFoundError` errors for `mmdc_clip_f.research.view_risk.features` and `.cache`.

## GREEN

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research/test_view_risk_features.py tests/research/test_view_risk_cache.py
```

Result: `52 passed in 0.83s`.

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research
```

Result: `150 passed in 0.95s`.

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m ruff check src/mmdc_clip_f/research/view_risk/features.py src/mmdc_clip_f/research/view_risk/cache.py tests/research/test_view_risk_features.py tests/research/test_view_risk_cache.py
```

Result: all checks passed. `git diff --check` also passed.

Behavioral coverage includes observed-only encoding; 768/1024 full-token widths; both configured fusion trees and full-view legacy prediction parity; detached/frozen features and mutation refusal; exact-once normalization from a `RealizedParent`; omission reuse without re-encoding; strict tensor-only checkpoint loading/hash binding; role denial before parent/model or cache-file access; the explicit confidence-fit/tune perturbation-operation adapter; exact regenerated targets/TCP; expected provenance field mismatches; mask/tree/version/key mismatch; metadata/tensor/missing/nonfinite tamper; and stale targets after recomputing every unsigned file/tensor/record hash.

The supported study builder is deliberately per exam because accepted `RealizedParent` is per exam. Lower-level normalized extraction and cache validation retain explicit batch axes; no multi-parent artifact producer or precomputed subset-token combinations were added. Unsigned hashes detect stale/mismatched data under trusted expected provenance and do not authenticate a hostile author. Actual pretrained parity remains deferred until verified external resources are available.
