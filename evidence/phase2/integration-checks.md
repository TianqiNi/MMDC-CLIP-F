# Phase 2 integration checks

Accepted role task `85c8d30`, perturbation task `1ee5555`, and cache task `b176bb5` assembled without source edits.

- `PYTHONPATH=src .venv/bin/python -m pytest -q`: 164 passed in 1.15 seconds.
- `.venv/bin/python -m ruff check src tests/research`: passed.
- `.venv/bin/python -m compileall -q src`: passed.
- `.venv/bin/mmdc-clip-f --help`: passed.
- `git diff --check`: passed.

Synthetic software checks only. Real pretrained parity, patient mapping and overlap/content audits, and fresh classifier training remain unexecuted. Per-extraction full checkpoint-state hashing cost remains unmeasured with real weights.
