# Phase 1 integration checks

Assembled accepted protocol ecb80e9 and foundation 6067abb without source edits.

- `PYTHONPATH=src .venv/bin/python -m pytest -q`: 40 passed in 0.62 seconds.
- `.venv/bin/python -m ruff check src tests/research`: passed.
- `.venv/bin/python -m compileall -q src`: passed.
- `.venv/bin/mmdc-clip-f --help`: passed; legacy command surface unchanged.
- `git diff --check`: passed.

Only synthetic CPU validation; no training or patient evaluation.
