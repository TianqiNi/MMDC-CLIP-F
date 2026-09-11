# Phase 3 integration checks

Accepted head `c3e7a67` and baseline `8dc66ca` source/tests assembled without edits.

- `PYTHONPATH=src .venv/bin/python -m pytest -q`: 211 passed in 1.94 seconds.
- `.venv/bin/python -m ruff check src tests/research`: passed.
- `.venv/bin/python -m compileall -q src`: passed.
- `.venv/bin/mmdc-clip-f --help`: passed.
- Source/test/ledger `git diff --check`: passed.

These are synthetic software checks. Candidate scaling, actual exposure matching, real pretrained parity, patient readiness, metrics/trainer integration and measured research remain pending.
