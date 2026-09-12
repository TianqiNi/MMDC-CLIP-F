Fixed the review-2 finding, uncommitted and ready for independent review.

- Both final `.json` and `.safetensors` destinations are resolved and validated before either write occurs.
- Ignoring only `artifact.json` now refuses with neither artifact created.
- Fully ignored and fully external destinations remain supported.
- `save_cache_bundle(...)` signature is unchanged; returned paths are now canonical absolute paths.

Tests:

```text
RED:             1 failed, 1 passed, 41 deselected
Targeted GREEN:  2 passed, 41 deselected
Owned suites:   63 passed
Research suite: 161 passed
```

Ruff, `py_compile`, and `git diff --check` passed.

Changed files:

- [cache.py](/home/tianqini/research/.ivr-worktrees/p2-cache/src/mmdc_clip_f/research/view_risk/cache.py)
- [test_view_risk_cache.py](/home/tianqini/research/.ivr-worktrees/p2-cache/tests/research/test_view_risk_cache.py)
- [p2-cache-tdd.md](/home/tianqini/research/.ivr-worktrees/p2-cache/evidence/p2-cache-tdd.md)

No unresolved issue for this finding. No dependencies, legacy code, private artifacts, training, commits, or pushes were touched.