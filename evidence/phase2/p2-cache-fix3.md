Implemented the accepted review-3 fix, uncommitted and ready for independent review.

- Both resolved final paths and both concrete staging paths are validated before private bytes are written.
- Interrupted tensor and metadata replacements cannot leave unignored private staging files.
- Exact-filename ignore rules that omit staging patterns now refuse.
- Fully external destinations and fully ignored directories remain supported.
- Existing negative destination tests remain intact.

Tests:

```text
RED:             3 failed, 1 passed, 42 deselected
Focused GREEN:   4 passed, 42 deselected
Owned suites:   66 passed
Research suite: 164 passed
```

Ruff, `py_compile`, and `git diff --check` passed.

The callable signature remains unchanged:

```python
save_cache_bundle(
    bundle: CacheBundle,
    metadata_path: str | Path,
) -> CachePaths
```

Compatibility note: ignoring only the final JSON and safetensors filenames may be insufficient. Callers should ignore the whole cache directory or explicitly cover final and staging patterns.

Changed files:

- [cache.py](/home/tianqini/research/.ivr-worktrees/p2-cache/src/mmdc_clip_f/research/view_risk/cache.py)
- [test_view_risk_cache.py](/home/tianqini/research/.ivr-worktrees/p2-cache/tests/research/test_view_risk_cache.py)
- [p2-cache-tdd.md](/home/tianqini/research/.ivr-worktrees/p2-cache/evidence/p2-cache-tdd.md)

No known unresolved implementation issue. No dependencies, training, private artifacts, commits, or pushes were introduced.