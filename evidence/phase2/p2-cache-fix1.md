Implemented all accepted P2C review fixes, uncommitted and ready for independent review.

Acceptance mapping:

- Loaded labels now match authorized manifest densities exactly by dtype, shape, value, and `exam_keys` order.
- Realized perturbation parameters require protocol-resolved fields, reject missing/unknown extras, and enforce crop bounds before encoding.
- Encoder token storage is independently cloned; caller mutation cannot affect extraction.
- Internal token contents are revalidated against the bound token/prompt hash before extraction.
- Direct `.data` model mutations are detected using a checkpoint-derived per-tensor content manifest, not only PyTorch version counters.

TDD results:

```text
RED:             9 failed, 52 passed
Focused GREEN:  61 passed in 0.87s
Research suite: 159 passed in 0.98s
```

Ruff, `py_compile`, and `git diff --check` passed.

Public function signatures remain unchanged. `VerifiedFrozenEncoder.input_ids` is now a defensive-copy property. Every extraction performs a full checkpoint-state content validation before encoding.

Changed files:

- [features.py](/home/tianqini/research/.ivr-worktrees/p2-cache/src/mmdc_clip_f/research/view_risk/features.py)
- [cache.py](/home/tianqini/research/.ivr-worktrees/p2-cache/src/mmdc_clip_f/research/view_risk/cache.py)
- [test_view_risk_features.py](/home/tianqini/research/.ivr-worktrees/p2-cache/tests/research/test_view_risk_features.py)
- [test_view_risk_cache.py](/home/tianqini/research/.ivr-worktrees/p2-cache/tests/research/test_view_risk_cache.py)
- [p2-cache-tdd.md](/home/tianqini/research/.ivr-worktrees/p2-cache/evidence/p2-cache-tdd.md)

Remaining limitations:

- State validation hashes the full model state on CPU for each extraction. Callers should batch a parent’s views to amortize this documented cost.
- Perturbation validation establishes metadata consistency, not authenticity of caller-authored pixels.
- Pretrained parity remains deferred pending verified resources.

No dependencies, legacy files, initializers, private artifacts, training, commits, or pushes were touched.