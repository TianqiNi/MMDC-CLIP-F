Implemented P2C locally, pending orchestrator integration and independent review.

### Acceptance mapping

- Frozen, eval-only, `no_grad` extraction of observed views only.
- Exposes detached logits, full tokens, projected embeddings, normalized text embeddings, fused scores/probabilities.
- Supports 768/1024 widths and both fusion trees with legacy full-view parity.
- Exact-once legacy normalization from `[0,1]` `RealizedParent` images.
- Omission children reuse parent tensors without re-encoding.
- Verified safetensors checkpoint loading and checkpoint-derived identity.
- Provenance binds ordered private keys, parent identity, roles, manifest, perturbations, model/preprocessing, prompts, tree, versions, and revision.
- Targets are regenerated from logits and labels; stale/tampered targets fail even after hashes are recomputed.
- Role guards precede model/loader/cache-file access; no test-role bypass.
- Persistence is restricted to caller-supplied external or gitignored destinations.
- Unsigned hashes are correctly described as trusted-provenance mismatch detection, not hostile-author authentication.

### TDD evidence

- RED: 2 expected collection failures before `features.py` and `cache.py` existed.
- Focused GREEN: `52 passed in 0.83s`.
- Research regression: `150 passed in 0.95s`.
- Ruff, compilation, and `git diff --check` passed.

Evidence: [p2-cache-tdd.md](/home/tianqini/research/.ivr-worktrees/p2-cache/evidence/p2-cache-tdd.md)

### Callable contracts

```python
load_verified_frozen_encoder(
    classifier: MultiViewCLIPClassifier,
    input_ids: Tensor,
    checkpoint_path: str | Path,
    *,
    backbone: str,
    prompts: Sequence[str],
) -> VerifiedFrozenEncoder

encoder.extract_normalized(
    normalized_views: Mapping[str, Tensor],
) -> FrozenViewFeatures

encoder.extract_realized_parent(
    parent: RealizedParent,
) -> FrozenViewFeatures

encoder.select_parent_views(
    parent: FrozenViewFeatures,
    observed: Iterable[str],
) -> FrozenViewFeatures

encoder.omit_parent_view(
    parent: FrozenViewFeatures,
    removed_view: str,
) -> FrozenViewFeatures

build_exam_cache_with_role_access(
    manifest: RoleManifest,
    *,
    operation: Operation,
    role: Role,
    exam_key: str,
    sample_key: str,
    parent_loader: Callable[[PrivateExamRecord], RealizedParent],
    encoder: VerifiedFrozenEncoder,
    implementation_revision: str,
) -> CacheBundle

save_cache_bundle(
    bundle: CacheBundle,
    metadata_path: str | Path,
) -> CachePaths

load_cache_bundle(
    metadata_path: str | Path,
    *,
    expected_provenance: CacheProvenance,
    manifest: RoleManifest,
    operation: Operation,
    role: Role,
) -> CacheBundle
```

The study artifact producer is deliberately per-exam; lower-level extraction retains explicit batch axes.

Changed files:

- [features.py](/home/tianqini/research/.ivr-worktrees/p2-cache/src/mmdc_clip_f/research/view_risk/features.py)
- [cache.py](/home/tianqini/research/.ivr-worktrees/p2-cache/src/mmdc_clip_f/research/view_risk/cache.py)
- [test_view_risk_features.py](/home/tianqini/research/.ivr-worktrees/p2-cache/tests/research/test_view_risk_features.py)
- [test_view_risk_cache.py](/home/tianqini/research/.ivr-worktrees/p2-cache/tests/research/test_view_risk_cache.py)
- [p2-cache-tdd.md](/home/tianqini/research/.ivr-worktrees/p2-cache/evidence/p2-cache-tdd.md)

No dependency, legacy source, or initializer changes; no commit, push, download, training, patient evaluation, or patient-bearing artifact. No integration defect was found. Actual pretrained parity remains deferred until verified external resources are available.