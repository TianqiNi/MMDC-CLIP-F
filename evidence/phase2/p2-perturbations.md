Implemented P2B with no protocol conflict found.

### Acceptance mapping

- Exact Section 5 Gaussian noise/blur, contrast, bidirectional brightness, square crop with antialiased bilinear resize, and horizontal/vertical motion blur.
- Correct 224/336 scaling, reflect padding, clipping, normalized kernels, and one spatial Gaussian field shared across RGB channels.
- Immutable, JSON-serializable perturbation specifications with SHA256-derived replay seeds; private keys are not retained or exposed as features.
- Named, nonempty view mappings; absent views remain absent.
- Omission children preserve the exact surviving tensor objects without rerandomization.
- Common-mode perturbations reuse identical stochastic realizations across views.
- Four equal training strata, all 14 proper masks, mild/moderate Gaussian-only fit/tune support, and exact 80/20 single/common weighting.
- Held-out families and strong severity are explicitly rejected during fitting/tuning.
- Evaluation panels expose 48 primary cells, nested brightness/motion variants, 14 clean masks, clean four-view, 8 strong-seen cells, and 18 common-mode cells.

### Verification

- Initial RED: collection failed because the perturbations module did not exist.
- Strengthened RED: `1 failed, 20 passed` for the missing clean-four-view panel contract.
- Assigned GREEN: `24 passed in 0.57s`.
- Full research suite: `64 passed in 0.65s`.
- Ruff lint and formatting: passed.
- Python compilation: passed.

Commands and evidence are recorded in [p2-perturbations-tdd.md](/home/tianqini/research/.ivr-worktrees/p2-perturbations/evidence/p2-perturbations-tdd.md).

### Changed files

- [perturbations.py](/home/tianqini/research/.ivr-worktrees/p2-perturbations/src/mmdc_clip_f/research/view_risk/perturbations.py)
- [test_view_risk_perturbations.py](/home/tianqini/research/.ivr-worktrees/p2-perturbations/tests/research/test_view_risk_perturbations.py)
- [p2-perturbations-tdd.md](/home/tianqini/research/.ivr-worktrees/p2-perturbations/evidence/p2-perturbations-tdd.md)

Only these three new files are present in the worktree status.

### Callable interfaces

```python
PerturbationSpec(
    family: str,
    severity: str | None = None,
    variant: str | None = None,
    realization_seed: int = 0,
)

PerturbationSpec.for_sample(
    family, severity, *,
    private_sample_key, cell, seed, variant=None,
) -> PerturbationSpec

PerturbationSpec.to_dict() -> dict
PerturbationSpec.from_dict(value) -> PerturbationSpec

derive_realization_seed(private_sample_key, cell, seed) -> int
resolve_parameters(spec, resolution) -> dict
apply_perturbation(image, spec) -> tuple[Tensor, PerturbationMetadata]

realize_parent(
    images, observed, *,
    perturbations=None,
    common_mode=None,
) -> RealizedParent

omit_parent_view(parent, removed_view) -> RealizedParent
enumerate_proper_masks() -> tuple[tuple[str, ...], ...]
enumerate_training_support() -> tuple[TrainingSupportCase, ...]
validate_training_perturbation(spec, *, operation) -> None

sample_training_spec(
    private_sample_key, cell, seed, *,
    operation="confidence-fit",
) -> TrainingSampleSpec

realize_training_parent(images, draw) -> RealizedParent
enumerate_evaluation_panels() -> EvaluationPanels
```

No P2B implementation issues remain. P2A/P2C binding, integration, and independent review remain for the orchestrator. No dependencies, initializers, training jobs, commits, or pushes were touched.
