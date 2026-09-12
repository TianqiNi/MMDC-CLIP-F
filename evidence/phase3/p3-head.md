Implemented P3A using a fresh GPT-5.6-Sol xhigh agent with no nested subagents. No P2/legacy files, dependencies, training paths, commits, or remotes were touched.

### Acceptance mapping

1. Label-free inputs: canonical `L_CC, L_MLO, R_CC, R_MLO` slots; detached pooled tokens; current scores/probabilities; evidence/vacuity; signed omission differences; validity and agreement metadata. Children are recomputed using accepted fusion and the original frozen logits.
2. Preparation: each head/control owns its projection to 128 dimensions. Scaling is a documented stateless identity transform pending P4.
3. Architecture: two relation-attention layers, 4×32 heads, 128→256→128 FFNs, residuals, layer norms, dropout 0.2, learned same-breast/projection/self biases, masked pooling, and explicit no-removal input.
4. Loss: global error BCE plus parent-normalized raw auxiliary CE. Signed order is `[-1, 0, +1]`; identical predictions report exactly `[0, 1, 0]` without bypassing raw-logit training. Singleton and λ=0 gradient behavior are covered.
5. Ablations: no-omission, no-relation-bias, magnitude, corruption, hidden-only, evidence-only, and combined. Corruption targets enter only the loss.
6. Scientific TDD: all 15 masks, both supported widths and fusion trees, detachment, prediction preservation, canonicalization, absent-slot invariance, gradients, exact constraints, hand-calculated loss, singleton behavior, leakage refusals, and ablation isolation.

### RED/GREEN evidence

- RED: 2 intended collection errors because `head.py` and `head_inputs.py` did not exist.
- Focused GREEN: **25 passed**
- Full suite: **189 passed**
- Ruff: passed
- `git diff --check`: passed

Full record: [p3-head-tdd.md](/home/tianqini/research/.ivr-worktrees/p3-head/evidence/p3-head-tdd.md)

### Parameter counts

| Configuration | Parameters | ViT-B default delta |
|---|---:|---:|
| Default ViT-B/32 | 433,436 | 0 |
| Default ViT-L/14-336 | 466,204 | +32,768 |
| No omission | 432,924 | −512 |
| No relation biases | 433,412 | −24 |
| Magnitude auxiliary | 433,307 | −129 |
| Corruption auxiliary | 433,307 | −129 |
| Hidden-only | 430,492 | −2,944 |
| Evidence-only | 335,004 | −98,432 |

### Public interfaces

```python
prepare_raw_head_inputs(
    features: FrozenViewFeatures, *, backbone: str
) -> RawHeadInputs

ViewRiskInputPreparation(
    hidden_width: int,
    *,
    input_mode="combined",
    hidden_dim=128,
    use_omission_features=True,
)

RelationAwareConfidenceHead(
    RelationAwareHeadConfig(...)
)(raw: RawHeadInputs) -> RelationAwareHeadOutput

compute_view_risk_loss(
    output,
    targets,
    *,
    auxiliary_weight=1.0,
    corruption_targets=None,
) -> ViewRiskLoss

count_trainable_parameters(module: nn.Module) -> int
```

Output shapes include error/confidence `[B]`, effect representations `[B,4,128]`, valid mask `[B,4]`, and auxiliary logits/probabilities `[B,4,3]` for signed or `[B,4,2]` for magnitude/corruption.

Token pooling excludes CLS for ViT-B/32 and includes it for ViT-L/14-336. Global feature order is current scores, probabilities, fused evidence, then fused vacuity.

### Changed files

- [head.py](/home/tianqini/research/.ivr-worktrees/p3-head/src/mmdc_clip_f/research/view_risk/head.py)
- [head_inputs.py](/home/tianqini/research/.ivr-worktrees/p3-head/src/mmdc_clip_f/research/view_risk/head_inputs.py)
- [test_view_risk_head.py](/home/tianqini/research/.ivr-worktrees/p3-head/tests/research/test_view_risk_head.py)
- [test_view_risk_head_inputs.py](/home/tianqini/research/.ivr-worktrees/p3-head/tests/research/test_view_risk_head_inputs.py)
- [p3-head-tdd.md](/home/tianqini/research/.ivr-worktrees/p3-head/evidence/p3-head-tdd.md)

Unresolved by design: fitted scaling remains deferred to P4, software tests do not demonstrate empirical uncertainty improvement, and independent review/integration/publication remain with the orchestrator.