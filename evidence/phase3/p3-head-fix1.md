Addressed both accepted review findings using a fresh GPT-5.6-Sol xhigh agent. No source or API changes were needed.

### Acceptance mapping

- Singleton corruption regression proves:
  - no valid removals;
  - the sole observed view still receives auxiliary supervision;
  - equality with hand-calculated CE;
  - nonzero raw-logit and effect-representation gradients only for that view.
- Mixed-mask regression uses valid-removal counts `[3, 2, 0]`, independently calculates each parent’s mean CE—including singleton zero—and then averages parents equally.

### Mutation-based RED

These were deliberate coverage mutations, not preexisting implementation failures:

- Observed corruption mask → removable mask: **1 failed**
- Per-parent averaging → global valid-slot averaging: **1 failed**
  - mutated result: `0.3908577`
  - expected parent-normalized result: `0.2435988`

The accepted `head.py` was restored byte-for-byte after each mutation; SHA-256 matched `46bd7ff775d99204e43d95170b507fceac788dd97a2413c4f1601a28532693bf`.

### GREEN

- New regressions: **2 passed**
- Focused P3A suites: **27 passed**
- Full synthetic suite: **191 passed**
- Ruff: passed
- `git diff --check`: passed

### Changed this pass

- [test_view_risk_head.py](/home/tianqini/research/.ivr-worktrees/p3-head/tests/research/test_view_risk_head.py)
- [p3-head-tdd.md](/home/tianqini/research/.ivr-worktrees/p3-head/evidence/p3-head-tdd.md)

Callable interfaces remain unchanged:

```python
RelationAwareConfidenceHead(config)(raw) -> RelationAwareHeadOutput

compute_view_risk_loss(
    output,
    targets,
    *,
    auxiliary_weight=1.0,
    corruption_targets=None,
) -> ViewRiskLoss
```

No unresolved implementation defect was found. No commit, push, dependency change, training, or source modification was performed.