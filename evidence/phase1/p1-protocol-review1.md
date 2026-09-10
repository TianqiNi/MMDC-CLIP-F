Reviewed commit: `2e9aebb68ea2bd67119d497d9b598f5de4388f48`  
Base: `b791722b78a3ba67ecad36d7635a5a7c80f3420a`  
The reviewed commit is a direct child of the base and changes only `docs/research/*.md`.

Findings:

- **High — Global error head does not consume learned effect representations.** [view_risk_protocol.md:122](/home/tianqini/research/.ivr-worktrees/p1-protocol-review1/docs/research/view_risk_protocol.md:122) defines effect logits and the global error logit as parallel outputs from pooled interaction features. It never creates or feeds learned effect representations into the error head alongside representations/evidence, so effect supervision is only an auxiliary multitask loss rather than the required risk-head input.

- **High — Identical predictions do not deterministically force an unchanged-effect output.** [view_risk_protocol.md:90](/home/tianqini/research/.ivr-worktrees/p1-protocol-review1/docs/research/view_risk_protocol.md:90) constrains the target, but [view_risk_protocol.md:126](/home/tianqini/research/.ivr-worktrees/p1-protocol-review1/docs/research/view_risk_protocol.md:126) merely asks to report inconsistent predictions. The unconstrained effect logits can still output repair or damage when the current and removal predictions are identical. The output contract must deterministically return the unchanged class in that case.

- **Medium — The ledger incorrectly splits the combined P1B into a new P1C task.** [implementation_status.md:19](/home/tianqini/research/.ivr-worktrees/p1-protocol-review1/docs/research/implementation_status.md:19), [implementation_status.md:77](/home/tianqini/research/.ivr-worktrees/p1-protocol-review1/docs/research/implementation_status.md:77), and [implementation_status.md:100](/home/tianqini/research/.ivr-worktrees/p1-protocol-review1/docs/research/implementation_status.md:100) separate subset fusion and intervention targets into P1B/P1C. The required orchestration assigns both to one P1B task and the same implementer.

- **Medium — The phase gate can block generic software development on unavailable patient metadata.** [implementation_status.md:101](/home/tianqini/research/.ivr-worktrees/p1-protocol-review1/docs/research/implementation_status.md:101), combined with the strict next-phase rule at line 39, requires verified patient grouping before P3 can begin. This contradicts the allowance at line 91 for synthetic-fixture implementation: missing DDSM patient mappings should block real-data studies, not generic P3/P4 software work.

- **Medium — The documents incorrectly describe the full study as unauthorized.** [implementation_status.md:36](/home/tianqini/research/.ivr-worktrees/p1-protocol-review1/docs/research/implementation_status.md:36) says the later locked study is “not yet authorized,” and [view_risk_protocol.md:401](/home/tianqini/research/.ivr-worktrees/p1-protocol-review1/docs/research/view_risk_protocol.md:401) requires separate authorization. The full study is already authorized conditionally on a successful pilot and available time/resources.

Verdict: **Request changes / not accepted.**

No tests were run; this was a documentation-only review.