**PASS — G4**, exact candidate **`9984a16b4c3734b93b358f77877380db7def5f35`**.

No material actionable findings. Phase 4 can be published as **software-only accepted**, after orchestrator finalization/publication and release CI.

Verified:

- Source/tests/CI match the independently accepted P4C candidate.
- 32 focused tests passed.
- Independent numerical checks passed: 1,364 tie-metric cases and 2,000 patient-paired bootstrap draws.
- Public CPU smoke reproduced the accepted digest: both fusion trees, all 15 masks, 72 checked inputs and 288 method-level scores.
- Scientific invariants, production configuration, role/provenance guards and test locks remain intact. Prior 269-test full-suite evidence applies to identical source.

Real-data readiness remains separately **blocked**:

- **RSNA:** missing locked-test identity overlap audit, recorded patient-mapping binding, content/duplicate audit and finalized eligible role counts.
- **DDSM:** those blockers plus missing verified exam-to-patient mapping.
- **Both:** public-weight usability, real training resources and fresh selected-classifier provenance remain unverified. Paths and synthetic outputs establish none of these.

This smoke establishes neither efficacy nor real-backbone costs. P5 was not required or run; tests remain locked.

Wrote [validation JSON](/home/tianqini/research/.ivr-p4c-worktrees/p4-gate/evidence/phase4/p4-gate-validation.json). Probe artifacts were cleaned; no other files were changed.