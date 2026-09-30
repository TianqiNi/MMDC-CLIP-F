# View-removal audit self-review

Reviewed implementation commit: `b806c4aa55746337627d45a7009693b342f35170`.
Reviewer: orchestrator; no independent agent review because the user prohibits
further AI subagents. No actionable Critical, High or Medium findings in this
bounded audit implementation review.

Acceptance checks:

- The committed plan fixes the cohort, checkpoint, cells, realizations and score
  orientations before the real execution. Only confidence-fit role access is
  requested. Empty/mismatched cohorts fail; workers also reject forbidden roles
  before image reads. Pilot/test and held-out families remain excluded.
- Exact existing correctness targets and the original RSNA subset fusion are
  reused. Child scores retain their parent's surviving view inputs. Clean
  preprocessing uses the same reader, resize, conversion and normalization.
- Per-view inference retains the original batch size and architecture. The runner
  checks exact original-forward equivalence and immutable model/checkpoint hashes.
  Private caches contain sufficient logits/labels/row bindings to reproduce the
  aggregate analysis; patient-level artifacts are never committed.
- Counts distinguish removal pairs from unique patients. Wrong-to-wrong remains
  zero effect; soft probability changes remain visible. Constant-correctness
  cohorts have no AUROC. AURC and AP reuse the protocol's tie-aware functions.
- Red/green evidence is in `software-validation.json`. The whole existing suite
  passed, and four new regressions pass. The one new test-fixture failure was
  corrected using distinct image paths, preserving all semantic assertions.
- Family/severity coupling is explicitly disclosed. The 21-cell design measures
  full-parent support, not all training masks or an estimate weighted by the
  training sampler. No numerical go/no-go threshold is invented after results.

Real-data results and scientific interpretation are pending execution. Software
acceptance does not establish sufficient support or a confidence-method benefit.

## Real-data acceptance

The committed implementation ran to completion with exit 0 and empty stderr.
All 987 patients and 21 cells are present. Exact preprocessing/forward/fusion
checks passed and model/checkpoint identity is unchanged. Private cached targets
were separately recounted from predictions and labels; hashes, effects, patient
unions and class totals agree. Both source CI runs passed: `36764155982` and
`36764161141`. The scientific interpretation in `results.md` is deliberately
limited: target support exists but is sparse, raw sensitivity underperforms MSP,
and no learned uncertainty improvement has been demonstrated.
