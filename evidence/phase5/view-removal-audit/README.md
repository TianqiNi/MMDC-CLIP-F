# Full-parent view-removal support audit

This audit tests whether the proposed signed correctness-effect target has enough
real development-patient support to justify confidence fitting. It does not train
an auxiliary model, alter the classifier, tune thresholds, or release pilot/test
outcomes. The analysis plan is recorded in `plan.json` before image inference.

Use the frozen epoch-38 RSNA ViT-B/32 classifier and all 987 confidence-fit patients.
For each full-view parent, evaluate its four three-view children with the existing
pruned RSNA fusion tree. Evaluate clean inputs plus Gaussian noise/blur at the
existing mild/moderate parameters, each targeting one of four views or all views.
There are 21 cells and four removals per cell. No held-out family or strong
severity is examined. This is a targeted full-parent diagnostic, not an audit of
all proper-mask training parents or the training sampler's weighted distribution.

Decode each image once. Prepare clean and four stressed variants before ImageNet
normalization, preserving the original deterministic resize, DICOM handling and
batch size of three. Reuse each realized stressed view for its single-target and
common-mode cells; this declared coupling saves inference and keeps the comparison
paired. Common-mode uses the existing shared-field implementation. Every child
retains its parent's surviving per-view scores. No zero-filled missing images are
encoded. Cache per-view logits and row identities only in the private run directory.

The runner validates the selected artifact and role binding, exact clean pixels
against the original preprocessing, exact original-forward equality for the first
batch of all five variants, exact clean fused-score equality for the entire cohort,
and unchanged frozen model/checkpoint bytes. A bounded spawn pool prepares up to
four batches ahead. It uses no AI subagents, schedule, training or optimizer.

Report repair/damage/unchanged counts and unique-patient support, density/view
strata, unchanged-prediction probability changes, and consistently wrong outcomes.
Compare four prespecified failure-ranking scores within each cell: MSP, absolute
removal score sensitivity, maximum probability total variation, and prediction
flip fraction. Raw sensitivity uses a fixed higher-is-riskier orientation; no
orientation or threshold is chosen after seeing results. AURC/AP use the existing
tie-aware metrics. AUROC is unavailable for a one-class correctness cohort.
Summarize cell means separately for clean, single-view and common-mode stresses;
do not treat removals or stress variants as independent patient observations.
These are descriptive development diagnostics, without significance or
publishability claims. Mere evidential uncertainty change after removing evidence
is not evidence that learned failure detection will improve.

Implementation tests cover exact existing effect targets, zero hard effect with
nonzero probability change, wrong-to-wrong cases, patient deduplication, one-class
metrics, family/severity holdouts, exact clean preprocessing, reproducibility,
shared-field semantics, unchanged RNG and forbidden-role rejection before IO.
