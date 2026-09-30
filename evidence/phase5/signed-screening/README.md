# Signed-supervision screening

This bounded development experiment asks whether signed repair/damage supervision
adds useful failure-ranking information beyond an identical error-only head and
unsigned effect supervision. It reuses the selected epoch-38 classifier and the
existing three seeds, 20 epochs, optimizer, masks and perturbation schedule.

The frozen plan precedes all learned outcomes. Training uses 987 confidence-fit
exams; screening uses the existing 370-exam tune role and its 17 allowed cells.
Every method gets the same cached draws and shared initialization within a seed.
Ground-truth labels generate training targets and evaluation errors; they never
enter the confidence forward path. No pilot or locked test outcomes are read.

The minimum balanced tune AURC checkpoint and fixed epoch 20 are both reported.
This screen cannot qualify the full P5B pilot: mandatory baselines and the
same-seed clean MV-ACN guardrail reference still need to be fitted. Tune gains
are development signals and are optimistic after checkpoint selection.

`run_screening.py` keeps image tensors, private schedules, prepared inputs,
predictions and resumable checkpoints outside Git. It validates role access,
manifest/checkpoint/config/plan bindings, row ordering, target consistency and
exact one-per-exam optimizer exposure. The batched encoder checks full frozen
state bytes before and after each atomic extraction and bounds vision batches.

No model parameters, architecture, sampling weights or training budgets are
tuned during this screen. The runner saves every epoch before evaluating tune
and can resume an interrupted unfinished model without discarding completed
epochs. A failure is reported rather than silently restarted.

The screen is complete. See `results.md`, `aggregate.json`, `summary.json` and
`artifact-review.json`. All nine runs completed in 19.5 minutes; the promising
signal was absent, and the full study is pending a separate research decision.

The generic optimizer's `TrainingResult.software_only` flag keeps these
diagnostic artifacts ineligible for the factory-verified production workflow.
The completed screen used the real, authorized RSNA patient roles; that flag
does not mean its patient inputs were synthetic. Miniature fixtures were used
only for software tests and were discarded during task cleanup.
