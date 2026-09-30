# P5A classifier selection

The classifier uses the original fixed 50-epoch schedule. Selection compares
all 50 checkpoints on the same 370 tune patients using multiclass NLL; an exact
tie selects the earlier epoch. Pilot and locked test outcomes are excluded.

The optional selector flags `--image-workers 12 --prefetch-batches 2
--tune-cache-mib 1024` reuse the deterministic evaluation tensors in bounded RAM.
They preserve batch size, patient order, transforms, logits and selection rule.
The budget covers image and label tensors (about 850 MiB for this cohort), not
the model, worker processes or Python metadata. Zero cache budget retains the
original streaming behavior. Cached tensors are scoped to one authorized tune
operation and are never written to a shared data cache.

`software-validation.json` records red/green regression evidence. The regression
compares every candidate NLL exactly, verifies one-time decoding, and rejects an
insufficient memory budget or empty authorized tune cohort before image reads.
The existing end-to-end selection test exercises multiprocessing and caching.

`validate_fit.py` verifies fit provenance, all checkpoint bytes/states, finite
parameters, final resume/model equality, optimizer state and the original
schedule. `tune_metrics.py` validates the selected artifact and computes aggregate
tune diagnostics, checking that the selected NLL reproduces selection. Private
manifests, patient predictions and model weights remain outside this repository.

Tune metrics are development diagnostics after checkpoint selection; they are
not unbiased held-out estimates and do not establish an uncertainty-method
improvement. That requires the subsequent matched confidence/control pilot.
No AI subagents or scheduled monitoring are used for this task.
