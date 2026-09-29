# Classifier image pipeline acceleration

The optional classifier CLI flags `--image-workers 12 --prefetch-batches 2`
parallelize CPU image preparation on the local Ryzen 9 7950X / RTX 4090 host.
The default remains `--image-workers 0`, the original serial implementation.
These are execution options; they do not change the scientific run configuration.

The worker pool persists across epochs and uses `spawn` so workers never inherit
the trainer's CUDA context. Each worker uses one Torch CPU thread. Only authorized
role records are submitted. Each image still passes the existing path and content
hash checks, DICOM decoding, normalization, full-resolution RandAugment, resize,
tensor conversion and normalization. There is no image conversion cache, change
to augmentation order, or skipped integrity check. Two batches are prefetched;
only resized tensors return to the parent. CUDA batches use pinned memory and
nonblocking transfers on the original stream.

## Reproducibility

Ordinary independently seeded workers would change the augmentations when
resuming a serial run. Instead, this loader reserves the existing Torch CPU RNG
sequence in view-major/sample-minor order. The supported torchvision 0.18
RandAugment draws depend on operation choice/sign, not image pixels/dimensions.
Running it on an 8-by-8 RGB placeholder reserves these draws; workers replay the
same torchvision transform on the original image, with the reserved RNG state.
Every worker checks the resulting RNG state against the reservation.

Prefetching uses private RNG copies. The parent advances its RNG only when a batch
is delivered. If a stochastic CPU consumer changes the RNG between batches, the
loader discards outstanding speculative work and regenerates it from that state.
CUDA RNG is untouched. Workers are bounded and shut down on completion/error.
Unrecognized future transforms fail closed instead of silently changing draws.

The architecture, original batch size 3, Adam settings, seed, full precision,
prompts, fusion and 50-epoch schedule are unchanged. Tests compare exact tensors,
labels, order, CPU RNG, Adam updates and switching loaders at an epoch boundary.
The production CLI integration test also exercises a persistent worker pool.

## Measurements and limits

See [aggregate benchmark evidence](../../evidence/phase5/hardware-pipeline/).
The loader benchmark uses 60 classifier-fit patients (240 mammograms), with no
tune, pilot or test exposure. Steady batch preparation times were 4.607 seconds
serial, 1.260 with 4 workers, 0.671 with 8, and 0.541 with 12. All four runs had
identical tensors, labels and final CPU RNG. Twelve workers gave an 8.5-fold
preprocessing speedup; startup-inclusive loader speedup was 7.1-fold.

A separate real CUDA check resumed the preserved epoch-15 state twice. Twelve
Adam updates took 49.821 seconds serial and 8.444 seconds parallel, including
worker startup (5.9-fold speedup). Final model, Adam-state, CPU-RNG and CUDA-RNG
hashes were identical. The benchmark never wrote training checkpoints.

This is a throughput result, not an accuracy or uncertainty benchmark. It does
not guarantee 100% GPU utilization or an 8.5-fold complete-training speedup.
A 20-second live sample after restart averaged 16.05% GPU utilization (6–43%
range). GPU saturation was not achieved: unchanged full-resolution CPU
preprocessing still dominates the roughly 0.05-second GPU update. Worker startup,
heterogeneous image sizes, checkpoint serialization and GPU compute still take time. Other CPUs/RAM capacities may need fewer workers.

The user explicitly approved a checkpoint restart on 2026-09-29. The original
epoch-15 optimizer/model/RNG checkpoint and sidecar are retained privately before
replacement. Only unfinished epoch-16 work is replayed. The hourly monitor was
disabled at the user's request; this change does not re-enable it. No AI subagents
were used for this optimization.
