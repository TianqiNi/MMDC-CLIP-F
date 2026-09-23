Outcome: compute/public initialization is ready, but real classifier training is blocked by an unresolved scientific schedule decision.

- Accepted commit: `9fc8eb758288e196c31f22469d734aa96bb02ee7`.
- The protocol specifies Adam `1e-4`, weight decay `0`, batch `6`, 20 RSNA epochs only for confidence fitting ([protocol](../../docs/research/view_risk_protocol.md:334)).
- `ResearchRunConfig` enforces those values globally, and production fresh-classifier fitting reuses them ([training.py](../../src/mmdc_clip_f/research/view_risk/training.py:138), [production.py](../../src/mmdc_clip_f/research/view_risk/production.py:1393)).
- Legacy RSNA ViT-B/32 classifier fitting is materially different: Adam `1e-7`, weight decay `1e-5`, batch `3`, 50 epochs ([legacy config](../../configs/stage1/rsna_vit_b_32_legacy.yaml:29)).
- The accepted protocol requires a fresh public classifier but does not explicitly select its schedule. No real fitting should begin until that choice is frozen.

Resource findings:

- PyTorch 2.3.1+cu121 successfully used the RTX 4090; initial free memory was 22,383 MiB.
- Offline cached `openai/clip-vit-base-patch32` revision `3d74acf…0268` loaded through the production API.
- Tokenization succeeded: `[4,11]`; synthetic forward produced finite `[1,4]` logits.
- Batch-6 float32 four-view forward/backward succeeded with 1,754 MiB peak allocated and 2,014 MiB peak reserved. No optimizer or update was used.
- Measured warmed GPU-only step: 39.5 ms. The arithmetic extrapolation is 21.1 s/epoch and 7.0 min/20 epochs, but excludes decoding, hashing, augmentation, transfers, Adam updates, checkpoint writes, and tune evaluation; it is not an end-to-end runtime estimate.
- Each model state is about 605 MB. Twenty epoch states are approximately 12.1 GB; 50 are 30.3 GB, excluding public initialization and resume/Adam-state artifacts.
- Pylibjpeg/libjpeg/openjpeg are available; GDCM and JPEG-LS handlers are unavailable. Actual patient transfer-syntax compatibility remains untested.
- The probe exited cleanly with no Python/model GPU process remaining.

Artifacts:

- [Diagnostic script](../../scripts/research/p5a_resource_probe.py)
- [Sanitized aggregate evidence](../../evidence/phase5/p5a-resources.json)

JSON validation, Ruff, compilation, and diff checks passed. No production, protocol, dependency, test, commit, or push changes were made.

Next action: explicitly choose and protocol-freeze the fresh classifier schedule—legacy reproduction versus a newly justified schedule—then implement a separate immutable classifier schedule with focused RED/GREEN coverage before independent review or training.
