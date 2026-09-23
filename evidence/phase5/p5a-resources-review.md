# PASS — Fresh independent P5A resource-audit reviewer

Reviewed exact commit: `182d5c08124311b254d5fc9199643cf64f110fde`
Evidence base: `9fc8eb758288e196c31f22469d734aa96bb02ee7`

PASS applies to the bounded resource-audit commit, not authorization to begin training.

## Findings

No Critical, High, or Medium findings.

Low:

- The Markdown report uses absolute links to the author’s `/p5a-resources` worktree, so its links are not portable to this detached review worktree.
- The 3,209-exam timing basis is the protocol target, not a patient-audited attained cohort count. The extrapolation is still correctly labeled synthetic and non-end-to-end.

## Evidence assessment

- The production loader resolves pinned `openai/clip-vit-base-patch32` revision `3d74acf…0268`.
- Offline operation is supported: Hugging Face and Transformers offline variables are set before importing the loader.
- The cached `pytorch_model.bin` remains present and hashes to the reported `a6308213…1576f`.
- Recorded batch-6, four-view float32 forward/backward succeeded with finite logits/loss:
  - 24 images per call
  - 1,754.239 MiB peak allocated
  - 2,014 MiB peak reserved
- No optimizer was constructed or stepped. The separately reported 1.21 GB Adam-moment figure is arithmetic, not a measured optimizer-step peak.
- The 39.5 ms measurement is accurately described as one warmed, GPU-resident forward/backward. The evidence explicitly excludes decoding, augmentation, transfers, optimizer work, checkpointing, tune evaluation, and hashing.
- Model-state and 20/50-checkpoint storage calculations are internally consistent.
- Cleanup is supported by the recorded post-probe process inspection; raw in-process cleanup retained only the CUDA context allocation before process exit.
- The decoder inventory matches the shared environment. Actual patient transfer-syntax compatibility remains untested.
- The probe path contains no patient, pilot/test, or original fine-tuned checkpoint access.

## Unresolved training decision

The evidence correctly identifies a concrete implementation discrepancy:

- Production fresh-classifier fitting currently receives Adam `1e-4`, weight decay `0`, batch `6`, 20 RSNA epochs, seed `42` from the confidence-oriented `ResearchRunConfig`.
- The legacy RSNA ViT-B/32 classifier configuration specifies Adam `1e-7`, weight decay `1e-5`, batch `3`, 50 epochs, seed `42`, no AMP, and RandAugment `3/9/31`.

The protocol requires a fresh public classifier but does not choose between those schedules. Selecting legacy reproduction or a newly justified schedule remains a user scientific decision. Real training should not begin until that decision is protocol-frozen.

## Validation performed

I did not repeat the established CUDA/model probe.

Bounded review checks:

- Confirmed detached HEAD equals the exact reviewed commit.
- Confirmed the commit adds only the probe and two evidence artifacts.
- Verified raw-output and script hashes/sizes against the aggregate JSON.
- Cross-checked load/token identities, batch metrics, state bytes, storage arithmetic, step count, and scope guards.
- Rehashed the pinned cached public weight and protocol.
- Traced the production, protocol, legacy configuration, and recorded originating commits.
- Independently checked installed decoder packages and handler availability.
- Confirmed the optional validation JSON parses cleanly.

Scientific limitations remain: no real decoding or throughput measurement, optimizer step, checkpoint-write benchmark, end-to-end epoch estimate, patient codec validation, attained cohort validation, or public-pretraining medical-independence audit.

Machine-readable result: [p5a-resources-review-validation.json](../../evidence/phase5/p5a-resources-review-validation.json)

No production/configuration changes, training, commits, or pushes were made.
