# Intervention-supervised MV-ACN research protocol

Status: planned protocol, prepared in P1A on 2026-09-10. No new training,
inventory validation, pilot measurement, or test evaluation is reported here.
Implementation and acceptance evidence belong in [implementation_status.md](implementation_status.md).
Only the intervention-supervised confidence project is authorized. Evidence
gating and conformal modality selection are related work, not additional projects.

## 1. Question, estimand, and limits

Does supervision about the **signed change in classification correctness after
removing an observed view** improve failure prediction for a frozen MMDC-CLIP
classifier, beyond the information, capacity, and augmentations available to
matched confidence baselines?

The intervention changes the computational input. It does not identify a causal
effect on a patient, diagnose image quality, or decompose epistemic and aleatoric
uncertainty. Synthetic stresses are not clinically validated acquisition defects.
The experiment estimates performance on the specified datasets and stresses;
it does not establish safety, clinical utility, or generalization to a hospital.
The auxiliary effect scores need not form a coherent joint uncertainty
decomposition with one another or with the global error probability.

All confidence methods must rank **the same frozen classifier predictions** on
each input. No confidence head may change fusion, discard views to improve the
reported prediction, or substitute its own class prediction. Removal is used to
construct features and supervision, not to deploy a new classifier.

## 2. Repository contract and compatibility

This protocol was grounded in repository revision
`b791722b78a3ba67ecad36d7635a5a7c80f3420a`, [README](../../README.md),
[package metadata](../../pyproject.toml), and these existing interfaces:

| Existing interface | Observed behavior and implication |
|---|---|
| `mmdc_clip_f.model.MultiViewCLIPClassifier` | Named-view mapping; `encode_views` exposes projected embeddings and `last_hidden_state`; currently validates that all four views exist. Subset support is planned P1 work. |
| `fuse_logits`, `classify_embeddings`, `forward_with_hidden_states` | Per-view alpha is `softplus(view_logits) + 1`; paired DS fusion returns **`softplus(fused_alpha)`** as the four class scores. Hidden-state and ordinary forward paths need a compatibility check. |
| `mmdc_clip_f.data.VIEWS`, `FourViewDataset`, `ExamRecord` | Canonical order is `L_CC,L_MLO,R_CC,R_MLO`. DDSM manifest columns instead list `L_CC,R_CC,L_MLO,R_MLO`; the reader maps names correctly. `sample_index` is manifest-local; DDSM `group_id` is an exam ID, not proof of patient identity. |
| `mmdc_clip_f.backbones.BACKBONES` | Pinned public CLIP ViT-B/32 at 224 pixels, hidden width 768; ViT-L/14-336 at 336 pixels, hidden width 1024. Preserve pinned revisions, prompts, and preprocessing for each run. |
| `confidence.model.MVACNHead` | Four-view token attention with four heads and a confidence token; MLP ends in one logit. `predict_confidence` applies sigmoid. Existing head does not accept missing views. |
| `confidence.training`, `confidence.artifacts` | Existing TCP artifacts bind checkpoint/manifest/prediction provenance. Legacy fitting uses MSE against cached TCP while training images can be augmented; these cached clean targets do not satisfy the new perturbation contract. |
| `confidence.metrics.confidence_selection_metrics` | Existing AUPR is **correct-positive**, with confidence as score. Existing AURC uses stable rank ordering. New metric naming and tie policy must be explicit. |
| `mmdc-clip-f` CLI | Existing commands include `validate-data`, `train-classifier`, `build-tcp`, `train-confidence`. Research role-aware commands/configuration are planned, not currently available. |

Legacy DDSM fusion first combines `(L_CC,R_CC)` and `(L_MLO,R_MLO)`;
RSNA first combines `(L_CC,L_MLO)` and `(R_CC,R_MLO)`. Both then combine the two
branches. Retain each configured tree, including its operation order. Do not
replace the legacy score by `fused_alpha / sum(fused_alpha)`, even though that is
a Dirichlet mean. Class probabilities used for TCP and scalar baselines are
`softmax(s_O)` where `s_O = softplus(fused_alpha_O)`.

Planned subset rule: prune absent leaves from the configured fusion tree; combine
two surviving children with the existing DS operation, pass one surviving child
through, and omit an empty branch. A singleton therefore has alpha
`softplus(view_logits) + 1`, followed by the same outer softplus score transform.
Never inject a zero image, duplicate a view, or treat absence as observed evidence.
The full four-view branch must preserve the legacy numerical path. Fail on
unknown/repeated names, inconsistent shapes, or nonfinite results; any numerical
stabilization that changes the legacy path needs explicit review and versioning.
Empty input returns an unavailable/referral status with **no confidence or class
prediction**. It is excluded from selective-risk denominators and counted
separately.

The classifier, including CLIP image/text encoders, projections, logit scale,
prompts, and DS operation, is frozen during confidence fitting and evaluation.
Use evaluation mode and detached/no-gradient features. Verify both unchanged
parameters and identical predictions for each method on the same input; changed
predictions across genuinely different masks or corruptions are expected.

## 3. Targets and head

Let `V = {L_CC,L_MLO,R_CC,R_MLO}`, `O` be a nonempty observed subset,
`y in {0,1,2,3}` the density label, and `yhat_O = argmax_k s_O[k]`.
Break exact class-score ties by the lowest class index consistently.

```text
e_O       = 1[yhat_O != y]
correct_O = 1 - e_O
d_(O,v)   = e_O - e_(O\{v}),    v in O and |O| >= 2
```

| Current error | Error after removal | Effect label | Meaning |
|---:|---:|---:|---|
| 1 | 0 | +1 | Removal repairs the prediction; the present view is harmful for this input/label. |
| 0 | 1 | -1 | Removal damages the prediction; the present view is helpful for this input/label. |
| 0 | 0 | 0 | Correctness unchanged. |
| 1 | 1 | 0 | Correctness unchanged, even if the wrong class changes. |

If `yhat_O == yhat_(O\{v})`, `d_(O,v)` **must be zero**. A changed prediction
does not imply a nonzero effect. Persist CE class ordering as `[-1,0,+1]` with
indices `[0,1,2]`; distinguish signed targets from class indices in artifacts.
Singletons contribute current-error supervision but have no removal label or
effect loss. Missing views have no target. Invalid slots are masked, not encoded
as unchanged correctness.

Targets must be recomputed from the exact realized input. Generate an augmented
parent once, then remove a view without redrawing the surviving views. Reuse
these same per-view tensors/embeddings for all its child subsets. Labels `y` are
used only for training targets and authorized evaluation; inference features
contain no correctness, true-class probability, or ground-truth removal label.

Planned head defaults:

1. Mean-pool frozen tokens within each observed view, then use a shared learned
   projection to 128 dimensions. Preserve the backbone-specific token policy
   (ViT-B excludes CLS; ViT-L includes CLS) and report it. Give matched controls
   exactly the same frozen source features and token policy. Shared feature
   preparation denotes the same specification, not reuse of a projection fitted
   by the candidate: each learned control fits its own projection within its
   reported parameter budget.
2. Add named-view/mask information and a projected feature vector containing
   current scores/probabilities, per-view evidence, and available signed
   leave-one-view-out score differences `s_O - s_(O\{v})`. Singleton removal
   features are absent with an explicit validity mask. Feature scaling is fitted
   on confidence-fit only. Do not input corruption family or severity IDs.
3. Apply two interaction layers, each with four attention heads of width 32,
   residual connections, layer normalization, and a 128-to-256-to-128 feed-forward
   block. Use learned attention biases for same-breast and same-projection
   relations, with an explicit self relation. Mask absent keys and queries.
   Default dropout is 0.2. Relation biases use view names, not manifest position.
4. For each removable view, form a learned 128-dimensional effect representation
   `r_(O,v)` with a shared MLP from its interacted view representation and available
   removal-score features. Project `r_(O,v)` to three raw `effect_logits_(O,v)`.
   Feed the masked mean of these **learned effect representations explicitly into
   the global risk MLP**, concatenated with masked pooled interacted view features
   and current score/evidence features. Gradients from both error BCE and effect
   CE reach the effect representation; do not detach it. For a singleton, use a
   zero effect-summary vector plus an explicit no-removal indicator, retaining
   its view/evidence inputs. Set `p_error = sigmoid(error_logit)` and final
   confidence `c = 1 - p_error`.
5. Compute raw effect probabilities with softmax, then enforce the deterministic
   reporting constraint: for every valid removal with
   `yhat_O == yhat_(O\{v})`, return exactly `[0,1,0]` in class order `[-1,0,+1]`.
   Thus reported repair and damage probabilities are exactly zero and unchanged
   probability is exactly one, regardless of raw logits. Otherwise return the
   raw softmax probabilities. This label-free constraint applies during training
   reporting and inference; missing/singleton slots remain invalid, not unchanged
   predictions. The risk head consumes learned representations as specified
   above, not ground-truth effects or forced probability targets. This local
   consistency constraint does not make the remaining effect probabilities a
   coherent joint decomposition or their sum an additive risk budget.

For a minibatch of nonempty parent inputs, average the following per-parent loss:

```text
L_O = BCEWithLogits(error_logit_O, e_O)
      + lambda * mean_{valid removable v} CE(effect_logits_(O,v), d_index_(O,v))
```

CE is computed on the **raw, unconstrained effect logits** for every valid
removal, including identical-prediction cases whose target is unchanged (index
1). The exact `[0,1,0]` overwrite is a reported-probability constraint after
logits; do not take logs of overwritten zeros, mask these valid CE examples, or
replace their learned loss by a constant. Report effect classification metrics
from constrained probabilities and label any raw-logit diagnostic separately.
With `lambda=0`, retain the learned effect-representation path into global risk;
it still receives error-BCE gradients, but no auxiliary CE gradients.

Set the empty effect mean to zero for singletons. Average over parents before
averaging batches so parents with more removable views do not receive extra
weight. Default `lambda=1`; `lambda=0` is the mandatory no-effect-supervision
ablation. Keep unweighted BCE/CE as defaults and report label frequencies; any
reweighting is a declared tune-only option, applied fairly and recorded before
pilot inspection. Confidence is a probability estimate, not an assertion that
calibration has already been demonstrated.

## 4. Roles, isolation, and provenance

Original test sets stay locked. The proposed non-test partition uses the union
of original train and validation, with these **target counts**, not an inventory
claim:

| Dataset | Classifier-fit | Confidence-fit | Tune | Pilot | Non-test total | Original locked test |
|---|---:|---:|---:|---:|---:|---:|
| RSNA-SMBC | 3,209 | 987 | 370 | 371 | 4,937 | 872 |
| MINI-DDSM | 879 | 270 | 102 | 102 | 1,353 | 339 |

README original train/valid totals are RSNA `4,647+290` and DDSM `1,082+271`.
P2 must audit raw inventory, class counts, missing/duplicate views, duplicate
files/content, patient identity across exams, and cross-role overlap before
declaring these counts attained. Use deterministic patient-grouped,
density-stratified assignment, default split seed 42. Keep all exams/images of
one patient in one role; never split a group to force the requested counts.
For conflicting labels within a patient, document the group-stratification rule
and actual class distribution. Report requested and actual patient/exam counts,
exclusions, and reasons separately. DDSM exam IDs require an external verified
patient mapping; absent that evidence, patient-level independence is unverified
and real-data fitting/evaluation for that dataset is blocked. This readiness
block does not prevent generic software implementation or phase acceptance on
synthetic fixtures that verify the checks and refusal behavior. Record software
acceptance and real-data readiness separately; passing fixtures never certifies
actual patient independence or counts.

| Role | Permitted use |
|---|---|
| Classifier-fit | Fine-tune a fresh classifier initialized from the pinned **public pretrained CLIP** weights. No confidence-fit, pilot, or test patient is used for classifier gradients. |
| Confidence-fit | Train every learned confidence head, auxiliary predictor, feature scaler, and logistic control using the frozen fresh classifier. These patients must also be absent from classifier checkpoint selection. |
| Tune | Select classifier checkpoint, then confidence model/hyperparameters/checkpoints and temperature/calibration parameters. Reuse is declared development exposure, not held-out performance. |
| Pilot | One frozen-protocol, non-test feasibility and go/no-go evaluation; no training, calibration, checkpoint selection, or choosing favorable seeds. A revision after seeing pilot results consumes that pilot as development data and needs a new independent evaluation plan. |
| Original test | The full study is already authorized conditional on pilot success and available time/resources. Access only after protocol/model/baselines/metrics are fixed, real-data readiness is verified, and the orchestrator records that those conditions hold and releases the lock. This is an operational gate, not a request for new user authorization. No role reassignment, target generation, fitting, or selection beforehand. |

Previously fine-tuned original checkpoints cannot yield genuinely held-out
confidence-fit predictions after repartitioning their training patients. They
are diagnostic-only, with exposure disclosed, and cannot support a pilot success
claim or a new held-out confidence comparison. Public pretrained CLIP provenance
does not establish that its pretraining corpus was medically independent; record
that limitation without implying a verified pretraining audit.

Current loaders/configuration expect `train/valid/test` and can eagerly construct
all splits. P2/P4 must implement explicit role access and lock enforcement before
using them for this study; simply renaming a role `test` or relying on
`--allow-test-selection` is unacceptable. Legacy ViT-L confidence configs select
on test and must not be used as research selection configs. Identity-only test
overlap auditing, if needed, uses a custodian-provided private denylist/digest
without loading test outcomes or images.

All patient identifiers, medical images, row-level predictions/targets, private
manifests, and checkpoints remain external to git, including pseudonymous
linkable records. Commit only schemas, synthetic fixtures, sanitized aggregate
counts, configuration, and nonidentifying artifact hashes. Artifact keys bind
dataset/role and manifest hash, private patient/exam key, mask/view order/fusion
tree, transform family/parameters/realization seed, preprocessing/backbone
revision, classifier checkpoint hash, target/schema version, and implementation
revision. Reordering, stale targets, role/checkpoint mismatch, or changed
perturbations must cause rejection or explicit regeneration. Store the private
patient-cluster map externally for paired evaluation. P1A does not access raw
inventory or any patient outcomes.

## 5. Planned perturbation protocol

Apply stresses after the existing deterministic resize and conversion to RGB
float `[0,1]`, before existing normalization (`mean=[0.485,0.456,0.406]`,
`std=[0.229,0.224,0.225]`). Retain legacy DICOM per-image scaling and configured
224/336 resolution. Clip stressed pixels to `[0,1]` and do not renormalize their
dynamic range. These are planned stress defaults, not clinically validated
defects, label changes, or an empirical simulation of missingness mechanisms.

Let `r` be image width/height and `a=r/224`. Use reflect padding for convolution.

| Family | Mild | Moderate | Strong, evaluation only | Exact operation |
|---|---|---|---|---|
| Gaussian noise | sigma 0.02 | sigma 0.05 | sigma 0.10 | Add zero-mean Gaussian noise on `[0,1]`; one spatial noise field replicated across RGB channels. |
| Gaussian blur | sigma `0.7a` px | sigma `1.4a` px | sigma `2.8a` px | Normalized Gaussian kernel, odd width `2*ceil(3*sigma)+1`. At 336, sigmas are 1.05/2.10/4.20 px. |
| Contrast | factor 0.80 | factor 0.60 | factor 0.40 | `mu + factor*(x-mu)` with per-image spatial/channel mean `mu`. |
| Brightness | factors 0.90 and 1.10 | 0.75 and 1.25 | 0.60 and 1.40 | Multiply intensities; evaluate both directions separately and average them within the brightness cell. |
| Crop | retained area 0.90 | 0.75 | 0.60 | Square crop of side `floor(r*sqrt(area))`, uniformly sampled valid location, then bilinear resize with antialiasing back to `r`. |
| Motion blur | length 5/7 px | 9/13 px | 17/25 px | Widths are for 224/336 respectively; normalized straight horizontal or vertical line kernel, both orientations evaluated and averaged within the cell. |

Confidence fitting and tune selection may use only Gaussian noise and Gaussian
blur at mild/moderate severity. Hold out contrast, brightness, crop, and motion
blur **as families from confidence training and tune selection**, and hold out
all strong severities. The classifier's legacy RandAugment can include related
operations; disclose this and describe holdout relative to confidence fitting,
not as proof the backbone has never seen such transformations. Record exact
classifier augmentation separately.

Training sampler default: equal parent probability for (a) clean four-view,
(b) clean proper nonempty mask, (c) stressed four-view, (d) stressed proper
nonempty mask. Within each masked stratum sample uniformly over all 14 proper
nonempty masks (`4+6+4` singletons/pairs/triples); retain mask names, not just
cardinality. Within stress strata balance the two fitting families and two
severities. Apply stress to one uniformly selected observed view in 80% of
stressed parents, and to all observed views in 20% as a common-mode control.
Common-mode uses the same family/severity and shared random parameters/noise
field or normalized crop location across views. A corrupted but absent view
cannot be treated as observed.

Evaluation panels, identical for all methods:

- Clean four-view, plus every one of the 14 proper nonempty clean masks.
- Primary held-out-family panel: four families by three severities by four
  targeted views, always starting with four observed views. This gives 48 cells;
  average brightness directions and motion orientations inside their cells.
- Strong Gaussian noise/blur on each target view, reported separately from the
  primary held-out-family panel.
- Common-mode versions of all evaluation families/severities, reported
  separately; do not assume a single bad view explains common-mode failure.

Use one deterministic realization per patient/exam/cell/direction by default,
with evaluation seed 4242 and stable private-key hashing; freeze the table before
pilot. Repeated realizations, if later authorized, remain within the patient
cluster and do not increase the independent sample count. Generate all removal
targets from the realized parent, including for baseline TCP and correctness
targets. No corruption metadata or clean counterpart is supplied as an inference
feature. Clean missing masks and synthetic removal are controlled computational
tests, not a model of naturally missing clinical views.

## 6. Mandatory controls and ablations

Every learned control uses the same eligible confidence-fit patients, frozen
classifier, applicable masks, training budget, seed list, and tune-only selection.
No baseline is fitted on pilot or test patients. Use a common finite candidate
budget and record trainable parameters, updates, and selection trials. Failure
to complete a mandatory control makes novelty unresolved, not supported.

| Control | Required definition and fairness constraint |
|---|---|
| MSP / TS / margin / entropy / Energy | Use `softmax(s_O)`: MSP maximum probability; temperature scaling with `T>0` fitted on tune NLL; top-two probability margin; negative entropy as confidence; Energy error score `-T*logsumexp(s_O/T)` with fixed `T=1` primary. Orient all rankings explicitly. TS does not change `argmax`. |
| DS vacuity/conflict logistic | Fit error logistic regression using `4/sum(alpha_O)`, per-view vacuities, and pair/tree conflicts from the same DS path; include validity/missingness masks. Fit weights/scaling on confidence-fit, regularization on tune. DS values are features, not validated uncertainty decompositions. |
| Original MV-ACN | Refit the existing architecture with TCP/MSE on clean confidence-fit inputs using this classifier, with original backbone-specific token/dropout conventions. Report the original cached-clean-target/RandAugment recipe only as a labeled diagnostic if reproduced; it cannot replace the matched control. |
| Correctness MV-ACN | Same architecture and clean inputs, BCE for current correctness (or equivalent error complement), isolating the objective from TCP. |
| Augmentation-matched MV-ACN | Same perturbation/mask draws as the proposed method; regenerate targets for each realized input. Include TCP/MSE and correctness/BCE variants so augmentation and objective are both controlled. |
| ViLU adaptation | Use frozen visual embedding, predicted-class text embedding, and image-conditioned class-text representation with failure/correctness BCE. Pool only observed views, preserve this classifier's prediction, document multi-view adaptation and any departure from original weighting. Match augmentation exposure. |
| Same-input capacity-matched MLP | Flatten the exact proposed prepared features in canonical slots with masks; replace relation interaction with an MLP, use current-error BCE, and match trainable parameters within 10%, reporting actual counts. No intervention labels are inference inputs. |
| Same-input four-class predictor | Use the same feature vector and comparable parameter budget to predict density via CE. Confidence is its probability assigned to the frozen classifier's `yhat_O`, not its own argmax. This tests whether gains come from relearning classification. |
| Absolute removal logit sensitivity | Error score `max_v mean_k abs(s_O[k]-s_(O\{v})[k])` over valid removals; singleton statistic is zero with an explicit no-removal indicator. Report raw ranking and tune-calibrated probability if Brier is reported. This tests unsigned sensitivity without effect learning. |

For original MV-ACN on missing masks, create a labeled masked adaptation that
omits absent view tokens while retaining its attention/MLP; demonstrate exact
four-view equivalence. Never use zero-image tokens as missing data. Its clean
four-view result remains the direct original-architecture comparison; identify
subset adaptation results explicitly.

For scalar scores without a probability interpretation, report ranking metrics
directly and correctness Brier only after a declared monotone logistic map fitted
on tune (or mark Brier unavailable). Report raw and calibrated results separately;
do not call sigmoid conversion alone evidence of calibration.

Required ablations, one factor at a time with the same patient/augmentation draws:

| Factor | Comparison and interpretation |
|---|---|
| Effect supervision | `lambda=1` versus `lambda=0`, retaining inputs/capacity. |
| Intervention features | Remove removal-score differences and their derived statistics while retaining effect supervision. |
| View relationships | Remove same-breast/same-projection biases while retaining named views and comparable capacity. |
| Signed versus magnitude effects | Replace three-way target by binary `abs(d)` while preserving the learned effect-to-risk path. For identical predictions, reported magnitude probabilities are exactly `[1,0]` in order `[0,1]`; retain raw-logit CE on all valid removals. |
| Task harm versus corruption recognition | Replace auxiliary effect task by whether each observed view was synthetically stressed, with equal weighting and the same learned auxiliary-representation input to risk; inspect effect performance for the effect model within clean and stressed strata. The identical-prediction constraint applies to effects, not corruption labels. Do not equate a stressed view with a harmful view. |
| Evidence versus hidden features | Evidence/score-only, hidden-only (with names/masks), and combined inputs; report parameter counts and inference costs. |
| Clean versus augmented fitting | Fit clean parents only versus the matched stress sampler; recompute all targets in both. |

## 7. Selection, metrics, and uncertainty

Planned confidence optimizer: Adam, learning rate `1e-4`, weight decay zero,
batch size 6; up to 20 epochs on RSNA or 50 on DDSM, matching existing confidence
budgets initially. Fit head seeds `[42,43,44]`; pair each method on the same
classifier checkpoint and data/perturbation schedule. A one-seed pilot is allowed
as feasibility evidence but cannot estimate seed spread. Default pilot first
configuration is RSNA ViT-B/32; other dataset/backbone combinations remain
required follow-up evidence for claims about both datasets/backbones. This order
is a compute default, not permission to claim unfinished combinations.

Use tune only for checkpoint selection: minimize mean AURC over the balanced
mild/moderate Gaussian noise/blur single-view panel subject to clean AURC no
more than 0.005 above the clean correctness-MV-ACN reference selected on tune.
Select that reference first by clean tune AURC. Resolve exact ties by lower
clean AURC, then earlier epoch. Freeze any hyperparameter search table before
execution and give learned methods equal selection budgets. Held-out families,
strong severity, pilot, and test cannot select the model. If no candidate passes
the tune clean guardrail, report that failure before pilot.

Primary endpoint is the **unweighted mean of the 48 held-out corruption-cell
AURCs**, with each family/severity/target-view cell equally weighted. Compute
AURC within each cell across eligible exams, then average; never concatenate all
variants as independent patients. Report dataset/backbone results separately;
any cross-dataset summary is an explicitly labeled equal-weight macro average
with all prespecified combinations present. Masks, common-mode, and stronger
seen-family stresses are secondary panels, not extra weights in the primary.

For `n` exams sorted by descending confidence, risk at rank `k` is the error
fraction among the first `k`, and `AURC = (1/n) * sum_{k=1..n} risk(k)`.
Use expected risk under random ordering within exact confidence ties, computed
from the tied group's mean error, so ordering cannot depend on labels or file
order. At coverage 0.8/0.9 use `k=ceil(coverage*n)` and the same tie expectation;
report realized coverage. This tie rule is a declared research metric extension,
not a claim of identical behavior to legacy stable sorting.

Report at least:

- AURC (lower is better), error-positive average precision
  `AP(e_O, p_error)` (higher is better), and risk at 80% and 90% coverage.
- **Legacy correct-positive AUPR**, `AP(1-e_O, c)`, explicitly separate from
  error-positive AUPR. Use non-interpolated AP with grouped score ties. If only
  one correctness class occurs, report AP unavailable and the class count.
- Correctness Brier `mean((c-(1-e_O))^2)`; classifier accuracy/error prevalence
  for each panel to expose endpoint difficulty.
- Effect confusion matrix, per-class support/recall, balanced accuracy and macro
  F1 with absent classes marked unavailable; separate repairs and damages, clean
  and stressed inputs, and singleton counts. Report the majority-zero reference.
- Batch-1 end-to-end latency and peak accelerator memory, plus head-only time,
  parameter count, and feature/target-cache cost. Use the same hardware, precision,
  warmup (20 calls), timed calls (100), and device synchronization. Include all
  required subset fusion/features and encoding, even when cached offline.

Use a **patient-paired bootstrap** with 2,000 resamples and seed 2026 for 95%
percentile intervals of candidate-minus-baseline differences. Sample patients
within each dataset with replacement, carrying all their exams, masks,
corruption realizations, and every method together; recompute cell metrics and
the macro endpoint each time. Never bootstrap variant rows independently.
Point metrics are exam-level, so patients with multiple exams contribute multiple
exams; disclose their counts while keeping patient clustering for inference.
Do not imply that 48 stresses multiply the sample size by 48. Compute paired
differences within each training seed, and optionally average those paired
differences within each bootstrap draw using the same patient draw. Separately
report mean/SD/range across the fixed training seeds; seeds are not additional
patients. Undefined replicate metrics must be counted and reported, not silently
converted to zero. Pilot intervals are exploratory; multiple baseline/ablation
comparisons do not establish unadjusted confirmatory significance.

## 8. Go/no-go and candidate kill rules

Freeze the candidate, control suite, score orientation, artifact hashes, and this
endpoint definition before pilot outcomes are read. Select a named reference
baseline by tune performance using the common guardrail; also report comparisons
against **all** mandatory controls so a weaker named reference cannot hide a
stronger same-input or augmentation-matched method.

The practical pilot target is approximately **10% relative reduction in primary
mean AURC** versus the strongest completed mandatory baseline, with clean
four-view AURC degradation at most **0.005 absolute** versus that same baseline.
For each baseline `b`, relative gain is `(AURC_b-AURC_new)/AURC_b`; if the
denominator is zero, mark the relative gain undefined and compare absolute
differences. Report paired intervals for primary and clean differences and
every mandatory comparator; a positive point estimate alone is not proof.
For the planned three-seed pilot, apply the 10% and 0.005 thresholds to the mean
of per-seed endpoints. A go additionally requires favorable primary point
differences in all three seeds, an upper 95% paired-bootstrap bound below zero
for the seed-mean primary difference against each mandatory control, and an
upper bound at most 0.005 for clean degradation against the strongest primary
baseline. These are exploratory progression criteria, not a multiplicity-adjusted
claim of significance. One-seed results, missing signed-effect classes, or an
interval crossing a progression boundary are inconclusive for this gate.

| Finding | Decision |
|---|---|
| Leakage, unverified patient groups, stale targets, changed classifier predictions, or a full-view compatibility failure | Stop scientific interpretation; fix and independently review. Do not unlock tests. |
| Fresh classifier/required data unavailable; mandatory controls or pilot not run; deadline prevents measurement | Inconclusive / pending, with blocker and remaining work. Never report success from a smoke run or diagnostic checkpoint. |
| Clean degradation exceeds 0.005, or primary gain is nonpositive against the strongest mandatory control | No-go for this candidate under this protocol. No test-based rescue. |
| Gain is positive but below 10%, intervals include no improvement, seed behavior is unstable, or repair/damage support is inadequate | Inconclusive; do not claim success. Any further study needs a recorded plan without reusing pilot as an unseen holdout. |
| At least 10% gain, clean guardrail and the interval/seed criteria above pass, all mandatory controls complete, no validity failures | Proceed to the already conditionally authorized full study if time/resources and real-data readiness permit; orchestrator records gate satisfaction and releases the test lock. Pilot is still preliminary evidence, not a clinical or final study claim. |

Novelty is conditional on beating **same-input capacity-matched, four-class,
and augmentation-matched controls**, and on the supervision/feature ablations
supporting the signed-effect account. If those controls explain the improvement,
reject the intervention-specific novelty claim even if a useful confidence head
remains. Corruption recognition alone, a better classifier, and unsigned logit
sensitivity are alternative explanations this design must actively test.

The five-hour implementation run ends at **2026-09-10 20:50:09 UTC**. Software
completion does not imply the multi-seed/multi-dataset study ran. Stop launching
work that cannot complete safely in the remaining budget, preserve paused
resources, and record actual partial progress. Only the orchestrator controls
integration, dependency resolution, commits, pushes, PRs, and phase release.

## 9. Closest primary work

- [Lafon et al., ViLU: Learning Vision-Language Uncertainties for Failure
  Prediction, ICCV 2025, arXiv:2507.07620](https://arxiv.org/abs/2507.07620):
  combines visual and task-text representations for binary failure prediction.
  This motivates a direct adapted confidence baseline; merely adding a failure
  head to CLIP is not the proposed novelty.
- [Shi et al., GAUGE: Granularity-Adaptive Counterfactual Gating of Evidence for
  Incomplete Multimodal Classification, arXiv:2608.05608](https://arxiv.org/abs/2608.05608),
  preprint: uses counterfactual evidence scores for gating in incomplete
  multimodal classification. Our proposed comparison keeps classifier outputs
  fixed and learns confidence using supervised discrete removal effects; it
  does not implement GAUGE's gating or claim counterfactual reasoning is new.
- [Chandy et al., Uncertainty-Aware Multimodal Learning via Conformal Shapley
  Intervals, arXiv:2602.00171](https://arxiv.org/abs/2602.00171), preprint:
  studies uncertainty-aware modality importance and selection. One-step signed
  correctness differences here are neither Shapley values nor conformal
  intervals and provide no coverage guarantee.

These brief comparisons identify adjacent work, not an exhaustive novelty
review. The base classifier and MV-ACN attribution remain the
[MMDC-CLIP-F paper linked by the repository](https://doi.org/10.1016/j.neucom.2026.134795).
