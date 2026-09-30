# View-removal support audit: measured results

**Decision: the signed signal exists, but it is sparse and its benefit remains
unproven.** This supports at most a bounded matched learning pilot, not a claim
that the current method is already strong enough for a paper. Shared labels do
not imply unchanged probabilities; the more serious concern is that the current
auxiliary target discards most probability changes and cannot distinguish many
stable wrong predictions from stable correct ones.

The frozen epoch-38 classifier was evaluated on all 987 confidence-fit patients.
The precommitted plan used 21 full-parent conditions and 82,908 removal pairs.
No confidence head was trained; no pilot or test outcomes were accessed. The
run took 416.6 seconds, including selected-artifact provenance verification.

## Clean inputs

| Outcome | Removal pairs | Share of 3,948 pairs | Distinct patients |
|---|---:|---:|---:|
| Repair | 87 | 2.20% | 61 |
| Damage | 103 | 2.61% | 79 |
| Unchanged | 3,758 | 95.19% | 987 |

Patient categories overlap for unchanged outcomes: every patient had at least
one unchanged removal. The 61 patients with repairs and 79 with damages are
mutually exclusive on the one clean parent, giving 140/987 patients (14.18%) with
any nonzero effect. Clean parent accuracy was 768/987 (77.81%). Of 219 errors,
61 (27.85%) had at least one repairing removal; 158 (72.15%) had no repairing
removal. The latter all produce zero correctness effects here. This is a
limitation of the auxiliary signal, not an upper bound on the global error
head, which still receives image/score features and direct error supervision.

Among 3,758 removals with unchanged predicted class, all changed probability
by more than 1e-6 in total variation; 3,313 (88.16%) changed it by at least 0.05.
Their mean total variation was 0.1212. Across all clean removals, maximum class
probability fell by 0.1104 on average and evidential uncertainty rose by 0.0801.
These changes may reflect the fusion's response to having less evidence; their
existence does not establish better uncertainty estimation.

| Density | Patients | Repair pairs / patients | Damage pairs / patients |
|---|---:|---:|---:|
| A | 95 | 19 / 15 | 13 / 11 |
| B | 425 | 39 / 24 | 50 / 37 |
| C | 416 | 23 / 18 | 36 / 28 |
| D | 51 | 6 / 4 | 4 / 3 |

Clean effects occur in every density class and every removed view, but class D
has only four patients supplying repair labels and three supplying damage
labels. That is weak support for a class-specific learning claim.

## Allowed stress conditions

| Stratum | Cells | Removal pairs | Repair pairs / distinct patients | Damage pairs / distinct patients | Unchanged share |
|---|---:|---:|---:|---:|---:|
| clean | 1 | 3,948 | 87 / 61 | 103 / 79 | 95.19% |
| single_view_stress | 16 | 63,168 | 1,483 / 230 | 1,994 / 302 | 94.50% |
| common_mode_stress | 4 | 15,792 | 365 / 228 | 421 / 259 | 95.02% |

The prescribed mild/moderate stresses broaden patient support, without needing
held-out families or strong severities. Nevertheless, zero effects still dominate.
The unique-patient counts above are unions across cells, not extra independent
patients. Single-view and common-mode cells have different accuracy/error rates;
their raw AURCs must not be interpreted as a controlled cross-stratum effect size.
The cell average does not reproduce training-sampler probabilities.

## Simple failure-ranking controls

Lower AURC is better; higher error AUROC is better. Orientations were fixed before
execution. These are descriptive confidence-fit results, with no fitted score,
post-hoc sign reversal, significance test or generalization claim.

| Fixed score | Clean error AUROC | Clean AURC | Mean single-stress AURC | Mean common-mode AURC |
|---|---:|---:|---:|---:|
| msp | 0.7061 | 0.1211 | 0.1238 | 0.1769 |
| score_sensitivity | 0.5063 | 0.2168 | 0.2192 | 0.3036 |
| probability_tv | 0.5167 | 0.2030 | 0.1876 | 0.2230 |
| prediction_flip_fraction | 0.5897 | 0.1888 | 0.1871 | 0.2489 |

MSP beats all three simple removal-based scores in mean AURC in each stratum.
On clean images, raw score sensitivity and probability sensitivity are near
chance by AUROC. This is a substantive caution: removal sensitivity is not, by
itself, a useful substitute for the classifier's existing confidence here.
It does not prove that a learned signed auxiliary task cannot help, but it gives
no empirical basis to claim that it will. The audit verifies target availability,
not target learnability, novelty, or added value over matched confidence heads.

## Recommended next decision

A small, fixed-budget matched learning pilot is defensible because clean repairs
and damages span multiple patients and classes, and permitted stresses broaden
support. Keep the original plan/settings; compare the signed-effect candidate
against the same-input global-error-only control and the mandatory confidence
baselines. The learned model must improve held-out failure ranking beyond these
controls without sacrificing the clean guardrail. A benefit only against the
weak raw sensitivity score is insufficient.

Do not start a full study or draft positive manuscript claims from this audit.
If the signed head adds no meaningful value over the same-input error head, or
its gains depend on synthetic stresses while harming clean performance, discuss
a methodological change rather than strengthening corruptions or selecting a
favorable subset. The current run does not examine proper-mask parents, which
remain part of the original training protocol; this limits conclusions about
auxiliary-target prevalence over the entire sampler.

## Verification and artifacts

All original 299 research tests and four added tests pass; both source CI runs
passed. Exact original clean preprocessing, first-batch classifier outputs for
all five variants, and all clean fused outputs were checked. Model state and
checkpoint bytes remained unchanged. A separate arithmetic verification checked
private cache hashes, all 82,908 signed targets from predictions/labels, patient
support and class totals. This is orchestrator review, without AI subagents.

`summary.json` contains all 21 cells and density/view breakdowns;
`result-validation.json` records acceptance checks. Private logits, labels,
row bindings and per-patient removal results remain outside git. Raw data,
weights, paused monitoring and unrelated user work are unchanged.
