# Intervention-supervised MV-ACN implementation ledger

Snapshot: 2026-09-10, P1A documentation handoff. Scope is the single
intervention-supervised confidence project described in
[view_risk_protocol.md](view_risk_protocol.md). Related evidence-gating and
conformal projects are alternatives and are not scheduled here.

**No phase is accepted or released by this document. No training, pilot result,
or test result is claimed.** The orchestrator committed P1A candidate
`2e9aebb68ea2bd67119d497d9b598f5de4388f48`; independent review 1 requested changes
with five substantive findings. This revision addresses those findings locally
and remains uncommitted, awaiting a new orchestrator candidate and fresh review.
Integration, push/PR evidence, and phase acceptance remain unverified here and
owned by the orchestrator. Sibling task status must be verified there; absence
of evidence here is not evidence of completion.

## 1. Rules and dependency map

```text
P1A protocol -------------------+
                               +--> P1 acceptance --> P2A roles/inventory
P1B subset fusion + targets ----+                       --> P2B stresses
                                                       --> P2C artifacts
                                                          |
                                                      P2 acceptance
                                                          |
                        P3A relation head/loss --> P3B matched controls
                                                          |
                                                      P3 acceptance
                                                          |
                 P4A metrics --> P4B fit/eval/CLI --> P4C smoke integration
                                                          |
                                                      P4 acceptance
                                                          |
              P5A fresh classifier --> P5B fit/freeze --> P5C measured pilot
                                                          |
                                                      P5 acceptance
                                                          |
                              full study (authorized if pilot succeeds and time/resources permit)
```

Within-phase dependencies are detailed below. **No task in the next phase may
start until the entire previous phase is integrated, reviewed, committed,
pushed, and its PR created or updated.** A local implementation or passing test
alone never opens the next phase. Refining boundaries must preserve this order
and accepted scientific scope; conflicts go to the orchestrator.

Software gates G1–G4 assess implemented contracts and scientific behavior using
synthetic fixtures, plus available external evidence. Missing raw data or patient
metadata must be recorded as a **real-data readiness blocker**, not a blocker to
generic P3/P4 software work after the preceding software phase is accepted.
The software must fail closed when real-data prerequisites are absent. No
synthetic check can mark an actual inventory audit or patient mapping complete.

Every task uses a fresh implementation agent. Coding and independent code
reviews use Sol at xhigh reasoning; writing uses Astra at high; phase gates use
Astra at medium. Review an immutable candidate commit with a fresh independent
reviewer. Return fixes to the same implementer; after fixes use a fresh reviewer
on the new immutable candidate. No nested agents. P1A spawns no agents.
Acceptance requires no critical, high, or medium actionable findings, no weakened
tests, and no unnecessary framework or dependency expansion.

The orchestrator alone owns integration, commits, pushing, PRs, dependency
conflicts, and phase release. Implementation agents change only assigned files;
P1A may write only these two documentation files and may not modify source,
install dependencies, train, or access test outcomes. Later training is scheduled
here, not authorized for this writing agent.

Coding tasks use meaningful red/green TDD for scientific behavior and negative
cases. Do not write tests for prose, boilerplate, or code-shaped assertions that
merely repeat the implementation. Existing CI only compiles/lints source and
checks CLI help; it is not scientific validation. No test weakening to obtain
green status. Keep private data/checkpoints/artifacts external; synthetic fixtures
must contain no real patient identifiers or images.

Status vocabulary: **pending** = no verified execution evidence;
**local draft** = written but not accepted; **in progress** = execution with an
owner and evidence; **blocked** = identified unmet prerequisite;
**accepted** = full reviewed integration evidence. An unrun experiment is pending,
not failed or complete.

## 2. Task ledger

| Task | Deliverable and dependencies | Acceptance evidence required | Current status |
|---|---|---|---|
| P1A — protocol documentation | These two documents; ground definitions in existing README/package/source. No predecessor within P1. | Correct signed targets and score contract; role/test isolation; explicit numerical stresses, fair controls, metrics, kill rules, task map; documentation review at candidate commit. | Accepted task; G1 passed at integration candidate `86031e5`. |
| P1B — combined subset fusion and intervention targets | One task and one implementer: named nonempty subsets using the legacy fusion tree, empty-input unavailable contract, error/correctness and signed three-way removal targets with validity masks. Uses P1A scientific contract; target construction follows fusion within this task. | Red/green fixtures for all 15 nonempty masks, both fusion trees, singleton/empty/invalid inputs, full-view numerical/prediction equivalence, repairs/damages/unchanged, identical predictions imply zero, wrong-to-different-wrong is zero, and no labels in inference features. | Accepted task; 40 tests passed; G1 passed at `86031e5`. |
| P2A — patient roles and inventory | Deterministic stratified patient manifests, private test denylist/lock, inventory/duplicate audit and sanitized counts; depends on accepted P1. | Software: synthetic patient-group isolation, count/duplicate checks, no eager test outcome access, and missing-mapping/mismatch refusal. Real data: verify patient mapping, inventory, and actual counts before fitting/evaluation; record unavailable prerequisites separately. | Software pending; real-data inventory and patient mapping unverified by P1A. |
| P2B — perturbations and masks | Reproducible parent/child transforms and sampling manifest; depends on P2A role schema and P1 subset contract. | Deterministic seeds; intensity/resolution defaults; held-out family/severity separation; common-mode control; all 14 proper clean masks; child retains exactly the parent's surviving tensors. | Pending. |
| P2C — provenance-bound artifacts | Frozen feature/target cache and role/checkpoint/mask/transform bindings; depends on P2A, P2B, P1B. | Tamper/stale/order/checkpoint/role rejection; all targets regenerated from realized inputs; external private records; prediction identity and finite-value checks. | Pending. |
| P3A — relation head and loss | 128-dimensional pooled views, two four-head relation layers, learned effect representations feeding global risk, constrained reported effects and raw-logit auxiliary CE plus error BCE; depends on accepted P2. | Frozen encoder gradients/parameters; both backbone dimensions; missing-view masks; parent-normalized raw-logit CE including deterministic-unchanged cases; exact reported `[0,1,0]` for identical predictions; effect-to-risk gradient path; singleton finite loss; lambda zero retains risk gradients; unchanged classifier predictions. | Pending. |
| P3B — fair baselines and ablations | All controls and ablations in protocol section 6, including same-input MLP/four-class and regenerated-target augmentation controls; depends on P3A input/loss contracts. | Input/exposure/capacity audit; score orientation; original four-view MV-ACN equivalence with labeled subset adaptation; auxiliary classifier never replaces frozen predictions; ViLU adaptation documented. | Pending. |
| P4A — metrics and paired inference | AURC/tie policy, both AP orientations, risk at coverage, Brier, effects, patient-paired bootstrap and seed summaries; depends on accepted P3. | Hand-computable ranking/tie/sign fixtures; one-class AP handling; all variants retained within patient clusters; paired difference and seed/patient separation. | Pending. |
| P4B — training/evaluation/CLI | Explicit role access, fresh-classifier initialization path, confidence fitting, tune selection, immutable pilot plan and lock enforcement; depends on P4A and P2/P3 artifacts. | Forbidden-role negative tests; deterministic resume/provenance; regenerated targets; fresh public CLIP versus diagnostic checkpoint distinction; no test-selection bypass. | Pending. |
| P4C — smoke integration and costs | End-to-end synthetic tiny run plus external-resource preflight; depends on P4B. | Synthetic nonempty masks and corruptions flow through target/head/control/metrics; CLI/config sanity; CPU smoke feasible without downloading weights; latency/memory procedure ready. Smoke outputs labeled non-scientific. | Pending. |
| P5A — fresh classifier pilot prerequisite | Audit real external resources; fit classifier only on classifier-fit, select on tune, freeze checkpoint; depends on accepted P4. | Public pinned initialization, permitted-role exposure log, actual counts, checkpoint/manifest hashes, verified predictions; original checkpoint results diagnostic only. | Pending; no training launched by P1A. |
| P5B — confidence/control fitting and freeze | Fit candidate and mandatory controls with matched draws/budgets, tune only, freeze evaluation table; depends on P5A. | Actual run/seed completion; candidate/control selections and hashes; no pilot/test exposure; incomplete controls disclosed. | Pending. |
| P5C — measured pilot and go/no-go | Run frozen non-test pilot; report primary/clean/secondary endpoints, paired intervals, seed spread, cost and limitations; depends on P5B. | Actual measurements against all mandatory controls; apply 10% target/0.005 clean guardrail and validity rules; go/no-go/inconclusive justified without test inspection. | Pending; no measured pilot or success claim. |

Fresh classifier training is P5 execution; P2–P4 may build and verify generic
data/artifact/head/training machinery using synthetic fixtures without starting
that study. P5 real-data fitting/evaluation requires a separate readiness record
for each dataset: verified patient grouping and inventory, reconciled actual
counts, role isolation, fresh-checkpoint provenance, and usable external resources.
Unavailable prerequisites block that dataset's real-data work, while unaffected
generic work can proceed through the software gates. P5A/P5B do not imply all datasets/backbones/seeds fit within the deadline.
Record the exact completed subset and retain the rest as pending.

## 3. Phase acceptance gates

| Gate | Required acceptance result | Evidence/status |
|---|---|---|
| G1: release P2 | P1A and combined P1B scientifically consistent; four-view legacy score/prediction compatibility and target edge cases verified; no source work inferred from documentation. | PASS: Astra medium reviewed `86031e5822680d223bed0175a83cc61280885056`, no actionable findings. | |
| G2: release P3 | P2 role/grouping, inventory-validation, lock/stress/provenance contracts pass synthetic positive/negative checks, including refusal of absent patient mappings. Record actual inventory evidence or real-data blockers separately; unavailable patient metadata does not block generic P3 software. No fabricated inventory. | Pending. |
| G3: release P4 | Head/loss and all matched control interfaces tested; frozen predictions, singleton/mask handling, capacity/input fairness demonstrated. | Pending. |
| G4: release P5 | Full synthetic pipeline and metric checks pass; real-data readiness is separately recorded as verified or blocked per dataset. Unknown/unavailable resources do not prevent software acceptance, but block dependent real-data execution. No real-data success inferred from smoke. | Pending. |
| G5: close pilot phase | P5 has measured outcomes with required controls or a truthful incomplete/blocker report; scientific go/no-go is separate from software acceptance. Full study is already authorized conditional on measured pilot success and time/resources; orchestrator releases tests only after those conditions and real-data readiness are verified. | Pending. |

For each task and gate, the orchestrator must append: owner, immutable candidate
commit, fresh reviewer identity/model, review disposition and fixed findings,
validation commands/results, integrated commit, pushed branch, PR URL/update,
and gate decision/time. The known P1A candidate/review evidence is recorded below;
remaining fields and fresh-review acceptance are **pending**. P1A itself has not
committed, pushed, opened a PR, or performed independent review.

Real-data readiness is independent of software acceptance: attach per-dataset
evidence or blockers to P2A and update it before P5 fitting/evaluation. A G2/G4
software pass cannot override missing patient mapping, exposure audit, or actual
counts. The full study follows a successful measured pilot under existing user
authorization if resources and the deadline permit; no new authorization request
is required. Otherwise retain it as pending and keep original tests locked.

## 4. Historical P1A drafting evidence and unresolved prerequisites

Completed local writing work:

- Read README, package metadata, classifier/fusion, backbone, data reader,
  confidence head/training/artifact/metric interfaces, legacy configs, and CI.
- Verified bibliographic summaries directly from the three requested primary
  arXiv pages; no broad literature search or original-paper reproduction.
- Wrote protocol definitions, planned stresses, fairness/metric assumptions,
  decision rules, and this dependency ledger. Numerical thresholds are design
  choices, not observed results.
- Reviewed local documentation consistency and repository diff scope. No prose
  tests were added, dependencies installed, source changed, or training run.

Local validation: `git diff --check` passed for tracked changes; a separate
documentation-only Python check covered both new files' relative links, balanced
code fences, trailing whitespace, final newlines, role-count sums, 14 proper
masks, and 48 primary cells. This is formatting/arithmetic validation, not model
or scientific-behavior testing. Review 1 was performed on the committed candidate;
fresh independent review of these fixes remains pending.

Review 1 evidence: `/tmp/mmdc-ivr-orchestration/results/p1-protocol-review1.md`,
reviewed candidate `2e9aebb68ea2bd67119d497d9b598f5de4388f48`, verdict **request
changes / not accepted**. The orchestrator accepted all five findings. Local fixes:

1. Learned effect representations explicitly feed the global risk MLP and receive
   both risk and auxiliary gradients.
2. Identical predictions force reported effect probabilities to `[0,1,0]`;
   auxiliary CE still uses raw logits for every valid removal.
3. Fusion and targets share one combined P1B task and implementer.
4. Synthetic software acceptance is separated from real-data readiness blockers.
5. Full-study authorization is conditional on pilot success/time/resources and
   requires no new user permission; operational test-lock gates remain intact.

These are documentation corrections, not claims that the software is implemented
or that the reviewer has accepted the new candidate.

Material prerequisites for the orchestrator:

| Item | Consequence / owner |
|---|---|
| Actual raw counts, patient mapping, duplicates, external weights and compute are unverified | P2/P5 must establish them. Counts in the protocol are targets; DDSM exam-level grouping alone is insufficient. |
| Existing code requires four views and eagerly resolves train/valid/test roles | P1/P2/P4 work is required before claiming subset support or safe locked-test operation. |
| Legacy confidence target generation and augmented fitting can disagree | P2 regenerated-target artifacts and P3 matched baselines are mandatory. Legacy reproduction alone does not validate the proposal. |
| Legacy ViT-L test selection and original checkpoint training exposure | Research entry points must reject these paths for held-out claims; checkpoint reuse is diagnostic only. |
| New numeric stress/head/training defaults are planned | Implement/review them consistently; scope conflicts go to orchestrator before changing the accepted design. |
| Review 1 candidate and findings are known; new candidate, fresh review and phase evidence are pending | Orchestrator creates the next immutable candidate and fills acceptance evidence. No phase is marked accepted here. |

There is no known contradiction with the accepted project scope in this draft.
Inventory disagreement or an implementation requiring changed classifier scores
would be a material conflict to escalate, not permission to silently alter the
protocol. No patient data was opened to resolve these questions in P1A.

## 5. Deadline, handoff, and retained work

Hard run deadline: **2026-09-10 20:50:09 UTC**. P1A target duration is at most
35 minutes for the initial task, with a 10-minute target for these review fixes.
The orchestrator proceeds phase by phase within the five-hour run;
it must not mark a multiday study complete because implementation is available.
If time/resources stop progress, record the last accepted phase, active task,
exact blocker, usable artifacts, and remaining task IDs. Keep tests locked and
classify an unmeasured pilot as pending/inconclusive.

Cleanup is limited to completed task-owned resources. Retain paused worktrees,
external run state, caches/checkpoints needed for resumption, and review evidence.
P1A owns only the two new documentation files and has no training resources to
clean up. Its handoff consists of those files, documentation validation, and the
unresolved prerequisites above; commits, integration, PRs, and subsequent phase
decisions remain with the orchestrator.

## Orchestrator resumption note — 2026-09-10

The user paused work at 16:34 UTC and requested continuation at 22:08 UTC.
The remaining active-work budget excludes that requested pause. The effective
cutoff for this resumed run is **2026-09-11 02:24:55 UTC** (September 10,
22:24:55 Toronto time), superseding the original wall-clock cutoff recorded
above and in the protocol. See `evidence/phase1/resumption-state.json`.
Scientific definitions and acceptance criteria are unchanged. P1A passed fresh
independent review at `ecb80e9bad005342575c526ceca99113a8044823`; P1B fixes passed fresh review at `6067abbe7da13f79c1831e21b26020ee5e560123`;
the phase gate remains pending. No training or patient evaluation has started.

## Orchestrator task-review record

P1A: original implementer GPT-6-Astra high; fresh reviews GPT-5.6-Sol xhigh.
Five findings at `2e9aebb` were fixed by the same writer; fresh review accepted
`ecb80e9bad005342575c526ceca99113a8044823` with no medium-or-higher findings.
P1B: original implementer GPT-5.6-Sol xhigh; fresh reviewers same model/effort.
Two high findings at `76cc677` were fixed by the same implementer. Regression
RED: 13 failed / 27 passed; GREEN: 40 passed. Fresh independent review accepted
`6067abbe7da13f79c1831e21b26020ee5e560123` with no material findings.
Reports, launch settings, and environment are retained in `evidence/phase1/`.
All agents ran through Herdr without nested agents. Phase gate, final phase
commit, push, and PR remain pending at this candidate.

## Phase 1 gate decision

G1 **passed** on immutable integration candidate
`86031e5822680d223bed0175a83cc61280885056`, reviewed by a fresh GPT-6-Astra
medium agent via Herdr. No actionable findings; source/tests unchanged from
accepted P1B. See `evidence/phase1/p1-gate.md` and `integration-checks.md`.
Final publication commit/PR information is recorded in the PR and subsequent
publication record. P2 starts only after the phase is committed, pushed, and
the PR is open. P2–P5 remain unexecuted at this boundary.

## Phase 1 publication

Phase commit `641b144` is pushed to `origin/research/intervention-supervised-confidence`.
Draft PR: https://github.com/TianqiNi/MMDC-CLIP-F/pull/1. Task and gate candidate
refs were also pushed so reviewed commits remain inspectable. G1 releases P2
after publication; software acceptance does not imply a measured research result.
Completed task/gate worktrees and Herdr workspaces were removed; required evidence
is retained under `evidence/`. Shared environment and orchestration remain for
dependent phases.
