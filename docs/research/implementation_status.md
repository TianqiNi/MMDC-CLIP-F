# Intervention-supervised MV-ACN implementation ledger

Snapshot: 2026-09-10, P1A documentation handoff. Scope is the single
intervention-supervised confidence project described in
[view_risk_protocol.md](view_risk_protocol.md). Related evidence-gating and
conformal projects are alternatives and are not scheduled here.

**No phase is accepted or released by this document. No training, pilot result,
or test result is claimed.** P1A has a local documentation draft; integration,
immutable candidate commit, independent review, commit/push/PR evidence, and
phase acceptance are pending and owned by the orchestrator. Status of sibling
work must be verified there; absence of evidence here is not evidence of completion.

## 1. Rules and dependency map

```text
P1A protocol -------------------+
                               +--> P1 acceptance --> P2A roles/inventory
P1B subset fusion --> P1C targets+                       --> P2B stresses
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
                              optional later locked study (not yet authorized)
```

Within-phase dependencies are detailed below. **No task in the next phase may
start until the entire previous phase is integrated, reviewed, committed,
pushed, and its PR created or updated.** A local implementation or passing test
alone never opens the next phase. Refining boundaries must preserve this order
and accepted scientific scope; conflicts go to the orchestrator.

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
| P1A — protocol documentation | These two documents; ground definitions in existing README/package/source. No predecessor within P1. | Correct signed targets and score contract; role/test isolation; explicit numerical stresses, fair controls, metrics, kill rules, task map; documentation review at candidate commit. | Local draft; independent review/integration pending. |
| P1B — subset-fusion primitives | Named nonempty subsets using configured legacy fusion tree; empty-input unavailable contract. Uses P1A scientific contract; source ownership coordinated by orchestrator. | Red/green fixtures for all 15 nonempty masks, both dataset fusion trees, singleton/empty/invalid inputs, and full-view numerical/prediction equivalence. | Pending; no execution evidence recorded here. |
| P1C — intervention targets | Error/correctness and signed three-way removal targets with validity masks; depends on P1B and P1A. | Truth-table tests for repairs/damages/unchanged; identical predictions imply zero, wrong-to-different-wrong is zero; singleton no effect loss; no labels in inference features. | Pending. |
| P2A — patient roles and inventory | Deterministic stratified patient manifests, private test denylist/lock, inventory/duplicate audit and sanitized counts; depends on accepted P1. | Patient-disjoint roles; DDSM patient mapping verified; requested versus actual counts; missing/content-duplicate detection; no eager test outcome access; mismatch refusal. | Pending; raw inventory and patient mapping not inspected by P1A. |
| P2B — perturbations and masks | Reproducible parent/child transforms and sampling manifest; depends on P2A role schema and P1 subset contract. | Deterministic seeds; intensity/resolution defaults; held-out family/severity separation; common-mode control; all 14 proper clean masks; child retains exactly the parent's surviving tensors. | Pending. |
| P2C — provenance-bound artifacts | Frozen feature/target cache and role/checkpoint/mask/transform bindings; depends on P2A, P2B, P1C. | Tamper/stale/order/checkpoint/role rejection; all targets regenerated from realized inputs; external private records; prediction identity and finite-value checks. | Pending. |
| P3A — relation head and loss | 128-dimensional pooled views, two four-head relation layers, three-way effects and error BCE; depends on accepted P2. | Frozen encoder gradients/parameters; both backbone dimensions; missing-view masks; parent-normalized CE; singleton finite loss; lambda zero; unchanged classifier predictions. | Pending. |
| P3B — fair baselines and ablations | All controls and ablations in protocol section 6, including same-input MLP/four-class and regenerated-target augmentation controls; depends on P3A input/loss contracts. | Input/exposure/capacity audit; score orientation; original four-view MV-ACN equivalence with labeled subset adaptation; auxiliary classifier never replaces frozen predictions; ViLU adaptation documented. | Pending. |
| P4A — metrics and paired inference | AURC/tie policy, both AP orientations, risk at coverage, Brier, effects, patient-paired bootstrap and seed summaries; depends on accepted P3. | Hand-computable ranking/tie/sign fixtures; one-class AP handling; all variants retained within patient clusters; paired difference and seed/patient separation. | Pending. |
| P4B — training/evaluation/CLI | Explicit role access, fresh-classifier initialization path, confidence fitting, tune selection, immutable pilot plan and lock enforcement; depends on P4A and P2/P3 artifacts. | Forbidden-role negative tests; deterministic resume/provenance; regenerated targets; fresh public CLIP versus diagnostic checkpoint distinction; no test-selection bypass. | Pending. |
| P4C — smoke integration and costs | End-to-end synthetic tiny run plus external-resource preflight; depends on P4B. | Synthetic nonempty masks and corruptions flow through target/head/control/metrics; CLI/config sanity; CPU smoke feasible without downloading weights; latency/memory procedure ready. Smoke outputs labeled non-scientific. | Pending. |
| P5A — fresh classifier pilot prerequisite | Audit real external resources; fit classifier only on classifier-fit, select on tune, freeze checkpoint; depends on accepted P4. | Public pinned initialization, permitted-role exposure log, actual counts, checkpoint/manifest hashes, verified predictions; original checkpoint results diagnostic only. | Pending; no training launched by P1A. |
| P5B — confidence/control fitting and freeze | Fit candidate and mandatory controls with matched draws/budgets, tune only, freeze evaluation table; depends on P5A. | Actual run/seed completion; candidate/control selections and hashes; no pilot/test exposure; incomplete controls disclosed. | Pending. |
| P5C — measured pilot and go/no-go | Run frozen non-test pilot; report primary/clean/secondary endpoints, paired intervals, seed spread, cost and limitations; depends on P5B. | Actual measurements against all mandatory controls; apply 10% target/0.005 clean guardrail and validity rules; go/no-go/inconclusive justified without test inspection. | Pending; no measured pilot or success claim. |

Fresh classifier training is P5 execution; P2/P4 may build and verify the generic
data/artifact/training machinery using synthetic fixtures without starting that
study. P5A/P5B do not imply all datasets/backbones/seeds fit within the deadline.
Record the exact completed subset and retain the rest as pending.

## 3. Phase acceptance gates

| Gate | Required acceptance result | Evidence/status |
|---|---|---|
| G1: release P2 | P1A–P1C scientifically consistent; four-view legacy score/prediction compatibility and target edge cases verified; no source work inferred from documentation. | Pending. |
| G2: release P3 | P2 roles and patient grouping verified, actual counts reconciled or discrepancy escalated; lock/stress/provenance behavior reviewed; no fabricated inventory. | Pending. |
| G3: release P4 | Head/loss and all matched control interfaces tested; frozen predictions, singleton/mask handling, capacity/input fairness demonstrated. | Pending. |
| G4: release P5 | Full synthetic pipeline and metric checks pass; external resources and role permissions known; no real-data success inferred from smoke. | Pending. |
| G5: close pilot phase | P5 has measured outcomes with required controls or a truthful incomplete/blocker report; scientific go/no-go is separate from software acceptance. Tests remain locked absent explicit release. | Pending. |

For each task and gate, the orchestrator must append: owner, immutable candidate
commit, fresh reviewer identity/model, review disposition and fixed findings,
validation commands/results, integrated commit, pushed branch, PR URL/update,
and gate decision/time. All fields are currently **pending**; P1A has not created
a candidate commit, reviewed itself independently, pushed, or opened a PR.

## 4. P1A evidence and unresolved prerequisites

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
or scientific-behavior testing. Independent review remains pending.

Material prerequisites for the orchestrator:

| Item | Consequence / owner |
|---|---|
| Actual raw counts, patient mapping, duplicates, external weights and compute are unverified | P2/P5 must establish them. Counts in the protocol are targets; DDSM exam-level grouping alone is insufficient. |
| Existing code requires four views and eagerly resolves train/valid/test roles | P1/P2/P4 work is required before claiming subset support or safe locked-test operation. |
| Legacy confidence target generation and augmented fitting can disagree | P2 regenerated-target artifacts and P3 matched baselines are mandatory. Legacy reproduction alone does not validate the proposal. |
| Legacy ViT-L test selection and original checkpoint training exposure | Research entry points must reject these paths for held-out claims; checkpoint reuse is diagnostic only. |
| New numeric stress/head/training defaults are planned | Implement/review them consistently; scope conflicts go to orchestrator before changing the accepted design. |
| Candidate commit/review/phase history is not available in P1A | Orchestrator fills evidence fields. No phase is marked accepted here. |

There is no known contradiction with the accepted project scope in this draft.
Inventory disagreement or an implementation requiring changed classifier scores
would be a material conflict to escalate, not permission to silently alter the
protocol. No patient data was opened to resolve these questions in P1A.

## 5. Deadline, handoff, and retained work

Hard run deadline: **2026-09-10 20:50:09 UTC**. P1A target duration is at most
35 minutes. The orchestrator proceeds phase by phase within the five-hour run;
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
