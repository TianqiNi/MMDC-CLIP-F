# P2A patient roles and inventory TDD evidence

Scope was limited to the P2A module, its behavioral tests, and sanitized aggregate
inventory evidence. No training ran. No original-test manifest, outcome, or image
was opened. No private identifier, image path, patient-level record, prediction,
or weight is recorded here.

## Red

Tests were written before the implementation and run with the required environment:

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research/test_view_risk_roles.py
```

Expected result: pytest exit 2 during collection, with one error:
`ModuleNotFoundError: mmdc_clip_f.research.view_risk.roles`. This was the intended
missing-interface RED, not a dependency failure.

## Green

The same scoped command passed after implementation: `14 passed in 0.56s`.

The new cases establish exact requested counts for feasible single-exam groups;
deterministic, row-order-independent seeded assignment; majority-density patient
strata with highest-density tie breaking; preservation and reporting of repeated
multi-exam patient groups; actual-versus-requested counts when indivisible groups
make targets unattainable; configurable proportions; and sanitized class, patient,
and group summaries. They also exercise refusal of absent mapping declarations,
invalid labels/views/provenance, duplicate exams, patient overlap, cross-role image
ID/content collisions, forbidden or locked role access before a loader runs, paths
outside a declared private root, and stale/tampered manifests.

A bounded synthetic property check covered fixture sizes 4 through 79 and reported
`single-exam exact-target property: 76 cases passed`.

A final provenance regression was then added before its source change. RED was one
behavioral failure (`DID NOT RAISE`) because two manifests in the same dataset
namespace could carry conflicting source hashes. After binding each namespace to
one source-hash set and mapping declaration, GREEN was `15 passed in 0.56s`.

Regression and style commands:

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m ruff check src/mmdc_clip_f/research/view_risk/roles.py tests/research/test_view_risk_roles.py
```

An intermediate full-suite result before the proportions case was added was
`53 passed in 0.63s` with `All checks passed!`.

## Aggregate local preflight

Only the named train/validation metadata and legacy path-building code/configuration
were inspected. The four non-test counts match the protocol totals: RSNA 4,937 and
DDSM 1,353. All expected named-view paths were present under both relevant legacy
layouts. This was a path-existence check; no image was opened or decoded, and no
content hashes were computed.

No real patient role split was derived. DDSM lacks the required verified
exam-to-patient mapping. RSNA's official schema identifies `patient_id`; no external
custodian signoff is required, and an authorized researcher can record the
schema-derived provenance declaration. That local binding was not executed here.
Neither dataset had an available identity-only locked-test patient denylist/digest,
so test overlap remains unaudited; content duplication also remains unaudited. See
`p2-inventory.json` for sanitized aggregate counts and exact audit coverage.

## Final validation

The final complete research suite passed: `55 passed in 0.62s`. Ruff lint returned
`All checks passed!`. Ruff initially requested formatting of the two new code files;
formatting was applied and the final format check passed. JSON parsing,
`git diff --check`, and owned-file scope checks also passed.

## Review 1 fix cycle

Fresh review of candidate `2b63cf5` found that ordinary manifests could expose
locked-test outcomes/paths before the callback guard, cross-manifest reuse of a
same-namespace image path was not checked, and the RSNA evidence incorrectly
required custodian signoff.

Direct regressions were added before source changes. RED was `3 failed, 15 passed`:
ordinary locked-record construction returned normally, a self-consistently rehashed
locked manifest loaded normally, and a same-namespace path reused across
classifier-fit and pilot manifests passed inventory validation. GREEN after the
fixes was `19 passed in 0.56s`. A defense-in-depth case also verifies that a
post-construction tampered manifest is refused by ordinary save without creating a
file. Final GREEN was `20 passed in 0.56s`; the complete research suite passed with
`60 passed in 0.61s`; the final full repository rerun passed `60 passed in 0.62s`.

Ordinary record construction and private save/load paths now fail closed on
outcome/image-bearing `locked_test` records. P2 exposes only an identity-digest
denylist and sanitized overlap check for locked patients; it contains no outcome,
image, generic boolean bypass, or test-release path. Cross-manifest image paths are
now scoped by dataset namespace and rejected on reuse. RSNA readiness language was
corrected as described above; the locked-identity and content-audit blockers remain.
