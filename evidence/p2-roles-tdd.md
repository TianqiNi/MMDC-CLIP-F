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
exam-to-patient mapping. Neither dataset had an available identity-only locked-test
patient denylist/digest, and no custodian mapping declaration was available for the
RSNA study manifest. Consequently patient independence and test overlap remain
blocked for real fitting/evaluation even though generic software tests pass. See
`p2-inventory.json` for sanitized aggregate counts and exact audit coverage.

## Final validation

The final complete research suite passed: `55 passed in 0.62s`. Ruff lint returned
`All checks passed!`. Ruff initially requested formatting of the two new code files;
formatting was applied and the final format check passed. JSON parsing,
`git diff --check`, and owned-file scope checks also passed.
