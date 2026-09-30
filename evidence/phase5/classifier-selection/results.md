# RSNA classifier prerequisite results

The original 50-epoch ViT-B/32 classifier fit and tune-only selection are complete
and validated. Minimum tune NLL selected epoch 38 from all 50 checkpoints.
A separate selected-checkpoint pass reproduced its NLL exactly.

| Metric on 370 tune patients | Result |
| --- | ---: |
| Accuracy | 78.38% (290/370) |
| Macro-F1 | 0.6583 |
| Balanced accuracy | 0.6349 |
| Quadratic weighted kappa | 0.7812 |
| Multiclass NLL | 0.5234 |
| Multiclass Brier, summed over classes | 0.3086 |
| Majority-class prevalence | 42.97% |

| Density class | Patients | Recall |
| --- | ---: | ---: |
| A | 36 | 66.67% |
| B | 159 | 89.31% |
| C | 156 | 76.92% |
| D | 19 | 21.05% (4/19) |

These are selection-set diagnostics, not held-out estimates. Class D is poorly
recalled and has a small tune sample. No uncertainty-method improvement or
publishability claim follows from these classifier results. The full confusion
matrix and all 50 NLL values are in `tune-metrics.json`. All checkpoints and
patient-level data remain private. Pilot and locked test outcomes were not used.

The fit completed 53,500 updates on 3,209 patients. The resumed 35 epochs took
5 hours 53 minutes; tune selection took 517 seconds and the separately verified
metric pass took 321 seconds, including repeated provenance validation.

## Next scientific check

Before full confidence/control fitting, inspect the support for unchanged,
repair and damage targets on authorized confidence-fit data under the existing
clean and prescribed stress conditions. Report unique-patient support, views,
classes and clean versus stressed cases. A shared exam label does not imply
identical predictions across view subsets, but redundant views could make
nonzero targets scarce. These frequencies have not yet been measured.

The auxiliary target describes correctness change, not uncertainty change:
confidence may change without changing the predicted class, in which case the
effect target is zero. All subsets may also be consistently wrong. If effect
support is inadequate, do not assert that this mechanism is useful or change
targets/stresses silently to manufacture a positive result. Bring any proposed
methodological change to the user. The global error head and matched controls
must ultimately be evaluated separately on the frozen non-test pilot.
