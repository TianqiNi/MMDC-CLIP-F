# P3A relation-head TDD evidence

Scope: generic software behavior on small synthetic tensors only.  No classifier
or confidence fitting, pretrained download, patient data, row prediction,
checkpoint, or uncertainty-improvement measurement was performed.

## RED

After writing the two P3A behavioral suites and before adding either source
module:

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q \
  tests/research/test_view_risk_head_inputs.py tests/research/test_view_risk_head.py

2 collection errors: ModuleNotFoundError for view_risk.head_inputs and
view_risk.head; execution stopped as intended.
```

The tests exercise all 15 nonempty masks under both 768/1024 hidden widths and
both accepted fusion trees, exact child recomputation, backbone token policy,
canonical named slots, source detachment/prediction preservation, inference
target refusals, absent-slot invariance, relation architecture, effect-to-risk
gradients at lambda zero, exact signed/magnitude reporting constraints, raw
parent-normalized CE, singleton loss, and one-factor input/auxiliary ablations.

## GREEN

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q \
  tests/research/test_view_risk_head_inputs.py tests/research/test_view_risk_head.py
25 passed in 0.79s

PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q
189 passed in 1.33s

PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m ruff check \
  src/mmdc_clip_f/research/view_risk/head.py \
  src/mmdc_clip_f/research/view_risk/head_inputs.py \
  tests/research/test_view_risk_head.py \
  tests/research/test_view_risk_head_inputs.py
All checks passed.
```

## Review-1 regression coverage follow-up

The accepted implementation already passed both new tests.  The following RED
results came from deliberate, temporary local mutations to prove the new tests'
sensitivity; they are not preexisting implementation failures.

1. Corruption-mask mutation: in the corruption branch only, replaced
   `prepared.observed_mask` with `prepared.removal_valid_mask`, then ran:

   ```text
   PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q \
     tests/research/test_view_risk_head.py::test_singleton_corruption_supervises_its_only_observed_view

   1 failed: the singleton produced no auxiliary-valid slot instead of supervising
   its sole observed `R_CC` view.
   ```

2. Loss-normalization mutation: replaced
   `auxiliary_ce = per_parent_auxiliary.mean()` with global valid-slot averaging
   `auxiliary_ce = per_slot_ce[valid].mean()`, then ran:

   ```text
   PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q \
     tests/research/test_view_risk_head.py::test_mixed_masks_average_each_parents_auxiliary_mean_equally

   1 failed: global-slot CE was 0.3908577 versus the hand-computed
   mean-of-parent-means 0.2435988 for valid-removal counts `[3,2,0]`.
   ```

After each RED run, `head.py` was restored from the accepted copy and compared
byte-for-byte.  Both files had SHA-256
`46bd7ff775d99204e43d95170b507fceac788dd97a2413c4f1601a28532693bf`.
No source change remains.

Final GREEN:

```text
PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q \
  tests/research/test_view_risk_head_inputs.py tests/research/test_view_risk_head.py
27 passed in 0.79s

PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q
191 passed in 1.34s

PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m ruff check \
  src/mmdc_clip_f/research/view_risk/head.py \
  src/mmdc_clip_f/research/view_risk/head_inputs.py \
  tests/research/test_view_risk_head.py \
  tests/research/test_view_risk_head_inputs.py
All checks passed.
```

The singleton corruption regression verifies positive raw auxiliary CE and
nonzero CE gradients at its sole observed view (including the learned auxiliary
representation), while all absent slots have zero gradients and removal validity
is empty.  The mixed-mask regression derives counts `[3,2,0]` from triple, pair,
and singleton observed masks, hand-computes each parent's mean raw CE (zero for
the singleton), and then averages the three parents equally.

## Public contract and counts

`prepare_raw_head_inputs(FrozenViewFeatures, *, backbone) -> RawHeadInputs`
has no label/target/corruption/TCP/clean-counterpart argument.  Canonical slot
order is `L_CC,L_MLO,R_CC,R_MLO`.  Slot tensors are pooled hidden tokens,
per-class view evidence, view vacuity, signed parent-minus-child class-score
differences, masks, and prediction/agreement metadata.  Global order is current
scores `[class 0..3]`, probabilities `[0..3]`, fused evidence `[0..3]`, then
fused vacuity.  ViT-B/32 excludes CLS before token mean; ViT-L/14-336 includes
CLS.  The default scaler is stateless identity.

`RelationAwareConfidenceHead(config)(raw) -> RelationAwareHeadOutput` returns
`error_logit [B]`, `confidence [B]`, frozen `classifier_prediction [B]`, learned
effect representations `[B,4,128]`, raw auxiliary logits `[B,4,3]` for signed
or `[B,4,2]` for magnitude/corruption, constrained probabilities of the same
shape, and `[B,4]` validity.  `compute_view_risk_loss(output, targets, *,
auxiliary_weight=1, corruption_targets=None)` returns scalar total/BCE/auxiliary
CE and both `[B]` parent components.

Trainable scalar counts (ViT-B unless noted):

| Setting | Count | Delta from default ViT-B |
|---|---:|---:|
| default ViT-B/32 | 433,436 | 0 |
| default ViT-L/14-336 | 466,204 | +32,768 |
| no omission features | 432,924 | -512 |
| no relation biases | 433,412 | -24 |
| magnitude auxiliary | 433,307 | -129 |
| corruption auxiliary | 433,307 | -129 |
| hidden-only | 430,492 | -2,944 |
| evidence-only | 335,004 | -98,432 |

Counts describe software capacity, not empirical uncertainty quality.
