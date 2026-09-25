# Recorded test commands

Long traces are truncated; synthetic fixtures only.

## p3-head-review1

`/bin/bash -lc 'PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research/test_view_risk_head_inputs.py tests/research/test_view_risk_head.py tests/research/test_view_risk_review_regressions.py'`

Exit 0

```text
......................................                                   [100%]
38 passed in 0.84s

```

## p3-head-review1

`/bin/bash -lc 'PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q'`

Exit 0

```text
........................................................................ [ 38%]
........................................................................ [ 76%]
.............................................                            [100%]
189 passed in 1.41s

```

## p3-head-review2

`/bin/bash -lc 'PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research/test_view_risk_head.py::test_singleton_corruption_supervises_its_only_observed_view tests/research/test_view_risk_head.py::test_mixed_masks_average_each_parents_auxiliary_mean_equally'`

Exit 0

```text
..                                                                       [100%]
2 passed in 0.60s

```

## p3-head-review2

`/bin/bash -lc 'PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research/test_view_risk_head.py::test_singleton_corruption_supervises_its_only_observed_view'`

Exit 1

```text
   RelationAwareHeadConfig(backbone="vit_b_32", auxiliary_task="corruption")
        ).eval()
        output = model(raw)
        output.raw_auxiliary_logits.retain_grad()
        output.effect_representations.retain_grad()
        corruption_targets = torch.tensor([[-1, -1, 1, -1], [-1, -1, 0, -1]], dtype=torch.long)
    
        assert not raw.removal_valid_mask.any()
>       assert torch.equal(output.auxiliary_valid_mask, raw.observed_mask)
E       AssertionError: assert False
E        +  where False = <built-in method equal of type object at 0x75cd99b957c0>(tensor([[False, False, False, False],\n        [False, False, False, False]]), tensor([[False, False,  True, False],\n        [False, False,  True, False]]))
E        +    where <built-in method equal of type object at 0x75cd99b957c0> = torch.equal
E        +    and   tensor([[False, False, False, False],\n        [False, False, False, False]]) = RelationAwareHeadOutput(error_logit=tensor([ 0.2705, -0.0170], grad_fn=<SqueezeBackward1>), confidence=tensor([0.4328,...lse, False],\n        [False, False, False, False]]), auxiliary_task='corruption', classifier_prediction=tensor([0, 2])).auxiliary_valid_mask
E        +    and   tensor([[False, False,  True, False],\n        [False, False,  True, False]]) = RawHeadInputs(pooled_tokens=tensor([[[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],\n         [ 0.0000, ...licy='exclude_cls', fusion_pairs=(('L_CC', 'L_MLO'), ('R_CC', 'R_MLO')), view_order=('L_CC', 'L_MLO', 'R_CC', 'R_MLO')).observed_mask

tests/research/test_view_risk_head.py:183: AssertionError
=========================== short test summary info ============================
FAILED tests/research/test_view_risk_head.py::test_singleton_corruption_supervises_its_only_observed_view
1 failed in 0.58s

```

## p3-head-review2

`/bin/bash -lc 'PYTHONPATH=src /home/tianqini/research/MMDC-CLIP-IVR/.venv/bin/python -m pytest -q tests/research/test_view_risk_head.py::test_mixed_masks_average_each_parents_auxiliary_mean_equally'`

Exit 1

```text
tensor([[0, 2, 1, -1], [1, 0, -1, -1], [-1, -1, -1, -1]])
        targets = InterventionTargets(
            observed_views=CANONICAL_VIEWS,
            observed_prediction=torch.tensor([0, 1, 2]),
            observed_error=torch.tensor([0, 1, 0]),
            omission_predictions=torch.full((3, 4), -1),
            omission_effects=torch.where(valid, labels - 1, labels),
            omission_labels=labels,
            valid_removal_mask=valid,
            fusion_pairs=RSNA_FUSION_PAIRS,
        )
    
        result = compute_view_risk_loss(output, targets)
        parent_0 = torch.stack(
            [
                F.cross_entropy(raw_logits[0, slot : slot + 1], labels[0, slot : slot + 1])
                for slot in range(3)
            ]
        ).mean()
        parent_1 = torch.stack(
            [
                F.cross_entropy(raw_logits[1, slot : slot + 1], labels[1, slot : slot + 1])
                for slot in range(2)
            ]
        ).mean()
        parent_2 = raw_logits.new_zeros(())
        expected_per_parent = torch.stack((parent_0, parent_1, parent_2))
    
        torch.testing.assert_close(result.per_parent_auxiliary_ce, expected_per_parent)
>       torch.testing.assert_close(result.auxiliary_ce, expected_per_parent.mean())
E       AssertionError: Scalars are not close!
E       
E       Expected 0.2435988187789917 but got 0.3908577263355255.
E       Absolute difference: 0.1472589075565338 (up to 1e-05 allowed)
E       Relative difference: 0.6045140460641414 (up to 1.3e-06 allowed)

tests/research/test_view_risk_head.py:306: AssertionError
=========================== short test summary info ============================
FAILED tests/research/test_view_risk_head.py::test_mixed_masks_average_each_parents_auxiliary_mean_equally
1 failed in 0.57s

```
