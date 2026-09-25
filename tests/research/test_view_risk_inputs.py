from __future__ import annotations

import pytest
import torch

from mmdc_clip_f.research.view_risk import CANONICAL_VIEWS, InferenceViewInputs


def test_inference_inputs_are_named_canonical_and_do_not_require_representations() -> None:
    right = torch.zeros(2, 4, dtype=torch.float64)
    left = torch.ones(2, 4, dtype=torch.float64)
    source = {"R_MLO": right, "L_CC": left}

    inputs = InferenceViewInputs(logits_by_view=source)
    source["L_MLO"] = torch.full((2, 4), 2.0, dtype=torch.float64)

    assert inputs.available_views == ("L_CC", "R_MLO")
    assert tuple(inputs.logits_by_view) == ("L_CC", "R_MLO")
    assert inputs.frozen_representations is None
    assert "L_MLO" not in inputs.logits_by_view


def test_optional_representations_must_be_frozen_and_match_available_views() -> None:
    logits = {"L_CC": torch.zeros(2, 4)}
    frozen = {"L_CC": torch.zeros(2, 3, 8)}

    inputs = InferenceViewInputs(logits, frozen_representations=frozen)

    assert inputs.frozen_representations is not None
    assert inputs.frozen_representations["L_CC"] is frozen["L_CC"]

    with pytest.raises(ValueError, match="frozen"):
        InferenceViewInputs(
            logits,
            frozen_representations={"L_CC": torch.zeros(2, 3, 8, requires_grad=True)},
        )
    with pytest.raises(ValueError, match="same view names"):
        InferenceViewInputs(logits, frozen_representations={"L_MLO": torch.zeros(2, 3, 8)})


def test_inference_contract_has_no_label_or_target_channel() -> None:
    logits = {CANONICAL_VIEWS[0]: torch.zeros(1, 4)}

    with pytest.raises(TypeError):
        InferenceViewInputs(logits_by_view=logits, labels=torch.tensor([0]))  # type: ignore[call-arg]
