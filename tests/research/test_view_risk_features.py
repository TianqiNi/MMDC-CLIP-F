from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import Tensor, nn

from mmdc_clip_f.backbones import BACKBONES, PROMPTS
from mmdc_clip_f.model import MultiViewCLIPClassifier
from mmdc_clip_f.research.view_risk.features import (
    LEGACY_IMAGE_MEAN,
    LEGACY_IMAGE_STD,
    load_verified_frozen_encoder,
)
from mmdc_clip_f.research.view_risk.fusion import DDSM_FUSION_PAIRS, RSNA_FUSION_PAIRS
from mmdc_clip_f.research.view_risk.perturbations import realize_parent


class FakeVision(nn.Module):
    def __init__(self, hidden_size: int, projection_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.weight = nn.Parameter(torch.linspace(0.2, 1.0, hidden_size))
        self.calls: list[Tensor] = []
        self.return_nonfinite = False

    def forward(self, *, pixel_values: Tensor) -> SimpleNamespace:
        self.calls.append(pixel_values.detach().clone())
        summary = pixel_values.mean(dim=(1, 2, 3), keepdim=False)
        pooled = summary[:, None] * self.weight[None, :]
        if self.return_nonfinite:
            pooled = pooled.clone()
            pooled[0, 0] = float("inf")
        hidden = torch.stack((pooled, pooled + 0.25, pooled - 0.5), dim=1)
        return SimpleNamespace(pooler_output=pooled, last_hidden_state=hidden)


class FakeCLIP(nn.Module):
    def __init__(self, hidden_size: int, projection_size: int = 6) -> None:
        super().__init__()
        self.vision_model = FakeVision(hidden_size, projection_size)
        self.visual_projection = nn.Linear(hidden_size, projection_size, bias=False)
        self.text_embedding = nn.Parameter(
            torch.arange(1, 1 + 4 * projection_size, dtype=torch.float32).reshape(4, -1)
        )
        self.logit_scale = nn.Parameter(torch.tensor(0.3))

    def get_text_features(self, *, input_ids: Tensor) -> Tensor:
        assert input_ids.shape[0] == 4
        return self.text_embedding

    def get_image_features(self, *, pixel_values: Tensor) -> Tensor:
        output = self.vision_model(pixel_values=pixel_values)
        return self.visual_projection(output.pooler_output)


@pytest.fixture(params=("vit_b_32", "vit_l_14_336"))
def verified_encoder(request: pytest.FixtureRequest, tmp_path):
    backbone = BACKBONES[request.param]
    clip = FakeCLIP(backbone.hidden_size)
    classifier = MultiViewCLIPClassifier(
        clip, ("L_CC", "L_MLO", "R_CC", "R_MLO"), RSNA_FUSION_PAIRS
    )
    checkpoint = tmp_path / f"{request.param}.safetensors"
    save_file(
        {name: value.detach().contiguous() for name, value in classifier.state_dict().items()},
        str(checkpoint),
    )
    encoder = load_verified_frozen_encoder(
        classifier,
        torch.arange(8, dtype=torch.long).reshape(4, 2),
        checkpoint,
        backbone=request.param,
        prompts=PROMPTS,
    )
    return encoder, clip, checkpoint


def _normalized_views(batch: int = 2) -> dict[str, Tensor]:
    return {
        name: torch.full((batch, 3, 2, 2), 0.1 + index * 0.2)
        for index, name in enumerate(("L_CC", "L_MLO", "R_CC", "R_MLO"))
    }


def test_observed_only_encoding_exposes_full_frozen_features(verified_encoder) -> None:
    encoder, clip, _ = verified_encoder
    before = {
        name: value.detach().clone() for name, value in encoder.classifier.state_dict().items()
    }

    features = encoder.extract_normalized(
        {"L_CC": _normalized_views()["L_CC"], "R_MLO": _normalized_views()["R_MLO"]}
    )

    assert features.observed_views == ("L_CC", "R_MLO")
    assert clip.vision_model.calls[-1].shape[0] == 4  # two views times batch two
    assert set(features.logits_by_view) == {"L_CC", "R_MLO"}
    assert features.hidden_by_view["L_CC"].shape == (2, 3, encoder.identity.hidden_size)
    assert features.projected_by_view["R_MLO"].shape == (2, 6)
    assert features.normalized_text_embeddings.shape == (4, 6)
    assert torch.allclose(features.normalized_text_embeddings.norm(dim=1), torch.ones(4))
    assert all(not tensor.requires_grad for tensor in features.all_tensors())
    assert all(parameter.requires_grad is False for parameter in encoder.classifier.parameters())
    assert encoder.classifier.training is False
    assert all(
        torch.equal(value, before[name]) for name, value in encoder.classifier.state_dict().items()
    )


@pytest.mark.parametrize("fusion_pairs", (RSNA_FUSION_PAIRS, DDSM_FUSION_PAIRS))
def test_full_path_matches_existing_classifier_exactly(tmp_path, fusion_pairs) -> None:
    clip = FakeCLIP(BACKBONES["vit_b_32"].hidden_size)
    classifier = MultiViewCLIPClassifier(clip, ("L_CC", "L_MLO", "R_CC", "R_MLO"), fusion_pairs)
    checkpoint = tmp_path / "weights.safetensors"
    save_file(
        {name: value.detach().contiguous() for name, value in classifier.state_dict().items()},
        str(checkpoint),
    )
    input_ids = torch.arange(8, dtype=torch.long).reshape(4, 2)
    encoder = load_verified_frozen_encoder(
        classifier, input_ids, checkpoint, backbone="vit_b_32", prompts=PROMPTS
    )
    views = _normalized_views()

    extracted = encoder.extract_normalized(views)
    with torch.no_grad():
        legacy_scores = classifier(views, input_ids)

    assert torch.allclose(extracted.scores, legacy_scores, atol=1e-6, rtol=1e-6)
    assert torch.equal(extracted.prediction, legacy_scores.argmax(dim=1))


def test_realized_parent_is_normalized_exactly_once(verified_encoder) -> None:
    encoder, clip, _ = verified_encoder
    raw = {
        "L_CC": torch.full((3, 2, 2), 0.5),
        "R_CC": torch.full((3, 2, 2), 0.25),
    }
    parent = realize_parent(raw, raw)

    features = encoder.extract_realized_parent(parent)

    expected = torch.stack(
        [
            (raw[view] - torch.tensor(LEGACY_IMAGE_MEAN)[:, None, None])
            / torch.tensor(LEGACY_IMAGE_STD)[:, None, None]
            for view in features.observed_views
        ]
    )
    assert torch.allclose(clip.vision_model.calls[-1], expected)
    assert features.batch_size == 1


def test_omission_reuses_exact_parent_features_without_encoding(verified_encoder) -> None:
    encoder, clip, _ = verified_encoder
    parent = encoder.extract_normalized(_normalized_views())
    calls = len(clip.vision_model.calls)

    child = encoder.omit_parent_view(parent, "L_MLO")

    assert len(clip.vision_model.calls) == calls
    assert child.observed_views == ("L_CC", "R_CC", "R_MLO")
    for view in child.observed_views:
        assert child.hidden_by_view[view] is parent.hidden_by_view[view]
        assert child.projected_by_view[view] is parent.projected_by_view[view]
        assert child.logits_by_view[view] is parent.logits_by_view[view]


def test_realized_parent_rejects_already_normalized_or_nonfinite_pixels(verified_encoder) -> None:
    encoder, _, _ = verified_encoder
    for invalid in (torch.full((3, 2, 2), -0.1), torch.full((3, 2, 2), float("nan"))):
        with pytest.raises(ValueError, match=r"finite|\[0,? ?1\]"):
            encoder.extract_realized_parent(realize_parent({"L_CC": invalid}, ("L_CC",)))


def test_nonfinite_or_missing_encoder_outputs_are_rejected(verified_encoder) -> None:
    encoder, clip, _ = verified_encoder
    clip.vision_model.return_nonfinite = True
    with pytest.raises(ValueError, match="finite"):
        encoder.extract_normalized({"L_CC": _normalized_views()["L_CC"]})


def test_weight_mutation_is_refused_before_encoding(verified_encoder) -> None:
    encoder, clip, _ = verified_encoder
    calls = len(clip.vision_model.calls)
    with torch.no_grad():
        clip.visual_projection.weight.add_(1)
    with pytest.raises(RuntimeError, match="weights changed"):
        encoder.extract_normalized({"L_CC": _normalized_views()["L_CC"]})
    assert len(clip.vision_model.calls) == calls


def test_checkpoint_file_is_loaded_strictly_and_bound_to_encoder(verified_encoder) -> None:
    encoder, _, checkpoint = verified_encoder
    import hashlib

    assert encoder.identity.checkpoint_sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert encoder.identity.text_input_sha256
    assert encoder.identity.prompt_order == PROMPTS
