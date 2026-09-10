from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest
import torch

from mmdc_clip_f.research.view_risk.perturbations import (
    CANONICAL_VIEWS,
    PerturbationSpec,
    TrainingSampleSpec,
    apply_perturbation,
    derive_realization_seed,
    enumerate_evaluation_panels,
    enumerate_proper_masks,
    enumerate_training_support,
    omit_parent_view,
    realize_parent,
    resolve_parameters,
    sample_training_spec,
    validate_training_perturbation,
)


def _images(size: int = 8) -> dict[str, torch.Tensor]:
    return {
        view: torch.full((3, size, size), 0.2 + 0.1 * index)
        for index, view in enumerate(CANONICAL_VIEWS)
    }


def test_clean_identity_is_exact_and_input_contract_rejects_invalid_pixels() -> None:
    image = torch.linspace(0.0, 1.0, 3 * 4 * 4).reshape(3, 4, 4)
    clean = PerturbationSpec("clean")

    output, metadata = apply_perturbation(image, clean)

    assert output is image
    assert metadata.spec == clean
    assert metadata.parameters == ()

    with pytest.raises(ValueError, match=r"\[0,1\]"):
        apply_perturbation(torch.full((3, 4, 4), 1.01), clean)
    with pytest.raises(ValueError, match="RGB"):
        apply_perturbation(torch.zeros(1, 4, 4), clean)
    with pytest.raises(TypeError, match="floating"):
        apply_perturbation(torch.zeros(3, 4, 4, dtype=torch.uint8), clean)


def test_contrast_and_brightness_follow_analytic_intensity_operations() -> None:
    base = torch.tensor([[0.0, 0.2], [0.6, 1.0]])
    image = torch.stack((base, base * 0.5, base * 0.25))

    contrast, _ = apply_perturbation(
        image,
        PerturbationSpec("contrast", "moderate"),
    )
    darker, _ = apply_perturbation(
        image,
        PerturbationSpec("brightness", "moderate", variant="lower"),
    )
    brighter, _ = apply_perturbation(
        image,
        PerturbationSpec("brightness", "moderate", variant="upper"),
    )

    expected_contrast = image.mean() + 0.6 * (image - image.mean())
    assert torch.allclose(contrast, expected_contrast)
    assert torch.allclose(darker, image * 0.75)
    assert torch.allclose(brighter, (image * 1.25).clamp(0.0, 1.0))
    assert float(brighter.min()) >= 0.0 and float(brighter.max()) <= 1.0


def test_noise_uses_one_rgb_coupled_field_and_replays_without_global_rng() -> None:
    image = torch.full((3, 16, 16), 0.5)
    spec = PerturbationSpec("gaussian_noise", "mild", realization_seed=917)

    first, _ = apply_perturbation(image, spec)
    torch.manual_seed(999_999)
    _ = torch.randn(100)
    random.seed(123)
    second, _ = apply_perturbation(image, spec)
    different, _ = apply_perturbation(
        image,
        PerturbationSpec("gaussian_noise", "mild", realization_seed=918),
    )

    assert torch.equal(first, second)
    assert torch.equal(first[0] - image[0], first[1] - image[1])
    assert torch.equal(first[1] - image[1], first[2] - image[2])
    assert not torch.equal(first, different)


@pytest.mark.parametrize(
    ("resolution", "severity", "sigma", "kernel", "crop_side", "motion_length"),
    [
        (224, "mild", 0.7, 7, 212, 5),
        (224, "moderate", 1.4, 11, 193, 9),
        (224, "strong", 2.8, 19, 173, 17),
        (336, "mild", 1.05, 9, 318, 7),
        (336, "moderate", 2.10, 15, 290, 13),
        (336, "strong", 4.20, 27, 260, 25),
    ],
)
def test_resolution_scaled_blur_crop_and_motion_parameters(
    resolution: int,
    severity: str,
    sigma: float,
    kernel: int,
    crop_side: int,
    motion_length: int,
) -> None:
    blur = resolve_parameters(PerturbationSpec("gaussian_blur", severity), resolution)
    crop = resolve_parameters(PerturbationSpec("crop", severity), resolution)
    motion = resolve_parameters(
        PerturbationSpec("motion_blur", severity, variant="horizontal"),
        resolution,
    )
    vertical_motion = resolve_parameters(
        PerturbationSpec("motion_blur", severity, variant="vertical"),
        resolution,
    )

    assert blur == {"sigma": pytest.approx(sigma), "kernel_size": kernel}
    assert crop["crop_side"] == crop_side
    assert motion == {"length": motion_length, "orientation": "horizontal"}
    assert vertical_motion == {"length": motion_length, "orientation": "vertical"}


@pytest.mark.parametrize(
    ("severity", "noise", "contrast", "lower", "upper", "area"),
    [
        ("mild", 0.02, 0.80, 0.90, 1.10, 0.90),
        ("moderate", 0.05, 0.60, 0.75, 1.25, 0.75),
        ("strong", 0.10, 0.40, 0.60, 1.40, 0.60),
    ],
)
def test_protocol_intensity_and_area_parameters(
    severity: str,
    noise: float,
    contrast: float,
    lower: float,
    upper: float,
    area: float,
) -> None:
    assert resolve_parameters(PerturbationSpec("gaussian_noise", severity), 224) == {"sigma": noise}
    assert resolve_parameters(PerturbationSpec("contrast", severity), 224) == {"factor": contrast}
    assert (
        resolve_parameters(PerturbationSpec("brightness", severity, variant="lower"), 224)["factor"]
        == lower
    )
    assert (
        resolve_parameters(PerturbationSpec("brightness", severity, variant="upper"), 224)["factor"]
        == upper
    )
    assert resolve_parameters(PerturbationSpec("crop", severity), 224)["retained_area"] == area


def test_blurs_and_crop_preserve_shape_range_and_motion_orientation() -> None:
    image = torch.zeros(3, 224, 224)
    image[:, 112, 112] = 1.0

    gaussian, _ = apply_perturbation(
        image,
        PerturbationSpec("gaussian_blur", "mild"),
    )
    horizontal, _ = apply_perturbation(
        image,
        PerturbationSpec("motion_blur", "mild", variant="horizontal"),
    )
    vertical, _ = apply_perturbation(
        image,
        PerturbationSpec("motion_blur", "mild", variant="vertical"),
    )
    cropped, metadata = apply_perturbation(
        image,
        PerturbationSpec("crop", "mild", realization_seed=8),
    )

    for stressed in (gaussian, horizontal, vertical, cropped):
        assert stressed.shape == image.shape
        assert torch.isfinite(stressed).all()
        assert float(stressed.min()) >= 0.0 and float(stressed.max()) <= 1.0
    assert torch.count_nonzero(horizontal[:, 112, :]) > 1
    assert torch.count_nonzero(horizontal[:, :, 112]) == 3
    assert torch.count_nonzero(vertical[:, :, 112]) > 1
    assert torch.count_nonzero(vertical[:, 112, :]) == 3
    assert metadata.parameter_dict()["crop_side"] == 212

    ramp = torch.arange(224, dtype=torch.float32).div(223).repeat(3, 224, 1)
    reflected, _ = apply_perturbation(
        ramp,
        PerturbationSpec("motion_blur", "mild", variant="horizontal"),
    )
    assert reflected[0, 100, 0].item() == pytest.approx(6.0 / (5.0 * 223.0))


def test_parent_is_realized_once_and_omission_reuses_exact_surviving_tensors() -> None:
    images = _images()
    parent = realize_parent(
        images,
        CANONICAL_VIEWS,
        perturbations={"L_CC": PerturbationSpec("gaussian_noise", "mild", realization_seed=31)},
    )

    child = omit_parent_view(parent, "R_MLO")

    assert tuple(parent.images) == CANONICAL_VIEWS
    assert tuple(child.images) == CANONICAL_VIEWS[:-1]
    assert all(parent.images[view] is images[view] for view in CANONICAL_VIEWS[1:])
    assert all(child.images[view] is parent.images[view] for view in child.images)
    assert "R_MLO" not in child.metadata
    with pytest.raises(ValueError, match="nonempty"):
        omit_parent_view(realize_parent(images, ("L_CC",)), "L_CC")


def test_common_mode_shares_realization_and_absent_views_are_never_materialized() -> None:
    images = _images()
    images["R_CC"] = images["L_CC"].clone()
    observed = ("L_CC", "R_CC")
    common = PerturbationSpec("gaussian_noise", "moderate", realization_seed=77)

    parent = realize_parent(images, observed, common_mode=common)

    assert torch.equal(parent.images["L_CC"], parent.images["R_CC"])
    assert tuple(parent.images) == observed
    assert "L_MLO" not in parent.images and "R_MLO" not in parent.images
    with pytest.raises(ValueError, match="absent"):
        realize_parent(
            images,
            observed,
            perturbations={"L_MLO": PerturbationSpec("gaussian_noise", "mild")},
        )


def test_named_parent_contract_refuses_empty_unknown_and_conflicting_stress_modes() -> None:
    images = _images()
    with pytest.raises(ValueError, match="at least one"):
        realize_parent(images, ())
    with pytest.raises(ValueError, match="unknown"):
        realize_parent({**images, "UNKNOWN": images["L_CC"]}, CANONICAL_VIEWS)
    with pytest.raises(ValueError, match="either"):
        realize_parent(
            images,
            CANONICAL_VIEWS,
            perturbations={"L_CC": PerturbationSpec("gaussian_noise", "mild")},
            common_mode=PerturbationSpec("gaussian_noise", "mild"),
        )


def test_specs_are_immutable_json_roundtrippable_and_private_seed_is_stable() -> None:
    private_key = "synthetic-private-key-never-returned"
    seed = derive_realization_seed(private_key, "primary/brightness/mild/L_CC/lower", 4242)
    replay = derive_realization_seed(private_key, "primary/brightness/mild/L_CC/lower", 4242)
    different_parent = derive_realization_seed(
        "different-synthetic-private-key",
        "primary/brightness/mild/L_CC/lower",
        4242,
    )
    spec = PerturbationSpec("brightness", "mild", variant="lower", realization_seed=seed)

    encoded = json.dumps(spec.to_dict(), sort_keys=True)

    assert seed == 4_454_856_240_536_416_292
    assert seed == replay
    assert seed != different_parent
    assert PerturbationSpec.from_dict(json.loads(encoded)) == spec
    assert private_key not in encoded and private_key not in repr(spec)
    with pytest.raises(FrozenInstanceError):
        spec.severity = "strong"  # type: ignore[misc]


def test_all_fourteen_named_proper_masks_are_enumerated_once() -> None:
    masks = enumerate_proper_masks()

    assert len(masks) == len(set(masks)) == 14
    assert {len(mask) for mask in masks} == {1, 2, 3}
    assert all(0 < len(mask) < 4 for mask in masks)
    assert all(tuple(view for view in CANONICAL_VIEWS if view in mask) == mask for mask in masks)


def test_training_support_has_exact_strata_masks_stresses_and_target_weights() -> None:
    support = enumerate_training_support()
    probability_by_stratum: dict[str, Fraction] = defaultdict(Fraction)
    masks_by_stratum: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    stress_pairs: set[tuple[str, str]] = set()
    mode_probability: dict[str, Fraction] = defaultdict(Fraction)
    mode_by_stratum: dict[tuple[str, str], Fraction] = defaultdict(Fraction)
    mask_probability: dict[tuple[str, tuple[str, ...]], Fraction] = defaultdict(Fraction)
    pair_probability: dict[tuple[str, str, str], Fraction] = defaultdict(Fraction)

    for case in support:
        probability_by_stratum[case.stratum] += case.probability
        masks_by_stratum[case.stratum].add(case.observed_views)
        if "proper_mask" in case.stratum:
            mask_probability[(case.stratum, case.observed_views)] += case.probability
        if case.family is not None:
            stress_pairs.add((case.family, case.severity))
            mode_probability[case.stress_mode] += case.probability
            mode_by_stratum[(case.stratum, case.stress_mode)] += case.probability
            pair_probability[(case.stratum, case.family, case.severity)] += case.probability

    assert probability_by_stratum == {
        "clean4": Fraction(1, 4),
        "clean_proper_mask": Fraction(1, 4),
        "stressed4": Fraction(1, 4),
        "stressed_proper_mask": Fraction(1, 4),
    }
    assert masks_by_stratum["clean_proper_mask"] == set(enumerate_proper_masks())
    assert masks_by_stratum["stressed_proper_mask"] == set(enumerate_proper_masks())
    assert stress_pairs == {
        ("gaussian_noise", "mild"),
        ("gaussian_noise", "moderate"),
        ("gaussian_blur", "mild"),
        ("gaussian_blur", "moderate"),
    }
    assert mode_probability == {"single": Fraction(2, 5), "common": Fraction(1, 10)}
    assert set(mask_probability.values()) == {Fraction(1, 56)}
    assert set(pair_probability.values()) == {Fraction(1, 16)}
    assert mode_by_stratum == {
        ("stressed4", "single"): Fraction(1, 5),
        ("stressed4", "common"): Fraction(1, 20),
        ("stressed_proper_mask", "single"): Fraction(1, 5),
        ("stressed_proper_mask", "common"): Fraction(1, 20),
    }


def test_seeded_training_draws_are_replayable_and_only_stress_observed_views() -> None:
    first = [sample_training_spec("synthetic-parent", f"draw-{index}", 42) for index in range(12)]
    torch.manual_seed(17)
    _ = torch.rand(40)
    replay = [sample_training_spec("synthetic-parent", f"draw-{index}", 42) for index in range(12)]
    other = [
        sample_training_spec("another-synthetic-parent", f"draw-{index}", 42) for index in range(12)
    ]

    assert first == replay
    assert first != other
    assert [draw.stratum for draw in first[:4]] == [
        "stressed4",
        "stressed_proper_mask",
        "clean_proper_mask",
        "clean4",
    ]
    assert first[0].stressed_views == ("R_MLO",)
    assert first[1].observed_views == ("L_CC", "L_MLO", "R_CC")
    assert first[1].stressed_views == ("L_CC",)
    assert TrainingSampleSpec.from_dict(json.loads(json.dumps(first[0].to_dict()))) == first[0]
    assert all(set(draw.stressed_views) <= set(draw.observed_views) for draw in first)
    assert all(
        draw.perturbation is None or draw.perturbation.family in {"gaussian_noise", "gaussian_blur"}
        for draw in first
    )


@pytest.mark.parametrize("operation", ["confidence-fit", "tune"])
def test_fit_and_tune_refuse_held_out_families_and_strong_seen_stress(
    operation: str,
) -> None:
    validate_training_perturbation(
        PerturbationSpec("gaussian_noise", "moderate"), operation=operation
    )

    with pytest.raises(ValueError, match="held out"):
        validate_training_perturbation(PerturbationSpec("contrast", "mild"), operation=operation)
    with pytest.raises(ValueError, match="strong"):
        validate_training_perturbation(
            PerturbationSpec("gaussian_blur", "strong"), operation=operation
        )


def test_evaluation_panels_keep_primary_nested_and_secondary_panels_separate() -> None:
    panels = enumerate_evaluation_panels()
    primary = panels.primary_cells

    assert len(primary) == 48
    assert len({(cell.family, cell.severity, cell.target_view) for cell in primary}) == 48
    assert {cell.family for cell in primary} == {
        "contrast",
        "brightness",
        "crop",
        "motion_blur",
    }
    assert {cell.severity for cell in primary} == {"mild", "moderate", "strong"}
    assert {cell.target_view for cell in primary} == set(CANONICAL_VIEWS)
    assert all(
        cell.variants == ("lower", "upper") for cell in primary if cell.family == "brightness"
    )
    assert all(
        cell.variants == ("horizontal", "vertical")
        for cell in primary
        if cell.family == "motion_blur"
    )
    assert all(cell.variants == (None,) for cell in primary if cell.family in {"contrast", "crop"})
    assert panels.clean_four_view == CANONICAL_VIEWS
    assert panels.clean_masks == enumerate_proper_masks()
    assert len(panels.strong_seen_family_cells) == 8
    assert {(cell.family, cell.severity) for cell in panels.strong_seen_family_cells} == {
        ("gaussian_noise", "strong"),
        ("gaussian_blur", "strong"),
    }
    assert len(panels.common_mode_cells) == 18
    assert all(cell.target_view is None and cell.common_mode for cell in panels.common_mode_cells)
    assert {cell.family for cell in panels.common_mode_cells} == {
        "gaussian_noise",
        "gaussian_blur",
        "contrast",
        "brightness",
        "crop",
        "motion_blur",
    }


def test_variant_requirements_and_unsupported_motion_resolution_fail_closed() -> None:
    with pytest.raises(ValueError, match="variant"):
        PerturbationSpec("brightness", "mild")
    with pytest.raises(ValueError, match="variant"):
        PerturbationSpec("motion_blur", "mild", variant="diagonal")
    with pytest.raises(ValueError, match="224 or 336"):
        resolve_parameters(
            PerturbationSpec("motion_blur", "mild", variant="vertical"),
            256,
        )
