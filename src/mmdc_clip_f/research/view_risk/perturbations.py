"""Reproducible pre-normalization perturbations and named-view sampling.

The functions in this module operate only on supplied RGB tensors and immutable
specifications.  Perturbation metadata is deliberately kept outside the image
mapping consumed by later inference code.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .inputs import CANONICAL_VIEWS


FAMILIES = (
    "clean",
    "gaussian_noise",
    "gaussian_blur",
    "contrast",
    "brightness",
    "crop",
    "motion_blur",
)
SEVERITIES = ("mild", "moderate", "strong")
FITTING_FAMILIES = ("gaussian_noise", "gaussian_blur")
HELD_OUT_FAMILIES = ("contrast", "brightness", "crop", "motion_blur")
FITTING_SEVERITIES = ("mild", "moderate")

_NOISE_SIGMA = {"mild": 0.02, "moderate": 0.05, "strong": 0.10}
_BLUR_SIGMA_AT_224 = {"mild": 0.7, "moderate": 1.4, "strong": 2.8}
_CONTRAST_FACTOR = {"mild": 0.80, "moderate": 0.60, "strong": 0.40}
_BRIGHTNESS_FACTOR = {
    "mild": {"lower": 0.90, "upper": 1.10},
    "moderate": {"lower": 0.75, "upper": 1.25},
    "strong": {"lower": 0.60, "upper": 1.40},
}
_CROP_AREA = {"mild": 0.90, "moderate": 0.75, "strong": 0.60}
_MOTION_LENGTH = {
    224: {"mild": 5, "moderate": 9, "strong": 17},
    336: {"mild": 7, "moderate": 13, "strong": 25},
}


def _validate_nonnegative_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("realization_seed must be an integer")
    if seed < 0 or seed >= 2**63:
        raise ValueError("realization_seed must be in range 0..2^63-1")


@dataclass(frozen=True)
class PerturbationSpec:
    """Serializable immutable description of one replayable perturbation.

    ``realization_seed`` is the derived seed, never the private sample key.  A
    private key can be converted to a spec with :meth:`for_sample` without the
    key becoming part of the resulting object or metadata.
    """

    family: str
    severity: str | None = None
    variant: str | None = None
    realization_seed: int = 0

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown perturbation family: {self.family!r}")
        _validate_nonnegative_seed(self.realization_seed)
        if self.family == "clean":
            if self.severity is not None or self.variant is not None:
                raise ValueError("clean perturbation cannot have severity or variant")
            return
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}")
        if self.family == "brightness":
            if self.variant not in ("lower", "upper"):
                raise ValueError("brightness variant must be 'lower' or 'upper'")
        elif self.family == "motion_blur":
            if self.variant not in ("horizontal", "vertical"):
                raise ValueError("motion_blur variant must be 'horizontal' or 'vertical'")
        elif self.variant is not None:
            raise ValueError(f"{self.family} does not accept a variant")

    @classmethod
    def for_sample(
        cls,
        family: str,
        severity: str | None,
        *,
        private_sample_key: str | bytes,
        cell: str,
        seed: int,
        variant: str | None = None,
    ) -> PerturbationSpec:
        """Build a replayable spec while retaining no private sample key."""

        return cls(
            family=family,
            severity=severity,
            variant=variant,
            realization_seed=derive_realization_seed(private_sample_key, cell, seed),
        )

    def to_dict(self) -> dict[str, str | int | None]:
        """Return a JSON-serializable representation."""

        return {
            "family": self.family,
            "severity": self.severity,
            "variant": self.variant,
            "realization_seed": self.realization_seed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PerturbationSpec:
        """Restore a spec from :meth:`to_dict` output."""

        if not isinstance(value, Mapping):
            raise TypeError("serialized perturbation spec must be a mapping")
        expected = {"family", "severity", "variant", "realization_seed"}
        if set(value) != expected:
            raise ValueError("serialized perturbation spec has missing or unknown fields")
        return cls(
            family=value["family"],  # type: ignore[arg-type]
            severity=value["severity"],  # type: ignore[arg-type]
            variant=value["variant"],  # type: ignore[arg-type]
            realization_seed=value["realization_seed"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class PerturbationMetadata:
    """Non-inference metadata returned separately from the transformed tensor."""

    spec: PerturbationSpec
    parameters: tuple[tuple[str, int | float | str], ...]

    def parameter_dict(self) -> dict[str, int | float | str]:
        return dict(self.parameters)

    def to_dict(self) -> dict[str, object]:
        return {"spec": self.spec.to_dict(), "parameters": self.parameter_dict()}


@dataclass(frozen=True)
class RealizedParent:
    """One realized nonempty parent and its out-of-band perturbation metadata."""

    images: Mapping[str, Tensor]
    metadata: Mapping[str, PerturbationMetadata]

    @property
    def observed_views(self) -> tuple[str, ...]:
        return tuple(self.images)


@dataclass(frozen=True)
class TrainingSupportCase:
    """One outcome in the exact default training sampling distribution."""

    stratum: str
    observed_views: tuple[str, ...]
    probability: Fraction
    family: str | None = None
    severity: str | None = None
    stress_mode: str | None = None
    stressed_views: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainingSampleSpec:
    """A seeded, replayable parent sampling decision."""

    stratum: str
    observed_views: tuple[str, ...]
    stress_mode: str | None
    stressed_views: tuple[str, ...]
    perturbation: PerturbationSpec | None

    def to_dict(self) -> dict[str, object]:
        return {
            "stratum": self.stratum,
            "observed_views": list(self.observed_views),
            "stress_mode": self.stress_mode,
            "stressed_views": list(self.stressed_views),
            "perturbation": (None if self.perturbation is None else self.perturbation.to_dict()),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
        *,
        operation: str = "confidence-fit",
    ) -> TrainingSampleSpec:
        if not isinstance(value, Mapping):
            raise TypeError("serialized training sample spec must be a mapping")
        expected = {
            "stratum",
            "observed_views",
            "stress_mode",
            "stressed_views",
            "perturbation",
        }
        if set(value) != expected:
            raise ValueError("serialized training sample spec has missing or unknown fields")
        perturbation_value = value["perturbation"]
        perturbation = (
            None if perturbation_value is None else PerturbationSpec.from_dict(perturbation_value)  # type: ignore[arg-type]
        )
        draw = cls(
            stratum=value["stratum"],  # type: ignore[arg-type]
            observed_views=tuple(value["observed_views"]),  # type: ignore[arg-type]
            stress_mode=value["stress_mode"],  # type: ignore[arg-type]
            stressed_views=tuple(value["stressed_views"]),  # type: ignore[arg-type]
            perturbation=perturbation,
        )
        validate_training_sample_spec(draw, operation=operation)
        return draw


@dataclass(frozen=True)
class EvaluationCell:
    """A panel cell; variants are nested replicates, not extra primary cells."""

    family: str
    severity: str
    target_view: str | None
    variants: tuple[str | None, ...]
    common_mode: bool = False


@dataclass(frozen=True)
class EvaluationPanels:
    """Frozen enumeration of primary and separately reported secondary panels."""

    clean_four_view: tuple[str, ...]
    clean_masks: tuple[tuple[str, ...], ...]
    primary_cells: tuple[EvaluationCell, ...]
    strong_seen_family_cells: tuple[EvaluationCell, ...]
    common_mode_cells: tuple[EvaluationCell, ...]


def _hash_component(hasher: object, value: bytes) -> None:
    # hashlib's concrete type is intentionally not part of its public typing API.
    hasher.update(len(value).to_bytes(8, "big"))  # type: ignore[attr-defined]
    hasher.update(value)  # type: ignore[attr-defined]


def derive_realization_seed(private_sample_key: str | bytes, cell: str, seed: int) -> int:
    """Derive a stable local Torch seed using SHA256 and length-delimited fields."""

    if isinstance(private_sample_key, str):
        key_bytes = private_sample_key.encode("utf-8")
    elif isinstance(private_sample_key, bytes):
        key_bytes = private_sample_key
    else:
        raise TypeError("private_sample_key must be str or bytes")
    if not key_bytes:
        raise ValueError("private_sample_key cannot be empty")
    if not isinstance(cell, str) or not cell:
        raise ValueError("cell must be a nonempty string")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    hasher = hashlib.sha256()
    _hash_component(hasher, b"mmdc-clip-f/view-risk/perturbation-seed/v1")
    _hash_component(hasher, key_bytes)
    _hash_component(hasher, cell.encode("utf-8"))
    _hash_component(hasher, str(seed).encode("ascii"))
    return int.from_bytes(hasher.digest()[:8], "big") & ((1 << 63) - 1)


def resolve_parameters(
    spec: PerturbationSpec,
    resolution: int,
) -> dict[str, int | float | str]:
    """Resolve protocol parameters for a square image resolution."""

    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution <= 0:
        raise ValueError("resolution must be a positive integer")
    if spec.family == "clean":
        return {}
    assert spec.severity is not None
    if spec.family == "gaussian_noise":
        return {"sigma": _NOISE_SIGMA[spec.severity]}
    if spec.family == "gaussian_blur":
        sigma = _BLUR_SIGMA_AT_224[spec.severity] * resolution / 224.0
        kernel_size = 2 * math.ceil(3.0 * sigma) + 1
        return {"sigma": sigma, "kernel_size": kernel_size}
    if spec.family == "contrast":
        return {"factor": _CONTRAST_FACTOR[spec.severity]}
    if spec.family == "brightness":
        assert spec.variant is not None
        return {
            "factor": _BRIGHTNESS_FACTOR[spec.severity][spec.variant],
            "direction": spec.variant,
        }
    if spec.family == "crop":
        area = _CROP_AREA[spec.severity]
        return {"retained_area": area, "crop_side": math.floor(resolution * math.sqrt(area))}
    if resolution not in _MOTION_LENGTH:
        raise ValueError("motion blur is specified only for resolution 224 or 336")
    assert spec.variant is not None
    return {
        "length": _MOTION_LENGTH[resolution][spec.severity],
        "orientation": spec.variant,
    }


def resolve_realized_parameters(
    spec: PerturbationSpec, resolution: int
) -> dict[str, int | float | str]:
    """Replay every persisted realization parameter, including seeded crop offsets."""

    parameters = resolve_parameters(spec, resolution)
    if spec.family == "crop":
        crop_side = int(parameters["crop_side"])
        max_offset = resolution - crop_side
        coordinates = torch.rand(
            2, generator=_generator(spec.realization_seed), dtype=torch.float64
        )
        parameters["top"] = min(
            math.floor(float(coordinates[0]) * (max_offset + 1)), max_offset
        )
        parameters["left"] = min(
            math.floor(float(coordinates[1]) * (max_offset + 1)), max_offset
        )
    return parameters


def _validate_image(image: Tensor) -> int:
    if not isinstance(image, Tensor):
        raise TypeError("image must be a torch.Tensor")
    if not image.is_floating_point():
        raise TypeError("image must have a floating dtype")
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must be an RGB tensor with shape [3, r, r]")
    if image.shape[1] != image.shape[2]:
        raise ValueError("image must be square")
    if not torch.isfinite(image).all():
        raise ValueError("image pixels must be finite")
    if ((image < 0) | (image > 1)).any():
        raise ValueError("image pixels must be in [0,1]")
    return image.shape[1]


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _convolve_reflect(image: Tensor, kernel: Tensor) -> Tensor:
    padding = kernel.shape[-1] // 2
    if image.shape[-1] <= padding or image.shape[-2] <= padding:
        raise ValueError("image is too small for the protocol reflect padding")
    channels = image.shape[0]
    weights = kernel.to(device=image.device, dtype=image.dtype).expand(channels, 1, -1, -1)
    padded = F.pad(image.unsqueeze(0), (padding, padding, padding, padding), mode="reflect")
    return F.conv2d(padded, weights, groups=channels).squeeze(0)


def _gaussian_kernel(sigma: float, kernel_size: int) -> Tensor:
    radius = kernel_size // 2
    coordinates = torch.arange(-radius, radius + 1, dtype=torch.float64)
    one_dimensional = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
    one_dimensional /= one_dimensional.sum()
    return one_dimensional[:, None] * one_dimensional[None, :]


def _motion_kernel(length: int, orientation: str) -> Tensor:
    kernel = torch.zeros((length, length), dtype=torch.float64)
    center = length // 2
    if orientation == "horizontal":
        kernel[center, :] = 1.0 / length
    else:
        kernel[:, center] = 1.0 / length
    return kernel


def _parameter_tuple(
    parameters: Mapping[str, int | float | str],
) -> tuple[tuple[str, int | float | str], ...]:
    return tuple(parameters.items())


def apply_perturbation(
    image: Tensor,
    spec: PerturbationSpec,
) -> tuple[Tensor, PerturbationMetadata]:
    """Apply one stress before normalization and return metadata out of band."""

    resolution = _validate_image(image)
    if not isinstance(spec, PerturbationSpec):
        raise TypeError("spec must be a PerturbationSpec")
    parameters = resolve_realized_parameters(spec, resolution)
    if spec.family == "clean":
        return image, PerturbationMetadata(spec, ())

    if spec.family == "gaussian_noise":
        spatial_noise = torch.randn(
            (resolution, resolution),
            generator=_generator(spec.realization_seed),
            dtype=torch.float64,
        ).to(device=image.device, dtype=image.dtype)
        stressed = image + float(parameters["sigma"]) * spatial_noise.unsqueeze(0)
    elif spec.family == "gaussian_blur":
        kernel = _gaussian_kernel(
            float(parameters["sigma"]),
            int(parameters["kernel_size"]),
        )
        stressed = _convolve_reflect(image, kernel)
    elif spec.family == "contrast":
        mean = image.mean()
        stressed = mean + float(parameters["factor"]) * (image - mean)
    elif spec.family == "brightness":
        stressed = image * float(parameters["factor"])
    elif spec.family == "crop":
        crop_side = int(parameters["crop_side"])
        top = int(parameters["top"])
        left = int(parameters["left"])
        crop = image[:, top : top + crop_side, left : left + crop_side]
        stressed = F.interpolate(
            crop.unsqueeze(0),
            size=(resolution, resolution),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
    else:
        kernel = _motion_kernel(int(parameters["length"]), str(parameters["orientation"]))
        stressed = _convolve_reflect(image, kernel)

    stressed = stressed.clamp(0.0, 1.0)
    return stressed, PerturbationMetadata(spec, _parameter_tuple(parameters))


def _canonical_observed(observed: Iterable[str] | Sequence[bool] | Tensor) -> tuple[str, ...]:
    if isinstance(observed, Tensor):
        if observed.ndim != 1 or observed.shape[0] != len(CANONICAL_VIEWS):
            raise ValueError("observed mask must have length 4")
        if observed.dtype != torch.bool:
            raise TypeError("observed tensor mask must have boolean dtype")
        materialized: tuple[object, ...] = tuple(observed.detach().cpu().tolist())
    else:
        if isinstance(observed, str):
            raise TypeError("observed views must not be one string")
        materialized = tuple(observed)

    if materialized and all(isinstance(value, bool) for value in materialized):
        if len(materialized) != len(CANONICAL_VIEWS):
            raise ValueError("observed mask must have length 4")
        selected = tuple(view for view, present in zip(CANONICAL_VIEWS, materialized) if present)
    else:
        if not all(isinstance(view, str) for view in materialized):
            raise TypeError("observed view names must be strings")
        if len(materialized) != len(set(materialized)):
            raise ValueError("observed views contain duplicate names")
        unknown = set(materialized).difference(CANONICAL_VIEWS)
        if unknown:
            raise ValueError(f"observed views contain unknown names: {sorted(unknown)}")
        names = set(materialized)
        selected = tuple(view for view in CANONICAL_VIEWS if view in names)
    if not selected:
        raise ValueError("at least one observed view is required")
    return selected


def realize_parent(
    images: Mapping[str, Tensor],
    observed: Iterable[str] | Sequence[bool] | Tensor,
    *,
    perturbations: Mapping[str, PerturbationSpec] | None = None,
    common_mode: PerturbationSpec | None = None,
) -> RealizedParent:
    """Realize a named parent once, omitting rather than encoding absent views."""

    if not isinstance(images, Mapping):
        raise TypeError("images must be a named mapping")
    supplied = dict(images)
    unknown = set(supplied).difference(CANONICAL_VIEWS)
    if unknown:
        raise ValueError(f"images contain unknown views: {sorted(unknown)}")
    observed_views = _canonical_observed(observed)
    missing = set(observed_views).difference(supplied)
    if missing:
        raise ValueError(f"observed views are missing supplied tensors: {sorted(missing)}")
    if perturbations is not None and common_mode is not None:
        raise ValueError("supply either per-view perturbations or common_mode, not both")
    per_view = {} if perturbations is None else dict(perturbations)
    unknown_stresses = set(per_view).difference(observed_views)
    if unknown_stresses:
        raise ValueError(f"cannot perturb absent or unknown views: {sorted(unknown_stresses)}")
    if common_mode is not None and not isinstance(common_mode, PerturbationSpec):
        raise TypeError("common_mode must be a PerturbationSpec")
    if any(not isinstance(spec, PerturbationSpec) for spec in per_view.values()):
        raise TypeError("all per-view perturbations must be PerturbationSpec instances")

    realized: dict[str, Tensor] = {}
    metadata: dict[str, PerturbationMetadata] = {}
    for view in observed_views:
        image = supplied[view]
        spec = common_mode if common_mode is not None else per_view.get(view)
        if spec is None:
            _validate_image(image)
            realized[view] = image
        else:
            realized[view], metadata[view] = apply_perturbation(image, spec)

    return RealizedParent(
        images=MappingProxyType(realized),
        metadata=MappingProxyType(metadata),
    )


def omit_parent_view(parent: RealizedParent, removed_view: str) -> RealizedParent:
    """Make an omission child by retaining the exact surviving tensor objects."""

    if not isinstance(parent, RealizedParent):
        raise TypeError("parent must be a RealizedParent")
    if removed_view not in parent.images:
        raise ValueError("removed_view must name an observed parent view")
    if len(parent.images) == 1:
        raise ValueError("omission child must remain nonempty")
    images = {view: tensor for view, tensor in parent.images.items() if view != removed_view}
    metadata = {view: record for view, record in parent.metadata.items() if view != removed_view}
    return RealizedParent(MappingProxyType(images), MappingProxyType(metadata))


@lru_cache(maxsize=1)
def enumerate_proper_masks() -> tuple[tuple[str, ...], ...]:
    """Return all 14 canonical proper nonempty named-view masks."""

    masks: list[tuple[str, ...]] = []
    for bits in range(1, (1 << len(CANONICAL_VIEWS)) - 1):
        masks.append(
            tuple(view for index, view in enumerate(CANONICAL_VIEWS) if bits & (1 << index))
        )
    return tuple(masks)


@lru_cache(maxsize=1)
def enumerate_training_support() -> tuple[TrainingSupportCase, ...]:
    """Enumerate the exact four-stratum training distribution without sampling."""

    full = tuple(CANONICAL_VIEWS)
    masks = enumerate_proper_masks()
    cases: list[TrainingSupportCase] = [TrainingSupportCase("clean4", full, Fraction(1, 4))]
    cases.extend(
        TrainingSupportCase("clean_proper_mask", mask, Fraction(1, 4 * len(masks)))
        for mask in masks
    )

    stress_combinations = tuple(
        (family, severity) for family in FITTING_FAMILIES for severity in FITTING_SEVERITIES
    )
    for stratum, observed_options in (
        ("stressed4", (full,)),
        ("stressed_proper_mask", masks),
    ):
        mask_probability = Fraction(1, len(observed_options))
        for observed_views in observed_options:
            for family, severity in stress_combinations:
                base_probability = (
                    Fraction(1, 4) * mask_probability * Fraction(1, len(stress_combinations))
                )
                cases.append(
                    TrainingSupportCase(
                        stratum=stratum,
                        observed_views=observed_views,
                        probability=base_probability * Fraction(1, 5),
                        family=family,
                        severity=severity,
                        stress_mode="common",
                        stressed_views=observed_views,
                    )
                )
                single_probability = base_probability * Fraction(4, 5) / len(observed_views)
                cases.extend(
                    TrainingSupportCase(
                        stratum=stratum,
                        observed_views=observed_views,
                        probability=single_probability,
                        family=family,
                        severity=severity,
                        stress_mode="single",
                        stressed_views=(view,),
                    )
                    for view in observed_views
                )
    if sum((case.probability for case in cases), Fraction()) != 1:
        raise RuntimeError("internal training support probabilities do not sum to one")
    return tuple(cases)


def validate_training_perturbation(
    spec: PerturbationSpec,
    *,
    operation: str,
) -> None:
    """Fail closed on stresses forbidden during confidence fit or tune."""

    if operation not in ("confidence-fit", "tune"):
        raise ValueError("operation must be 'confidence-fit' or 'tune'")
    if not isinstance(spec, PerturbationSpec):
        raise TypeError("spec must be a PerturbationSpec")
    if spec.family == "clean":
        return
    if spec.family in HELD_OUT_FAMILIES:
        raise ValueError(f"{spec.family} is held out from {operation}")
    if spec.family not in FITTING_FAMILIES:
        raise ValueError(f"{spec.family} is not authorized for {operation}")
    if spec.severity == "strong":
        raise ValueError(f"strong severity is evaluation-only and forbidden in {operation}")
    if spec.severity not in FITTING_SEVERITIES:
        raise ValueError(f"severity is not authorized for {operation}")


def validate_training_sample_spec(
    draw: TrainingSampleSpec,
    *,
    operation: str,
) -> None:
    """Validate a complete fit/tune draw, including its structural invariants."""

    if not isinstance(draw, TrainingSampleSpec):
        raise TypeError("draw must be a TrainingSampleSpec")
    if operation not in ("confidence-fit", "tune"):
        raise ValueError("operation must be 'confidence-fit' or 'tune'")

    try:
        observed_views = _canonical_observed(draw.observed_views)
    except (TypeError, ValueError) as error:
        raise ValueError("training sample has invalid observed views") from error
    if observed_views != tuple(draw.observed_views):
        raise ValueError("training sample observed views must use canonical order")

    clean_strata = ("clean4", "clean_proper_mask")
    stressed_strata = ("stressed4", "stressed_proper_mask")
    if draw.stratum not in (*clean_strata, *stressed_strata):
        raise ValueError("unknown training stratum")
    expects_four = draw.stratum in ("clean4", "stressed4")
    if expects_four and observed_views != tuple(CANONICAL_VIEWS):
        raise ValueError(f"{draw.stratum} stratum requires all four observed views")
    if not expects_four and observed_views not in enumerate_proper_masks():
        raise ValueError(f"{draw.stratum} stratum requires a proper nonempty mask")

    if draw.stratum in clean_strata:
        if draw.stress_mode is not None or draw.stressed_views or draw.perturbation is not None:
            raise ValueError(
                "clean stratum cannot contain a stress mode, stressed views, or stress"
            )
        return

    if draw.perturbation is None:
        raise ValueError("stressed stratum requires a perturbation")
    if draw.perturbation.family == "clean":
        raise ValueError("stressed stratum requires a non-clean perturbation")
    validate_training_perturbation(draw.perturbation, operation=operation)
    if draw.stress_mode not in ("single", "common"):
        raise ValueError("stressed stratum mode must be 'single' or 'common'")

    stressed_views = tuple(draw.stressed_views)
    canonical_stressed = tuple(view for view in CANONICAL_VIEWS if view in stressed_views)
    if (
        not stressed_views
        or len(stressed_views) != len(set(stressed_views))
        or any(view not in CANONICAL_VIEWS for view in stressed_views)
        or stressed_views != canonical_stressed
    ):
        raise ValueError("stressed views must be unique known names in canonical order")
    if draw.stress_mode == "single":
        if len(stressed_views) != 1 or stressed_views[0] not in observed_views:
            raise ValueError("single stress mode requires exactly one observed stressed view")
    elif stressed_views != observed_views:
        raise ValueError("common stress mode requires all and only observed views")


def _choose_training_case(
    private_sample_key: str | bytes,
    cell: str,
    seed: int,
) -> TrainingSupportCase:
    support = enumerate_training_support()
    denominator = math.lcm(*(case.probability.denominator for case in support))
    ticket = (
        derive_realization_seed(
            private_sample_key,
            f"training-support/{cell}",
            seed,
        )
        % denominator
    )
    boundary = 0
    for case in support:
        boundary += case.probability.numerator * (denominator // case.probability.denominator)
        if ticket < boundary:
            return case
    raise RuntimeError("internal training support selection failed")


def sample_training_spec(
    private_sample_key: str | bytes,
    cell: str,
    seed: int,
    *,
    operation: str = "confidence-fit",
) -> TrainingSampleSpec:
    """Select one deterministic default training draw from the exact support."""

    if operation not in ("confidence-fit", "tune"):
        raise ValueError("operation must be 'confidence-fit' or 'tune'")
    case = _choose_training_case(private_sample_key, cell, seed)
    perturbation: PerturbationSpec | None = None
    if case.family is not None:
        perturbation = PerturbationSpec.for_sample(
            case.family,
            case.severity,
            private_sample_key=private_sample_key,
            cell=f"training-realization/{cell}/{case.stress_mode}",
            seed=seed,
        )
        validate_training_perturbation(perturbation, operation=operation)
    draw = TrainingSampleSpec(
        stratum=case.stratum,
        observed_views=case.observed_views,
        stress_mode=case.stress_mode,
        stressed_views=case.stressed_views,
        perturbation=perturbation,
    )
    validate_training_sample_spec(draw, operation=operation)
    return draw


def realize_training_parent(
    images: Mapping[str, Tensor],
    draw: TrainingSampleSpec,
    *,
    operation: str = "confidence-fit",
) -> RealizedParent:
    """Realize a sampled training parent without exposing sampling metadata as input."""

    validate_training_sample_spec(draw, operation=operation)
    if draw.perturbation is None:
        return realize_parent(images, draw.observed_views)
    if draw.stress_mode == "common":
        return realize_parent(images, draw.observed_views, common_mode=draw.perturbation)
    perturbations = {view: draw.perturbation for view in draw.stressed_views}
    return realize_parent(images, draw.observed_views, perturbations=perturbations)


def _variants_for(family: str) -> tuple[str | None, ...]:
    if family == "brightness":
        return ("lower", "upper")
    if family == "motion_blur":
        return ("horizontal", "vertical")
    return (None,)


@lru_cache(maxsize=1)
def enumerate_evaluation_panels() -> EvaluationPanels:
    """Enumerate frozen primary, clean-mask, and separate secondary panels."""

    primary = tuple(
        EvaluationCell(
            family=family,
            severity=severity,
            target_view=target_view,
            variants=_variants_for(family),
        )
        for family in HELD_OUT_FAMILIES
        for severity in SEVERITIES
        for target_view in CANONICAL_VIEWS
    )
    strong_seen = tuple(
        EvaluationCell(
            family=family,
            severity="strong",
            target_view=target_view,
            variants=(None,),
        )
        for family in FITTING_FAMILIES
        for target_view in CANONICAL_VIEWS
    )
    common_mode = tuple(
        EvaluationCell(
            family=family,
            severity=severity,
            target_view=None,
            variants=_variants_for(family),
            common_mode=True,
        )
        for family in (*FITTING_FAMILIES, *HELD_OUT_FAMILIES)
        for severity in SEVERITIES
    )
    return EvaluationPanels(
        clean_four_view=tuple(CANONICAL_VIEWS),
        clean_masks=enumerate_proper_masks(),
        primary_cells=primary,
        strong_seen_family_cells=strong_seen,
        common_mode_cells=common_mode,
    )
