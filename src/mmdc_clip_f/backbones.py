"""Pinned CLIP backbone specifications used by the released core experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    hf_model: str
    revision: str
    image_size: int
    hidden_size: int
    artifact_tag: str


BACKBONES: dict[str, BackboneSpec] = {
    "vit_l_14_336": BackboneSpec(
        name="vit_l_14_336",
        hf_model="openai/clip-vit-large-patch14-336",
        revision="ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
        image_size=336,
        hidden_size=1024,
        artifact_tag="336",
    ),
    "vit_b_32": BackboneSpec(
        name="vit_b_32",
        hf_model="openai/clip-vit-base-patch32",
        revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        image_size=224,
        hidden_size=768,
        artifact_tag="224",
    ),
}


PROMPTS = (
    "almost entirely fatty tissue.",
    "scattered areas of dense tissue in the breast.",
    "heterogeneously dense.",
    "extremely dense throughout.",
)


def get_backbone(name: str) -> BackboneSpec:
    try:
        return BACKBONES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(BACKBONES))
        raise ValueError(f"Unknown backbone {name!r}; choose one of: {choices}") from exc
