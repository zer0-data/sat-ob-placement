"""Terrain module for satellite placement guardrailing."""

from seeing_unseen.terrain.terrain_segmentor import TerrainSegmentor, TERRAIN_LABELS
from seeing_unseen.terrain.validity_mask import ValidityMaskGenerator, MaskedAffordanceModel

__all__ = [
    "TerrainSegmentor",
    "TERRAIN_LABELS",
    "ValidityMaskGenerator",
    "MaskedAffordanceModel",
]
