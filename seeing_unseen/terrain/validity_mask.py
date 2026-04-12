"""
ValidityMaskGenerator
=====================
Cross-references a terrain label map with per-military-class deployment
constraints to produce a binary validity mask:

    mask[b, 0, h, w] = 0  if terrain is forbidden for the given military class
    mask[b, 0, h, w] = 1  otherwise

The mask is then multiplied onto the model's raw affordance logit to enforce
tactical placement realism before the final sigmoid activation.

Also exports MaskedAffordanceModel — a thin nn.Module wrapper that applies
guardrailing inside the standard forward() contract.
"""

import json
import os
from typing import Dict, List, Optional, Set, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from seeing_unseen.core.logger import logger
from seeing_unseen.core.registry import registry
from seeing_unseen.models.base import SPModel
from seeing_unseen.terrain.terrain_segmentor import (
    TERRAIN_LABEL_TO_ID,
    TERRAIN_LABELS,
    TerrainSegmentor,
)
from seeing_unseen.utils.utils import load_json


# ---------------------------------------------------------------------------
# ValidityMaskGenerator
# ---------------------------------------------------------------------------

class ValidityMaskGenerator:
    """
    Generates a binary validity mask for placement affordance post-processing.

    Usage
    -----
    vmg = ValidityMaskGenerator(
        constraints_path="data/metadata/deployment_constraints.json",
        terrain_segmentor=my_segmentor,
        dilation_px=5,
        min_valid_area_frac=0.05,
    )
    mask = vmg.generate(images, military_class="armored_vehicle")
    # mask: (B, 1, H, W) float32 ∈ {0, 1}
    masked_heatmap = raw_affordance_logit * mask
    """

    def __init__(
        self,
        constraints_path: str,
        terrain_segmentor: Optional[TerrainSegmentor] = None,
        dilation_px: int = 5,
        min_valid_area_frac: float = 0.05,
    ):
        """
        Parameters
        ----------
        constraints_path    : path to deployment_constraints.json
        terrain_segmentor   : a TerrainSegmentor instance (created externally
                              so the segmentor can be shared across calls)
        dilation_px         : pixels to dilate forbidden regions (soft boundary)
        min_valid_area_frac : warn if valid area fraction drops below this value
        """
        self.segmentor = terrain_segmentor
        self.dilation_px = dilation_px
        self.min_valid_area_frac = min_valid_area_frac

        # Load constraints JSON
        if not os.path.isfile(constraints_path):
            raise FileNotFoundError(
                f"Deployment constraints file not found: '{constraints_path}'"
            )
        self._constraints: Dict = load_json(constraints_path)
        logger.info(
            f"ValidityMaskGenerator loaded {len(self._constraints)} asset classes "
            f"from {constraints_path}"
        )

        # Pre-compute {class_name: frozenset of forbidden terrain IDs}
        self._forbidden_ids: Dict[str, Set[int]] = {}
        for cls_name, info in self._constraints.items():
            if cls_name.startswith("_"):
                continue
            forbidden_names: List[str] = info.get("forbidden", [])
            self._forbidden_ids[cls_name] = frozenset(
                TERRAIN_LABEL_TO_ID.get(name, -1) for name in forbidden_names
            ) - {-1}

        logger.info(
            f"Forbidden terrain IDs per class: "
            + ", ".join(
                f"{k}→{v}" for k, v in self._forbidden_ids.items()
            )
        )

    def _get_forbidden_ids(self, military_class: str) -> Set[int]:
        if military_class in self._forbidden_ids:
            return self._forbidden_ids[military_class]
        logger.warning(
            f"Unknown military class '{military_class}'. Using 'default' constraints."
        )
        return self._forbidden_ids.get("default", frozenset({3}))  # water only

    def _apply_morphology(
        self,
        forbidden_mask_np: np.ndarray,
        dilation_px: int,
    ) -> np.ndarray:
        """
        Dilate the forbidden mask by `dilation_px` pixels to create soft
        exclusion boundaries.

        Parameters
        ----------
        forbidden_mask_np : (H, W) uint8 — 1 where terrain is forbidden
        dilation_px       : kernel size for dilation

        Returns
        -------
        dilated_forbidden : (H, W) uint8
        """
        if dilation_px <= 0:
            return forbidden_mask_np
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1)
        )
        return cv2.dilate(forbidden_mask_np, kernel, iterations=1)

    def generate(
        self,
        images: torch.Tensor,
        military_class: Union[str, List[str]] = "default",
        precomputed_terrain: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate binary validity mask.

        Parameters
        ----------
        images              : (B, 3, H, W) float32 [0, 255]
        military_class      : single string or list of B strings (per-image class)
        precomputed_terrain : (B, H, W) int64 — if provided, skip segmentation

        Returns
        -------
        validity_mask : (B, 1, H, W) float32 ∈ {0.0, 1.0}
                        0 = forbidden / invalid placement
                        1 = valid placement
        """
        B, C, H, W = images.shape
        device = images.device

        # Normalise military_class to a list
        if isinstance(military_class, str):
            classes = [military_class] * B
        else:
            assert len(military_class) == B, (
                f"Length of military_class list ({len(military_class)}) "
                f"must match batch size ({B})"
            )
            classes = list(military_class)

        # -------------------------------------------------------------------
        # 1. Get terrain label map
        # -------------------------------------------------------------------
        if precomputed_terrain is not None:
            terrain_map = precomputed_terrain.to(device)   # (B, H, W) int64
        elif self.segmentor is not None:
            terrain_map = self.segmentor.segment(images, H, W)
        else:
            raise RuntimeError(
                "ValidityMaskGenerator requires either a TerrainSegmentor or "
                "precomputed_terrain to be provided."
            )

        # -------------------------------------------------------------------
        # 2. Build validity mask for each sample
        # -------------------------------------------------------------------
        mask_list = []
        for b in range(B):
            forbidden_ids = self._get_forbidden_ids(classes[b])
            terrain_b = terrain_map[b]   # (H, W) int64

            # Forbidden pixel map: 1 where terrain is in forbidden set
            forbidden = torch.zeros(H, W, dtype=torch.uint8, device=device)
            for fid in forbidden_ids:
                forbidden |= (terrain_b == fid).to(torch.uint8)

            # Apply morphological dilation to forbidden region (on CPU via opencv)
            if self.dilation_px > 0:
                forbidden_np = forbidden.cpu().numpy()
                forbidden_np = self._apply_morphology(forbidden_np, self.dilation_px)
                forbidden = torch.from_numpy(forbidden_np).to(device)

            # Validity = NOT forbidden
            valid = (forbidden == 0).float()   # (H, W)

            # Warn if very little valid area remains
            valid_frac = valid.mean().item()
            if valid_frac < self.min_valid_area_frac:
                logger.warning(
                    f"[Sample {b}, class '{classes[b]}'] Valid area is very small: "
                    f"{valid_frac:.2%} < {self.min_valid_area_frac:.2%}. "
                    "Consider relaxing constraints or checking terrain segmentation."
                )

            mask_list.append(valid)

        # Stack → (B, H, W) → (B, 1, H, W)
        mask = torch.stack(mask_list, dim=0).unsqueeze(1)
        return mask


# ---------------------------------------------------------------------------
# MaskedAffordanceModel
# ---------------------------------------------------------------------------

@registry.register_affordance_model(name="remote_clip_unet_masked")
class MaskedAffordanceModel(SPModel):
    """
    Wraps any registered affordance model with terrain validity masking.

    The mask is applied to the **raw logit** (before Sigmoid) during inference:
        output["affordance"] = raw_logit * validity_mask

    During training the guardrailing can be disabled via config to prevent
    the terrain segmentor from interfering with gradient flow.

    Config section (under terrain_filter in clip_unet.yaml):
        enabled                : bool   — toggle the whole filter
        segmentor.model_name   : str    — HuggingFace SegFormer model ID
        segmentor.fallback     : str    — "clip_zero_shot" | "none"
        segmentor.ade20k_remap_path : str
        constraints_path       : str    — deployment_constraints.json path
        morphology.dilation_px : int
        morphology.min_valid_area_frac : float
        apply_during_training  : bool   — if False, skip mask at train time
    """

    def __init__(
        self,
        input_shape: tuple,
        target_input_shape: tuple,
        base_model_name: str = "remote_clip_unet",
        terrain_cfg: Optional[DictConfig] = None,
        **base_model_kwargs,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # 1. Build the wrapped base model
        # ------------------------------------------------------------------
        base_cls = registry.get_affordance_model(base_model_name)
        if base_cls is None:
            raise ValueError(
                f"Base model '{base_model_name}' not found in registry. "
                f"Did you import the models module?"
            )
        self.base_model = base_cls(
            input_shape=input_shape,
            target_input_shape=target_input_shape,
            **base_model_kwargs,
        )
        logger.info(
            f"MaskedAffordanceModel: base model = '{base_model_name}'"
        )

        # ------------------------------------------------------------------
        # 2. Build terrain segmentor + validity mask generator (if enabled)
        # ------------------------------------------------------------------
        self.guardrailing_enabled = False
        self.apply_during_training = False
        self._vmg: Optional[ValidityMaskGenerator] = None

        if terrain_cfg is not None:
            cfg = OmegaConf.to_container(terrain_cfg, resolve=True) \
                if isinstance(terrain_cfg, DictConfig) else terrain_cfg

            self.guardrailing_enabled = cfg.get("enabled", True)
            self.apply_during_training = cfg.get("apply_during_training", False)

            if self.guardrailing_enabled:
                seg_cfg  = cfg.get("segmentor", {})
                morph    = cfg.get("morphology", {})

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                segmentor = TerrainSegmentor(
                    model_name=seg_cfg.get(
                        "model_name",
                        "nvidia/segformer-b0-finetuned-ade-512-512",
                    ),
                    ade20k_remap_path=seg_cfg.get(
                        "ade20k_remap_path",
                        "data/metadata/ade20k_to_terrain.json",
                    ),
                    device=device,
                    fallback=seg_cfg.get("fallback", "clip_zero_shot"),
                )

                self._vmg = ValidityMaskGenerator(
                    constraints_path=cfg.get(
                        "constraints_path",
                        "data/metadata/deployment_constraints.json",
                    ),
                    terrain_segmentor=segmentor,
                    dilation_px=morph.get("dilation_px", 5),
                    min_valid_area_frac=morph.get("min_valid_area_frac", 0.05),
                )
                logger.info("MaskedAffordanceModel: terrain guardrailing ENABLED.")
            else:
                logger.info("MaskedAffordanceModel: terrain guardrailing DISABLED by config.")
        else:
            logger.info(
                "MaskedAffordanceModel: no terrain_cfg provided — guardrailing disabled."
            )

    def _should_apply_mask(self) -> bool:
        if not self.guardrailing_enabled or self._vmg is None:
            return False
        if self.training and not self.apply_during_training:
            return False
        return True

    def forward(self, **kwargs) -> Dict:
        batch = kwargs["batch"]

        # Run base model
        output = self.base_model(batch=batch)

        # Apply validity mask if enabled
        if self._should_apply_mask():
            images = batch["image"]                   # (B, 3, H, W)
            military_class = batch.get(
                "military_class", "default"
            )                                         # str or List[str]

            validity_mask = self._vmg.generate(
                images=images,
                military_class=military_class,
            )

            # Resize mask to match affordance spatial size (handles any resolution)
            aff = output["affordance"]                # (B, 1, H, W)
            if validity_mask.shape[-2:] != aff.shape[-2:]:
                validity_mask = F.interpolate(
                    validity_mask,
                    size=aff.shape[-2:],
                    mode="nearest",
                )

            # Multiply raw logit by mask (zeros out forbidden regions)
            output["affordance"] = aff * validity_mask.to(aff.device)
            output["validity_mask"] = validity_mask   # expose for visualization

        return output

