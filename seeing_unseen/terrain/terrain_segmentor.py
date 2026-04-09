"""
TerrainSegmentor
================
Classifies pixels of a satellite image into one of 8 canonical terrain types:

    ID  Label           Description
    --  -----           -----------
    0   background      Uncategorised / unknown
    1   flat_open       Open flat ground, grassland, bare earth
    2   road            Paved or unpaved road surface
    3   water           Rivers, lakes, sea, wetlands
    4   dense_forest    Heavy canopy, woodland
    5   urban           Built-up structures, rooftops, walls
    6   mud_loose       Loose soil, sand, low scrub
    7   steep_terrain   Rocky slopes, cliffs, mountains

Primary backend  : HuggingFace SegFormer (mit-b0 trained on ADE20K-150).
Fallback backend : CLIP zero-shot patch classification when SegFormer
                   weights are unavailable or the `transformers` package
                   is not installed.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T

from seeing_unseen.core.logger import logger
from seeing_unseen.utils.utils import load_json


# ---------------------------------------------------------------------------
# Canonical terrain label registry
# ---------------------------------------------------------------------------

TERRAIN_LABELS: Dict[int, str] = {
    0: "background",
    1: "flat_open",
    2: "road",
    3: "water",
    4: "dense_forest",
    5: "urban",
    6: "mud_loose",
    7: "steep_terrain",
}

TERRAIN_LABEL_TO_ID: Dict[str, int] = {v: k for k, v in TERRAIN_LABELS.items()}

NUM_TERRAIN_CLASSES = len(TERRAIN_LABELS)

# One CLIP text description per terrain class for zero-shot fallback
_TERRAIN_TEXT_PROMPTS: Dict[int, List[str]] = {
    0: ["featureless terrain", "blank area"],
    1: ["open flat grassland aerial view", "bare earth field satellite image",
        "meadow from above", "farmland aerial view"],
    2: ["road from above", "highway satellite image", "dirt track aerial view",
        "paved road bird's eye view"],
    3: ["water body from above", "river satellite image", "lake aerial view",
        "sea from above", "flooded terrain"],
    4: ["dense forest satellite image", "woodland canopy aerial view",
        "jungle from above", "tree cover remote sensing"],
    5: ["urban area aerial view", "buildings rooftop satellite image",
        "city from above", "constructed structures bird's eye view"],
    6: ["sandy terrain from above", "muddy ground aerial view",
        "loose soil satellite image", "desert scrub remote sensing"],
    7: ["mountain terrain aerial view", "rocky slope from above",
        "cliff satellite image", "steep hillside remote sensing"],
}


# ---------------------------------------------------------------------------
# Helper: build ADE20K → terrain ID look-up array
# ---------------------------------------------------------------------------

def _build_ade20k_lut(remap_json_path: str) -> torch.Tensor:
    """
    Returns a (150,) LongTensor mapping ADE20K class index → terrain label ID.
    Unknown or missing IDs map to 0 (background).
    """
    remap: Dict[str, str] = load_json(remap_json_path)
    lut = torch.zeros(150, dtype=torch.long)
    for ade_str, terrain_name in remap.items():
        ade_id = int(ade_str)
        terrain_id = TERRAIN_LABEL_TO_ID.get(terrain_name, 0)
        if ade_id < 150:
            lut[ade_id] = terrain_id
    return lut


# ---------------------------------------------------------------------------
# SegFormer backend
# ---------------------------------------------------------------------------

class _SegFormerBackend:
    """
    Thin wrapper around HuggingFace SegFormer for semantic segmentation.
    Uses the ADE20K-150 model by default.
    """

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        ade20k_lut: torch.Tensor,
    ):
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        logger.info(f"Loading SegFormer: {model_name}")
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.device = device
        self.ade20k_lut = ade20k_lut.to(device)
        logger.info("SegFormer loaded and frozen.")

    @torch.no_grad()
    def segment(
        self,
        images: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        images   : (B, 3, H, W) float32 in [0, 255]
        target_h : output height to resize label map to
        target_w : output width to resize label map to

        Returns
        -------
        terrain_map : (B, H, W) int64  canonical terrain IDs
        """
        # SegFormerImageProcessor expects uint8 PIL or numpy images
        imgs_uint8 = images.to(torch.uint8).cpu()
        pil_list = [
            T.ToPILImage()(img) for img in imgs_uint8
        ]

        inputs = self.processor(images=pil_list, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        outputs = self.model(pixel_values=pixel_values)
        # logits: (B, num_labels, H/4, W/4)
        logits = outputs.logits

        # Upsample logits to target size
        logits_up = F.interpolate(
            logits,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        ade20k_labels = logits_up.argmax(dim=1)  # (B, H, W) in [0, 149]

        # Remap ADE20K → terrain IDs
        ade20k_labels_clipped = ade20k_labels.clamp(0, 149)
        terrain_map = self.ade20k_lut[ade20k_labels_clipped]   # (B, H, W)
        return terrain_map


# ---------------------------------------------------------------------------
# CLIP zero-shot backend (fallback)
# ---------------------------------------------------------------------------

class _CLIPZeroShotBackend:
    """
    CLIP-based zero-shot terrain labelling.

    Divides each image into a P×P grid of patches and classifies each patch
    by cosine similarity to terrain text descriptions.  The resulting coarse
    label map is bilinearly upsampled to the target resolution.
    """

    PATCH_GRID = 7     # 7×7 = 49 patches per image

    def __init__(self, device: torch.device):
        try:
            import open_clip
            self._open_clip = open_clip
        except ImportError:
            import clip as openai_clip
            self._open_clip = None
            self._clip = openai_clip

        self.device = device
        self._init_model()
        self._encode_text_prompts()

    def _init_model(self):
        if self._open_clip is not None:
            self.model, _, self.preprocess = self._open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            self.tokenize = self._open_clip.tokenize
        else:
            self.model, self.preprocess = self._clip.load("ViT-B/32", device=self.device)
            self.tokenize = self._clip.tokenize

        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _encode_text_prompts(self):
        """Pre-compute mean text embeddings for each terrain class."""
        embeddings = []
        for tid in range(NUM_TERRAIN_CLASSES):
            prompts = _TERRAIN_TEXT_PROMPTS[tid]
            tokens = self.tokenize(prompts).to(self.device)
            feats = self.model.encode_text(tokens)          # (N, D)
            feats = F.normalize(feats, dim=-1).mean(0)      # (D,)
            feats = F.normalize(feats, dim=-1)
            embeddings.append(feats)
        # (num_classes, D)
        self.text_embeddings = torch.stack(embeddings, dim=0)

    @torch.no_grad()
    def segment(
        self,
        images: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        images      : (B, 3, H, W) float32 [0, 255]
        target_h/w  : output resolution

        Returns
        -------
        terrain_map : (B, H, W) int64
        """
        B, C, H, W = images.shape
        P = self.PATCH_GRID

        # Resize images to (224*P/P) and split into P×P patches
        # Simpler: resize to (P*32, P*32) and extract 32×32 patches
        patch_size = 32
        resized = F.interpolate(
            images / 255.0,
            size=(P * patch_size, P * patch_size),
            mode="bilinear",
            align_corners=False,
        )  # (B, 3, P*32, P*32)

        # CLIP normalization
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                             device=self.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                             device=self.device).view(1, 3, 1, 1)
        resized = (resized - mean) / std

        # Extract patches: (B, P*P, 3, 32, 32) → (B*P*P, 3, 32, 32)
        # Use unfold for efficiency
        patches = resized.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # patches: (B, 3, P, P, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        # (B, P, P, 3, patch_size, patch_size)
        patches = patches.view(B * P * P, 3, patch_size, patch_size)

        # Resize patches to 224×224 for CLIP
        patches_224 = F.interpolate(patches, size=(224, 224), mode="bilinear", align_corners=False)

        # Encode image patches
        img_feats = self.model.encode_image(patches_224)           # (B*P*P, D)
        img_feats = F.normalize(img_feats.float(), dim=-1)

        # Cosine similarity to terrain text embeddings
        # (B*P*P, num_classes)
        sims = img_feats @ self.text_embeddings.T
        patch_labels = sims.argmax(dim=-1)   # (B*P*P,)

        # Reshape to (B, P, P)
        patch_labels = patch_labels.view(B, 1, P, P).float()

        # Nearest-neighbour upsample to target resolution
        terrain_map = F.interpolate(
            patch_labels,
            size=(target_h, target_w),
            mode="nearest",
        ).squeeze(1).long()   # (B, H, W)

        return terrain_map


# ---------------------------------------------------------------------------
# TerrainSegmentor — public API
# ---------------------------------------------------------------------------

class TerrainSegmentor:
    """
    Classifies satellite image pixels into canonical terrain types.

    Uses SegFormer as the primary backend (automatically falls back to CLIP
    zero-shot if `transformers` is not installed or the model cannot be loaded).

    Parameters
    ----------
    model_name         : HuggingFace model ID for SegFormer
    ade20k_remap_path  : path to JSON mapping ADE20K IDs → terrain names
    device             : torch.device
    fallback           : "clip_zero_shot" to use CLIP if SegFormer fails;
                         "none" to raise an exception instead
    """

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        ade20k_remap_path: str = "data/metadata/ade20k_to_terrain.json",
        device: Optional[torch.device] = None,
        fallback: str = "clip_zero_shot",
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.fallback_mode = fallback
        self._backend = None

        # Try SegFormer
        try:
            lut = _build_ade20k_lut(ade20k_remap_path)
            self._backend = _SegFormerBackend(model_name, self.device, lut)
            self._backend_name = "segformer"
        except Exception as e:
            logger.warning(
                f"SegFormer backend failed to load ({e}). "
                f"Falling back to: {fallback}"
            )
            if fallback == "clip_zero_shot":
                self._backend = _CLIPZeroShotBackend(self.device)
                self._backend_name = "clip_zero_shot"
            else:
                raise

        logger.info(f"TerrainSegmentor initialised. Backend: {self._backend_name}")

    def segment(
        self,
        images: torch.Tensor,
        target_h: Optional[int] = None,
        target_w: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Classify terrain for a batch of satellite images.

        Parameters
        ----------
        images    : (B, 3, H, W) float32 in [0, 255] — raw pixel values
        target_h  : output height (defaults to input H)
        target_w  : output width  (defaults to input W)

        Returns
        -------
        terrain_map : (B, H_out, W_out)  int64 in [0, NUM_TERRAIN_CLASSES-1]
                      Values are canonical terrain IDs (see TERRAIN_LABELS).
        """
        B, C, H, W = images.shape
        out_h = target_h if target_h is not None else H
        out_w = target_w if target_w is not None else W

        images = images.to(self.device)
        return self._backend.segment(images, out_h, out_w)

    def label_name(self, terrain_id: int) -> str:
        return TERRAIN_LABELS.get(terrain_id, "background")

    def label_id(self, terrain_name: str) -> int:
        return TERRAIN_LABEL_TO_ID.get(terrain_name, 0)
