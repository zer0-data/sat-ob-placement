"""
Smoke test for VocAda-Inspired Output Guardrailing.
=====================================================
Tests ValidityMaskGenerator and MaskedAffordanceModel end-to-end with
random weights (no SegFormer download needed — uses CLIP zero-shot fallback).

Run from the repo root:
    python scripts/test_terrain_filter.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

DEVICE = "cpu"
B, H, W = 2, 480, 640

print("=" * 65)
print("VocAda Terrain Guardrailing — smoke tests (random weights, CPU)")
print("=" * 65)

# ---------------------------------------------------------------------------
# 1. TerrainSegmentor — CLIP zero-shot fallback (no SegFormer download needed)
# ---------------------------------------------------------------------------
print("\n[1] TerrainSegmentor (CLIP zero-shot fallback, no checkpoint) ...")
from seeing_unseen.terrain.terrain_segmentor import (
    TerrainSegmentor, TERRAIN_LABELS, NUM_TERRAIN_CLASSES
)

# Build a tiny ade20k remap manually (don't read from disk for portability)
import json, tempfile
remap = {str(i): "flat_open" for i in range(150)}
remap["22"] = "water"
remap["87"] = "water"
remap["60"] = "steep_terrain"
remap["5"] = "dense_forest"

with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    json.dump(remap, f)
    remap_path = f.name

# Force CLIP fallback by passing a non-existent model name and fallback="clip_zero_shot"
segmentor = TerrainSegmentor(
    model_name="__nonexistent_model__",   # will fail → triggers fallback
    ade20k_remap_path=remap_path,
    device=torch.device(DEVICE),
    fallback="clip_zero_shot",
)

dummy_images = torch.randint(0, 256, (B, 3, H, W)).float()
terrain_map = segmentor.segment(dummy_images, H, W)

assert terrain_map.shape == (B, H, W), f"Expected ({B},{H},{W}), got {terrain_map.shape}"
assert terrain_map.dtype == torch.int64,  f"Expected int64, got {terrain_map.dtype}"
assert terrain_map.max() < NUM_TERRAIN_CLASSES, f"Terrain ID out of range: {terrain_map.max()}"
print(f"  terrain_map.shape = {tuple(terrain_map.shape)}  ✓")
print(f"  Unique terrain IDs: {terrain_map.unique().tolist()}")

# ---------------------------------------------------------------------------
# 2. ValidityMaskGenerator — shape and value checks
# ---------------------------------------------------------------------------
print("\n[2] ValidityMaskGenerator ...")
import tempfile as _tmpf

constraints = {
    "_comment": "test",
    "armored_vehicle": {"forbidden": ["water", "dense_forest", "steep_terrain"]},
    "default":         {"forbidden": ["water"]},
}
with _tmpf.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    json.dump(constraints, f)
    constraints_path = f.name

from seeing_unseen.terrain.validity_mask import ValidityMaskGenerator

vmg = ValidityMaskGenerator(
    constraints_path=constraints_path,
    terrain_segmentor=segmentor,
    dilation_px=3,
    min_valid_area_frac=0.0,  # disable 0-area warning in this test
)

mask = vmg.generate(dummy_images, military_class="armored_vehicle")

assert mask.shape == (B, 1, H, W), f"Expected ({B},1,{H},{W}), got {mask.shape}"
assert mask.dtype == torch.float32,  f"Expected float32, got {mask.dtype}"
unique_vals = mask.unique().tolist()
for v in unique_vals:
    assert v in (0.0, 1.0), f"Mask contains non-binary value: {v}"
print(f"  mask.shape   = {tuple(mask.shape)}  ✓")
print(f"  mask dtype   = {mask.dtype}  ✓")
print(f"  mask values  = {unique_vals}  ✓ (binary)")

# ---------------------------------------------------------------------------
# 3. All-water terrain → complete exclusion for armored_vehicle
# ---------------------------------------------------------------------------
print("\n[3] All-water terrain → full exclusion for armored_vehicle ...")
all_water = torch.full((B, H, W), fill_value=3, dtype=torch.long)  # 3 = water
mask_water = vmg.generate(
    dummy_images, military_class="armored_vehicle",
    precomputed_terrain=all_water,
)
# All pixels are water → should all be 0 after dilation
# (some edge pixels might remain 1 if dilation doesn't fully cover the map,
#  but since it's already all-forbidden, dilation of forbidden should stay 1)
# The validity mask should be entirely 0 (forbidden)
assert mask_water.max().item() == 0.0, (
    f"Expected all-zero mask for all-water terrain, "
    f"but max = {mask_water.max().item()}"
)
print(f"  All-water mask max = {mask_water.max().item()}  ✓ (fully excluded)")

# ---------------------------------------------------------------------------
# 4. All-flat terrain → zero exclusion for default class
# ---------------------------------------------------------------------------
print("\n[4] All-flat terrain → no exclusion for 'default' class ...")
all_flat = torch.full((B, H, W), fill_value=1, dtype=torch.long)  # 1 = flat_open
mask_flat = vmg.generate(
    dummy_images, military_class="default",
    precomputed_terrain=all_flat,
)
assert mask_flat.min().item() == 1.0, (
    f"Expected all-ones mask on flat terrain for 'default', "
    f"but min = {mask_flat.min().item()}"
)
print(f"  All-flat mask min = {mask_flat.min().item()}  ✓ (fully valid)")

# ---------------------------------------------------------------------------
# 5. MaskedAffordanceModel — end-to-end forward
# ---------------------------------------------------------------------------
print("\n[5] MaskedAffordanceModel end-to-end forward ...")
from seeing_unseen.terrain.validity_mask import MaskedAffordanceModel

# Use a small fake terrain_cfg dict (not DictConfig to keep this script simple)
terrain_cfg_dict = {
    "enabled": True,
    "apply_during_training": False,
    "segmentor": {
        "model_name": "__nonexistent__",
        "fallback": "clip_zero_shot",
        "ade20k_remap_path": remap_path,
    },
    "constraints_path": constraints_path,
    "morphology": {"dilation_px": 3, "min_valid_area_frac": 0.0},
}

model = MaskedAffordanceModel(
    input_shape=(3, H, W),
    target_input_shape=(3, 128, 128),
    base_model_name="remote_clip_unet",
    terrain_cfg=terrain_cfg_dict,
    remote_clip_cfg={"checkpoint_path": None},   # random weights
)
model.eval()

batch = {
    "image":          dummy_images,
    "target_query":   torch.randn(B, 1024),   # CLIP text embedding dim for RN50
    "military_class": "armored_vehicle",
}

with torch.no_grad():
    out = model(batch=batch)

assert "affordance"    in out, "Missing 'affordance' in output"
assert "validity_mask" in out, "Missing 'validity_mask' in output (should be exposed)"
assert out["affordance"].shape    == (B, 1, H, W), f"affordance shape mismatch"
assert out["validity_mask"].shape == (B, 1, H, W), f"validity_mask shape mismatch"

# The mask must zero out affordance where invalid
aff_max_on_forbidden = (out["affordance"] * (1 - out["validity_mask"])).abs().max()
assert aff_max_on_forbidden == 0.0, (
    f"Affordance is non-zero on forbidden terrain! max={aff_max_on_forbidden}"
)
print(f"  affordance.shape    = {tuple(out['affordance'].shape)}  ✓")
print(f"  validity_mask.shape = {tuple(out['validity_mask'].shape)}  ✓")
print(f"  Affordance on forbidden terrain = {aff_max_on_forbidden:.6f}  ✓ (zero)")

# ---------------------------------------------------------------------------
# Clean up temp files
# ---------------------------------------------------------------------------
os.unlink(remap_path)
os.unlink(constraints_path)

print("\n" + "=" * 65)
print("All terrain guardrailing smoke tests passed ✓")
print("=" * 65)
