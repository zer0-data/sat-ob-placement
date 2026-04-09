"""
Smoke test for RemoteCLIP encoder integration.
================================================
Verifies output shapes for RemoteCLIPRN50Encoder, RemoteCLIPViTEncoder,
RemoteCLIPUNet, and RemoteCLIPViTUNet using random weights (no checkpoint needed).

Run from the repo root:
    python scripts/test_remote_clip_encoder.py
"""

import sys
import os

# Ensure the package is importable from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

print("=" * 60)
print("RemoteCLIP shape smoke tests (random weights, CPU)")
print("=" * 60)

DEVICE = "cpu"
B = 2  # batch size

# ---------------------------------------------------------------------------
# 1. RemoteCLIPRN50Encoder — prepool mode
# ---------------------------------------------------------------------------
print("\n[1] RemoteCLIPRN50Encoder (backbone_type='prepool') ...")
from seeing_unseen.models.encoders.remote_clip_encoder import RemoteCLIPRN50Encoder

enc_rn50 = RemoteCLIPRN50Encoder(
    input_shape=(3, 480, 640),
    backbone_type="prepool",
    checkpoint_path=None,   # random weights
).to(DEVICE)
enc_rn50.eval()

dummy_img = torch.randint(0, 256, (B, 3, 224, 224)).float().to(DEVICE)
feat, im_feats = enc_rn50(dummy_img)

assert feat.shape == (B, 2048, 7, 7), f"Expected (B,2048,7,7), got {feat.shape}"
assert len(im_feats) == 8, f"Expected 8 intermediate feats, got {len(im_feats)}"
print(f"  feat.shape   = {tuple(feat.shape)}  ✓")
print(f"  len(im_feats)= {len(im_feats)}  ✓")
print(f"  im_feat dims = {[tuple(f.shape) for f in im_feats]}")

# ---------------------------------------------------------------------------
# 2. RemoteCLIPRN50Encoder — none mode (target encoder)
# ---------------------------------------------------------------------------
print("\n[2] RemoteCLIPRN50Encoder (backbone_type='none') ...")
enc_rn50_none = RemoteCLIPRN50Encoder(
    input_shape=(3, 128, 128),
    backbone_type="none",
    checkpoint_path=None,
).to(DEVICE)
enc_rn50_none.eval()

dummy_target = torch.randint(0, 256, (B, 3, 224, 224)).float().to(DEVICE)
feat_t, feats_t = enc_rn50_none(dummy_target)

assert feat_t.shape == (B, 2048), f"Expected (B,2048), got {feat_t.shape}"
assert feats_t == [], f"Expected empty list in 'none' mode"
print(f"  feat.shape = {tuple(feat_t.shape)}  ✓")

# ---------------------------------------------------------------------------
# 3. RemoteCLIPViTEncoder — ViT-B-32
# ---------------------------------------------------------------------------
print("\n[3] RemoteCLIPViTEncoder (ViT-B-32) ...")
from seeing_unseen.models.encoders.remote_clip_encoder import RemoteCLIPViTEncoder

enc_vit = RemoteCLIPViTEncoder(
    model_type="ViT-B-32",
    checkpoint_path=None,
    image_size=224,
).to(DEVICE)
enc_vit.eval()

dummy_vit = torch.randint(0, 256, (B, 3, 224, 224)).float().to(DEVICE)
cls_feat, spatial_feats = enc_vit(dummy_vit)

assert cls_feat.shape == (B, 768), f"Expected (B,768), got {cls_feat.shape}"
assert len(spatial_feats) == 4, f"Expected 4 skip grids, got {len(spatial_feats)}"
for i, g in enumerate(spatial_feats):
    assert g.shape == (B, 768, 7, 7), f"Grid {i}: expected (B,768,7,7), got {g.shape}"
print(f"  cls_feat.shape    = {tuple(cls_feat.shape)}  ✓")
print(f"  len(spatial_feats)= {len(spatial_feats)}  ✓")
print(f"  grid shapes       = {[tuple(g.shape) for g in spatial_feats]}")

# ---------------------------------------------------------------------------
# 4. RemoteCLIPUNet — full model forward (text-query variant)
# ---------------------------------------------------------------------------
print("\n[4] RemoteCLIPUNet.forward (text-query, RN50) ...")
from seeing_unseen.models.clip_unet import RemoteCLIPUNet

model_rn50 = RemoteCLIPUNet(
    input_shape=(3, 480, 640),
    target_input_shape=(3, 128, 128),
    remote_clip_cfg={"checkpoint_path": None},
).to(DEVICE)
model_rn50.eval()

H, W = 480, 640
batch = {
    "image":        torch.randint(0, 256, (B, 3, H, W)).float().to(DEVICE),
    # target_query: text CLIP embedding, shape (B, 1024) for RN50 CLIP text encoder
    "target_query": torch.randn(B, 1024).to(DEVICE),
}

with torch.no_grad():
    out = model_rn50(batch=batch)

assert "affordance" in out, "Output missing 'affordance' key"
assert out["affordance"].shape == (B, 1, H, W), (
    f"Expected (B,1,{H},{W}), got {out['affordance'].shape}"
)
print(f"  affordance.shape = {tuple(out['affordance'].shape)}  ✓")

# ---------------------------------------------------------------------------
# 5. RemoteCLIPViTUNet — full model forward
# ---------------------------------------------------------------------------
print("\n[5] RemoteCLIPViTUNet.forward (ViT-B-32) ...")
from seeing_unseen.models.clip_unet import RemoteCLIPViTUNet

model_vit = RemoteCLIPViTUNet(
    input_shape=(3, 480, 640),
    target_input_shape=(3, 128, 128),
    remote_clip_cfg={
        "model_type": "ViT-B-32",
        "checkpoint_path": None,
    },
).to(DEVICE)
model_vit.eval()

batch_vit = {
    "image":        torch.randint(0, 256, (B, 3, H, W)).float().to(DEVICE),
    # target_query: ViT CLS embedding, shape (B, 768)
    "target_query": torch.randn(B, 768).to(DEVICE),
}

with torch.no_grad():
    out_vit = model_vit(batch=batch_vit)

assert "affordance" in out_vit, "Output missing 'affordance' key"
assert out_vit["affordance"].shape == (B, 1, H, W), (
    f"Expected (B,1,{H},{W}), got {out_vit['affordance'].shape}"
)
print(f"  affordance.shape = {tuple(out_vit['affordance'].shape)}  ✓")

# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("All shape smoke tests passed ✓")
print("=" * 60)
