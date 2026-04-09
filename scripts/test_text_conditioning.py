"""
Smoke test for Text Conditioning (Prompt Engineering).
=======================================================
Verifies PromptFormatter, RemoteCLIPTextEncoder (random weights),
and the on-the-fly fallback in RemoteCLIPUNet — no checkpoint download needed.

Run from the repo root:
    python scripts/test_text_conditioning.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

print("=" * 60)
print("Text Conditioning smoke tests (random weights, CPU)")
print("=" * 60)

B = 2
DEVICE = "cpu"

# ---------------------------------------------------------------------------
# 1. PromptFormatter
# ---------------------------------------------------------------------------
print("\n[1] PromptFormatter ...")
from seeing_unseen.models.encoders.remote_clip_text_encoder import (
    REMOTE_SENSING_TEMPLATES,
    PromptFormatter,
    RemoteCLIPTextEncoder,
)

fmt = PromptFormatter()

prompts = fmt.format("tank")
assert len(prompts) == 5, f"Expected 5 prompts, got {len(prompts)}"
assert all("{c}" not in p for p in prompts), "Template placeholder not expanded!"
assert "an aerial view of a tank"       in prompts
assert "a satellite image of a tank"    in prompts
assert "a top-down view of a tank"      in prompts
assert "a remote sensing image of a tank" in prompts
assert "a bird's eye view of a tank"    in prompts
print(f"  Prompts for 'tank':")
for p in prompts:
    print(f"    • {p}")
print("  ✓ All 5 templates expanded correctly")

# Batch
batch_prompts = fmt.format_batch(["tank", "radar"])
assert len(batch_prompts) == 2
assert len(batch_prompts[0]) == 5
print(f"  format_batch(['tank','radar']) → {len(batch_prompts)} × {len(batch_prompts[0])} prompts  ✓")

# Custom templates
custom_fmt = PromptFormatter(templates=["overhead view of {c}", "nadir view of {c}"])
custom = custom_fmt.format("radar")
assert custom == ["overhead view of radar", "nadir view of radar"]
print(f"  Custom templates work ✓")

# ---------------------------------------------------------------------------
# 2. RemoteCLIPTextEncoder shape + normalisation
# ---------------------------------------------------------------------------
print("\n[2] RemoteCLIPTextEncoder (random weights) ...")
enc = RemoteCLIPTextEncoder(
    model_name="RN50",
    checkpoint_path=None,   # random weights
    device=torch.device(DEVICE),
)

categories = ["tank", "radar", "helicopter pad", "S-400", "observation post"]
embeddings = enc.encode(categories)

assert embeddings.shape == (5, enc.embed_dim), (
    f"Expected (5, {enc.embed_dim}), got {embeddings.shape}"
)
print(f"  embeddings.shape = {tuple(embeddings.shape)}  ✓")

# Check L2 norm ≈ 1.0 for each row
norms = embeddings.norm(dim=-1)
assert (norms - 1.0).abs().max().item() < 1e-5, (
    f"Embeddings not unit-norm: max deviation = {(norms-1.0).abs().max().item():.2e}"
)
print(f"  L2 norms = {norms.tolist()}  ✓ (all ≈ 1.0)")

# encode_to_numpy
numpy_dict = enc.encode_to_numpy(["tank", "radar"])
assert set(numpy_dict.keys()) == {"tank", "radar"}
assert numpy_dict["tank"].shape == (enc.embed_dim,)
assert isinstance(numpy_dict["tank"], np.ndarray)
print(f"  encode_to_numpy keys: {list(numpy_dict.keys())}  ✓")

# ---------------------------------------------------------------------------
# 3. Bare-noun vs prompted embedding should differ
# ---------------------------------------------------------------------------
print("\n[3] Bare-noun vs satellite-prompted embedding ...")
bare_enc = RemoteCLIPTextEncoder(
    model_name="RN50",
    checkpoint_path=None,
    device=torch.device(DEVICE),
    templates=["{c}"],   # bare noun (no template)
)
prompted_enc = RemoteCLIPTextEncoder(
    model_name="RN50",
    checkpoint_path=None,
    device=torch.device(DEVICE),
    templates=REMOTE_SENSING_TEMPLATES,
)
# With random weights they will differ due to different tokenisation
bare_emb     = bare_enc.encode(["tank"])
prompted_emb = prompted_enc.encode(["tank"])
cos_sim = float(F.cosine_similarity(bare_emb, prompted_emb).item())
print(f"  Cosine sim (bare vs prompted): {cos_sim:.4f}  ✓ (should be < 1.0 with random weights)")
assert cos_sim < 1.0, "Bare and prompted embeddings are identical — unexpected"

# ---------------------------------------------------------------------------
# 4. RemoteCLIPUNet on-the-fly string encoding
# ---------------------------------------------------------------------------
print("\n[4] RemoteCLIPUNet forward with string target_query (on-the-fly) ...")
from seeing_unseen.models.clip_unet import RemoteCLIPUNet

model = RemoteCLIPUNet(
    input_shape=(3, 480, 640),
    target_input_shape=(3, 128, 128),
    remote_clip_cfg={"checkpoint_path": None},
).eval()

# Pass strings instead of pre-computed tensors — on-the-fly path
batch_str = {
    "image":          torch.randint(0, 256, (B, 3, 480, 640)).float(),
    "target_query":  ["armored vehicle", "radar installation"],  # list[str]
}
with torch.no_grad():
    out_str = model(batch=batch_str)

assert out_str["affordance"].shape == (B, 1, 480, 640)
print(f"  affordance.shape (string input) = {tuple(out_str['affordance'].shape)}  ✓")

# Pre-computed tensor path still works
batch_tensor = {
    "image":          torch.randint(0, 256, (B, 3, 480, 640)).float(),
    "target_query":   torch.randn(B, 1024),   # pre-computed embedding
}
with torch.no_grad():
    out_tensor = model(batch=batch_tensor)

assert out_tensor["affordance"].shape == (B, 1, 480, 640)
print(f"  affordance.shape (tensor input) = {tuple(out_tensor['affordance'].shape)}  ✓")

# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("All text conditioning smoke tests passed ✓")
print("=" * 60)
