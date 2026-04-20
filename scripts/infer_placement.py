"""
infer_placement.py
==================
End-to-end inference: satellite image + class label → placement bounding boxes.

Pipeline
--------
  1. Load image
  2. Run RemoteCLIPUNet (+ terrain validity mask if enabled)
  3. Apply sigmoid → affordance heatmap in [0, 1]
  4. BBoxProposer: peak detection → jitter → NMS → top-N boxes
  5. Save annotated image + JSON results

Usage
-----
    python scripts/infer_placement.py \
        --image      path/to/satellite.tif \
        --label      "S-400" \
        --checkpoint path/to/ckpt.pth \
        --box_w      80 \
        --box_h      40 \
        --n_boxes    5 \
        --out_dir    outputs/inference/

Optional flags
--------------
    --config          path to clip_unet.yaml (default: config/baseline/clip_unet.yaml)
    --model_name      remote_clip_unet | remote_clip_vit_unet | remote_clip_unet_masked
    --score_fn        mean | max | sum
    --nms_iou         NMS IoU threshold (default 0.3)
    --smooth_sigma    Gaussian smoothing sigma (default 3.0)
    --jitter_frac     jitter fraction of box size (default 0.5)
    --n_jitter        jittered candidates per peak (default 8)
    --n_peaks         local maxima to seed from (default 20)
    --no_terrain      disable terrain validity mask even if model supports it
    --seed            RNG seed for reproducible jitter
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T

# ── Make sure the repo root is on sys.path ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seeing_unseen.core.registry import registry  # noqa: E402 (after sys.path)
from seeing_unseen.models import clip_unet         # noqa: F401 – registers models
from seeing_unseen.terrain import validity_mask    # noqa: F401 – registers masked model
from seeing_unseen.placement import BBoxProposer   # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image_tensor(path: str, device: torch.device) -> torch.Tensor:
    """Load an image as (1, 3, H, W) float32 tensor with values in [0, 255]."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32)          # (H, W, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return tensor.to(device)


def build_model(
    model_name: str,
    checkpoint_path: str,
    remote_clip_cfg: dict,
    terrain_cfg: dict,
    device: torch.device,
) -> torch.nn.Module:
    """Instantiate and load the affordance model from a checkpoint."""
    model_cls = registry.get_affordance_model(model_name)
    if model_cls is None:
        raise ValueError(
            f"Model '{model_name}' not found in registry. "
            f"Available: {list(registry._affordance_model_registry.keys())}"
        )

    kwargs = {}
    if model_name.startswith("remote_clip"):
        kwargs["remote_clip_cfg"] = remote_clip_cfg
    if model_name == "remote_clip_unet_masked":
        kwargs["terrain_cfg"] = terrain_cfg

    # Input shapes — CLIP encoders resize internally to 224×224
    model = model_cls(
        input_shape=(3, 480, 640),
        target_input_shape=(3, 128, 128),
        **kwargs,
    ).to(device)
    model.eval()

    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("ckpt_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing = model.load_state_dict(state, strict=False)
        print(f"[model] Loaded checkpoint. Missing keys: {len(missing.missing_keys)}")
    else:
        print("[model] WARNING — no checkpoint loaded, using random weights.")

    return model


def run_model(
    model: torch.nn.Module,
    image: torch.Tensor,
    label: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns
    -------
    heatmap      : (1, H, W) float32, sigmoid-activated, in [0, 1]
    validity_mask: (1, H, W) float32 binary, or ones if not available
    """
    # Normalise to [0, 1] for model transforms (applied inside trainer.apply_transforms)
    # The model's CLIP encoder calls its own internal normalisation — feed raw [0,255]
    # Trainer's apply_transforms divides by 255 before the model sees the image.
    # The CLIP encoder's T.ConvertImageDtype(float32) is a no-op on float input
    # and assumes [0, 1], so we must normalise here.
    image_norm = image.to(device) / 255.0

    batch = {
        "image": image_norm,
        "target_query": [label],   # on-the-fly encoding via RemoteCLIPTextEncoder
        "military_class": label,
    }

    with torch.no_grad():
        output = model(batch=batch)

    logit = output["affordance"]                          # (1, 1, H, W)
    heatmap = torch.sigmoid(logit).squeeze(0)             # (1, H, W)

    # Resize to original image resolution
    _, _, H, W = image.shape
    heatmap = F.interpolate(
        heatmap.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
    ).squeeze(0)

    validity = output.get("validity_mask", torch.ones_like(heatmap))
    if validity.shape[-2:] != (H, W):
        validity = F.interpolate(
            validity if validity.dim() == 4 else validity.unsqueeze(0),
            size=(H, W), mode="nearest",
        ).squeeze(0)

    # Apply validity mask to heatmap
    heatmap = heatmap * validity

    return heatmap, validity


def annotate_image(
    image_path: str,
    boxes: List[Tuple[int, int, int, int, float]],
    heatmap: torch.Tensor,
    label: str,
    out_path: str,
) -> None:
    """Save original image overlaid with heatmap + bounding boxes."""
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    # --- Heatmap overlay ---
    hm_np = heatmap.squeeze().cpu().numpy()
    hm_np = (hm_np - hm_np.min()) / (hm_np.max() - hm_np.min() + 1e-8)
    hm_uint8 = (hm_np * 255).astype(np.uint8)
    hm_rgb = np.stack([hm_uint8, np.zeros_like(hm_uint8), np.zeros_like(hm_uint8)], axis=2)
    hm_pil = Image.fromarray(hm_rgb.astype(np.uint8), mode="RGB").resize((W, H))
    overlay = Image.blend(img, hm_pil, alpha=0.4)

    # --- Bounding boxes ---
    draw = ImageDraw.Draw(overlay)
    colors = ["#00FF00", "#00DDFF", "#FFD700", "#FF6600", "#FF00FF"]
    for rank, (x1, y1, x2, y2, score) in enumerate(boxes):
        color = colors[rank % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text(
            (x1 + 2, y1 + 2),
            f"#{rank+1} {score:.3f}",
            fill=color,
        )

    # --- Title ---
    draw.text((8, 8), f"Label: {label}  |  {len(boxes)} boxes", fill="white")

    overlay.save(out_path)
    print(f"[viz]  Saved annotated image → {out_path}")


def save_json(
    boxes: List[Tuple[int, int, int, int, float]],
    label: str,
    image_path: str,
    box_w: int,
    box_h: int,
    out_path: str,
) -> None:
    result = {
        "image": image_path,
        "label": label,
        "box_size": {"w": box_w, "h": box_h},
        "boxes": [
            {"rank": i + 1, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": round(score, 5)}
            for i, (x1, y1, x2, y2, score) in enumerate(boxes)
        ],
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[json] Saved results → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Satellite object placement inference")

    # Required
    p.add_argument("--image",      required=True,  help="Path to input satellite image")
    p.add_argument("--label",      required=True,  help="Object class label, e.g. 'S-400'")
    p.add_argument("--box_w",      required=True,  type=int, help="Bounding box width in pixels")
    p.add_argument("--box_h",      required=True,  type=int, help="Bounding box height in pixels")

    # Output
    p.add_argument("--out_dir",    default="outputs/inference", help="Output directory")

    # Model
    p.add_argument("--checkpoint",  default=None,  help="Path to model checkpoint (.pth)")
    p.add_argument("--model_name",  default="remote_clip_unet",
                   choices=["remote_clip_unet", "remote_clip_vit_unet",
                            "remote_clip_unet_masked", "clip_unet"],
                   help="Registered model name")
    p.add_argument("--rc_checkpoint", default="data/pretrained_models/RemoteCLIP-RN50.pt",
                   help="Path to RemoteCLIP encoder checkpoint")

    # BBoxProposer
    p.add_argument("--n_boxes",     default=5,   type=int,   help="Number of final boxes")
    p.add_argument("--n_peaks",     default=20,  type=int,   help="Local maxima to seed from")
    p.add_argument("--n_jitter",    default=8,   type=int,   help="Jittered candidates per peak")
    p.add_argument("--jitter_frac", default=0.5, type=float, help="Jitter fraction of box size")
    p.add_argument("--smooth_sigma",default=3.0, type=float, help="Gaussian smoothing sigma")
    p.add_argument("--nms_iou",     default=0.3, type=float, help="NMS IoU threshold")
    p.add_argument("--score_fn",    default="mean",
                   choices=["mean", "max", "sum"], help="Box scoring function")
    p.add_argument("--seed",        default=42,  type=int,   help="RNG seed for jitter")

    # Terrain
    p.add_argument("--no_terrain",  action="store_true",
                   help="Disable terrain validity mask")
    p.add_argument("--constraints_path",
                   default="data/metadata/deployment_constraints.json")
    p.add_argument("--ade20k_remap_path",
                   default="data/metadata/ade20k_to_terrain.json")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info]  Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)
    stem = Path(args.image).stem

    # ── Build config dicts ────────────────────────────────────────────
    remote_clip_cfg = {
        "model_type":    "RN50",
        "checkpoint_path": args.rc_checkpoint,
        "text_templates": [
            "an aerial view of a {c}",
            "a satellite image of a {c}",
            "a top-down view of a {c}",
            "a remote sensing image of a {c}",
            "a bird's eye view of a {c}",
        ],
    }

    terrain_cfg = None if args.no_terrain else {
        "enabled": True,
        "apply_during_training": False,
        "segmentor": {
            "model_name":        "nvidia/segformer-b0-finetuned-ade-512-512",
            "fallback":          "clip_zero_shot",
            "ade20k_remap_path": args.ade20k_remap_path,
        },
        "constraints_path": args.constraints_path,
        "morphology": {
            "dilation_px":         5,
            "min_valid_area_frac": 0.05,
        },
    }

    # ── Load model ────────────────────────────────────────────────────
    print(f"[model] Loading '{args.model_name}' ...")
    model = build_model(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        remote_clip_cfg=remote_clip_cfg,
        terrain_cfg=terrain_cfg,
        device=device,
    )

    # ── Load image ────────────────────────────────────────────────────
    print(f"[image] Loading {args.image}")
    image = load_image_tensor(args.image, device)
    print(f"[image] Shape: {tuple(image.shape)}")

    # ── Model forward ─────────────────────────────────────────────────
    print(f"[infer] Running inference for label: '{args.label}' ...")
    heatmap, validity = run_model(model, image, args.label, device)
    print(f"[infer] Heatmap: min={heatmap.min():.4f}  max={heatmap.max():.4f}  "
          f"mean={heatmap.mean():.4f}")

    # ── BBoxProposer ──────────────────────────────────────────────────
    proposer = BBoxProposer(
        box_w=args.box_w,
        box_h=args.box_h,
        n_boxes=args.n_boxes,
        n_peaks=args.n_peaks,
        n_jitter=args.n_jitter,
        jitter_frac=args.jitter_frac,
        smooth_sigma=args.smooth_sigma,
        nms_iou_thresh=args.nms_iou,
        score_fn=args.score_fn,
        seed=args.seed,
    )
    print(f"[bbox]  {proposer}")

    boxes = proposer.propose(heatmap)

    if not boxes:
        print("[bbox]  WARNING — no boxes produced. "
              "Try lowering --nms_iou, increasing --n_peaks, "
              "or checking the heatmap max value.")
    else:
        print(f"[bbox]  {len(boxes)} box(es) returned:")
        for rank, (x1, y1, x2, y2, score) in enumerate(boxes):
            print(f"         #{rank+1}  ({x1},{y1}) → ({x2},{y2})  score={score:.4f}")

    # ── Save outputs ──────────────────────────────────────────────────
    label_safe = args.label.replace(" ", "_").replace("/", "-")

    annotate_image(
        image_path=args.image,
        boxes=boxes,
        heatmap=heatmap,
        label=args.label,
        out_path=os.path.join(args.out_dir, f"{stem}_{label_safe}_annotated.png"),
    )

    save_json(
        boxes=boxes,
        label=args.label,
        image_path=args.image,
        box_w=args.box_w,
        box_h=args.box_h,
        out_path=os.path.join(args.out_dir, f"{stem}_{label_safe}_boxes.json"),
    )

    # Optionally save raw heatmap as greyscale PNG
    hm_np = heatmap.squeeze().cpu().numpy()
    hm_img = Image.fromarray((hm_np * 255).astype(np.uint8), mode="L")
    hm_path = os.path.join(args.out_dir, f"{stem}_{label_safe}_heatmap.png")
    hm_img.save(hm_path)
    print(f"[viz]  Saved raw heatmap   → {hm_path}")

    print("\n[done] Inference complete.")


if __name__ == "__main__":
    main()
