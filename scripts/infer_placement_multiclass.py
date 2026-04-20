"""
infer_placement_multiclass.py
=============================
End-to-end multi-class inference: satellite image + list of class specs
→ class-aware placement bounding boxes.

Pipeline (see docs/multi_class_pipeline.md)
-------------------------------------------
  1. Load image
  2. Replicate image to (N_classes, 3, H, W) and run one forward pass with
     target_query = [label_1, ..., label_N]. Produces per-class heatmaps.
  3. Apply validity mask (class-agnostic).
  4. MultiClassBBoxProposer: calibrate → argmax+threshold → connected
     components → tile → class-agnostic NMS → caps.
  5. Save annotated image + JSON + (optional) debug artifacts.

Classes are supplied via --classes_json, pointing at a JSON file of the form:
    [
        {"name": "S-400",  "box_w": 80,  "box_h": 40, "threshold": 0.35, "max_boxes": 5},
        {"name": "tank",   "box_w": 24,  "box_h": 16, "threshold": 0.30, "max_boxes": 10},
        {"name": "hangar", "box_w": 180, "box_h": 90, "max_boxes": 3}
    ]

Usage
-----
    python scripts/infer_placement_multiclass.py \
        --image        path/to/satellite.tif \
        --classes_json config/classes.json \
        --checkpoint   path/to/ckpt.pth \
        --out_dir      outputs/inference_mc/
"""

import argparse
import colorsys
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

# ── Repo root on sys.path ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seeing_unseen.core.registry import registry  # noqa: E402
from seeing_unseen.models import clip_unet         # noqa: F401  (registers models)
from seeing_unseen.terrain import validity_mask    # noqa: F401  (registers masked model)
from seeing_unseen.placement import (              # noqa: E402
    Box,
    ClassSpec,
    MultiClassBBoxProposer,
    MultiClassResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image_tensor(path: str, device: torch.device) -> torch.Tensor:
    """Load an image as (1, 3, H, W) float32 tensor with values in [0, 255]."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def load_class_specs(path: str) -> List[ClassSpec]:
    with open(path, "r") as f:
        entries = json.load(f)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: expected a non-empty JSON list of class specs.")
    specs = []
    for i, e in enumerate(entries):
        if "name" not in e or "box_w" not in e or "box_h" not in e:
            raise ValueError(
                f"{path}[{i}]: each entry must have name, box_w, box_h."
            )
        specs.append(
            ClassSpec(
                name=str(e["name"]),
                box_w=int(e["box_w"]),
                box_h=int(e["box_h"]),
                threshold=(float(e["threshold"]) if "threshold" in e else None),
                max_boxes=(int(e["max_boxes"]) if "max_boxes" in e else None),
            )
        )
    return specs


def build_model(
    model_name: str,
    checkpoint_path: str,
    remote_clip_cfg: dict,
    terrain_cfg: dict,
    device: torch.device,
) -> torch.nn.Module:
    model_cls = registry.get_affordance_model(model_name)
    if model_cls is None:
        raise ValueError(
            f"Model '{model_name}' not found. "
            f"Available: {list(registry._affordance_model_registry.keys())}"
        )

    kwargs = {}
    if model_name.startswith("remote_clip"):
        kwargs["remote_clip_cfg"] = remote_clip_cfg
    if model_name == "remote_clip_unet_masked":
        kwargs["terrain_cfg"] = terrain_cfg

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


def run_model_multiclass(
    model: torch.nn.Module,
    image: torch.Tensor,
    labels: List[str],
    device: torch.device,
    per_class_batch: int,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Run one (or more) batched forward(s) across all classes.

    Returns
    -------
    heatmaps      : {label: (1, H, W) float32 in [0, 1]}
    validity_mask : (1, H, W) float32 binary (ones if model does not expose one)
    """
    image_norm = (image.to(device) / 255.0)            # (1, 3, H, W)
    _, _, H, W = image.shape

    heatmaps: Dict[str, torch.Tensor] = {}
    validity_ref: torch.Tensor = torch.ones((1, H, W), device=device)

    for start in range(0, len(labels), per_class_batch):
        chunk = labels[start : start + per_class_batch]
        n = len(chunk)
        batch_img = image_norm.expand(n, -1, -1, -1).contiguous()

        batch = {
            "image": batch_img,
            "target_query": list(chunk),
            "military_class": chunk[0],   # scalar field; not used by non-masked models
        }

        with torch.no_grad():
            output = model(batch=batch)

        logit = output["affordance"]                   # (n, 1, h, w)
        hm = torch.sigmoid(logit)                      # (n, 1, h, w)
        if hm.shape[-2:] != (H, W):
            hm = F.interpolate(hm, size=(H, W), mode="bilinear", align_corners=False)

        # Validity mask: same for every class — capture once.
        validity = output.get("validity_mask", None)
        if validity is not None:
            v = validity
            if v.dim() == 3:
                v = v.unsqueeze(0) if v.shape[0] != n else v
            if v.shape[-2:] != (H, W):
                v = F.interpolate(v.float(), size=(H, W), mode="nearest")
            # All rows should be identical; take row 0.
            validity_ref = v[0:1].squeeze(0).unsqueeze(0).float()   # (1, H, W)

        for i, label in enumerate(chunk):
            heatmaps[label] = hm[i].detach()           # (1, H, W)

    # Apply validity to each heatmap (matches single-class behavior).
    for label in list(heatmaps.keys()):
        heatmaps[label] = heatmaps[label] * validity_ref

    return heatmaps, validity_ref


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _class_color(idx: int, n: int) -> Tuple[int, int, int]:
    h = (idx / max(1, n)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def annotate_image(
    image_path: str,
    result: MultiClassResult,
    class_specs: List[ClassSpec],
    out_path: str,
) -> None:
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    overlay = img.copy()

    # Per-class argmax overlay (colored by class).
    class_map = result.class_map
    if class_map.shape != (H, W):
        cm_img = Image.fromarray(
            ((class_map + 1).astype(np.uint8)), mode="L"
        ).resize((W, H), resample=Image.NEAREST)
        class_map_vis = np.array(cm_img).astype(np.int32) - 1
    else:
        class_map_vis = class_map

    rgb = np.array(overlay, dtype=np.uint8)
    colors = [_class_color(i, len(class_specs)) for i in range(len(class_specs))]
    mask_rgb = np.zeros_like(rgb)
    for cid in range(len(class_specs)):
        sel = class_map_vis == cid
        if sel.any():
            mask_rgb[sel] = colors[cid]
    blended = (0.6 * rgb + 0.4 * mask_rgb).clip(0, 255).astype(np.uint8)
    overlay = Image.fromarray(blended, mode="RGB")

    draw = ImageDraw.Draw(overlay)
    name_to_color = {s.name: colors[i] for i, s in enumerate(class_specs)}
    for rank, b in enumerate(result.boxes):
        color = name_to_color.get(b.class_name, (255, 255, 255))
        hex_color = "#{:02X}{:02X}{:02X}".format(*color)
        draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=hex_color, width=2)
        draw.text(
            (b.x1 + 2, b.y1 + 2),
            f"#{rank + 1} {b.class_name} {b.score:.3f}",
            fill=hex_color,
        )

    # Legend
    draw.text(
        (8, 8),
        f"{len(result.boxes)} boxes across {len(class_specs)} class(es)",
        fill="white",
    )
    for i, spec in enumerate(class_specs):
        hex_color = "#{:02X}{:02X}{:02X}".format(*colors[i])
        draw.text((8, 24 + 14 * i), f"■ {spec.name}", fill=hex_color)

    overlay.save(out_path)
    print(f"[viz]  Saved annotated image → {out_path}")


def save_json(
    result: MultiClassResult,
    class_specs: List[ClassSpec],
    image_path: str,
    out_path: str,
) -> None:
    payload = {
        "image": image_path,
        "classes": [
            {
                "name": s.name,
                "box_w": s.box_w,
                "box_h": s.box_h,
                "threshold": s.threshold,
                "max_boxes": s.max_boxes,
            }
            for s in class_specs
        ],
        "boxes": [
            {
                "rank": i + 1,
                "class": b.class_name,
                "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
                "score": round(b.score, 5),
            }
            for i, b in enumerate(result.boxes)
        ],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[json] Saved results → {out_path}")


def save_debug_artifacts(
    result: MultiClassResult,
    class_specs: List[ClassSpec],
    out_dir: str,
    stem: str,
) -> None:
    debug_dir = os.path.join(out_dir, f"{stem}_debug")
    os.makedirs(debug_dir, exist_ok=True)

    # Argmax map (colored)
    H, W = result.class_map.shape
    colors = [_class_color(i, len(class_specs)) for i in range(len(class_specs))]
    cmap_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cid in range(len(class_specs)):
        cmap_rgb[result.class_map == cid] = colors[cid]
    Image.fromarray(cmap_rgb, mode="RGB").save(
        os.path.join(debug_dir, "class_map.png")
    )

    # Per-class normalized heatmaps + binary masks
    for spec in class_specs:
        norm = result.normalized.get(spec.name)
        if norm is not None:
            arr = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)
            Image.fromarray(arr, mode="L").save(
                os.path.join(debug_dir, f"{spec.name}_normalized.png")
            )
        mask = result.binary_masks.get(spec.name)
        if mask is not None:
            Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(
                os.path.join(debug_dir, f"{spec.name}_mask.png")
            )

    print(f"[debug] Saved debug artifacts → {debug_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-class satellite placement inference")

    # Required
    p.add_argument("--image",         required=True)
    p.add_argument("--classes_json",  required=True,
                   help="JSON file listing class specs (see module docstring).")

    # Output
    p.add_argument("--out_dir",       default="outputs/inference_mc")
    p.add_argument("--save_debug",    action="store_true",
                   help="Save per-class normalized heatmaps, binary masks, class map.")

    # Model
    p.add_argument("--checkpoint",    default=None)
    p.add_argument("--model_name",    default="remote_clip_unet",
                   choices=["remote_clip_unet", "remote_clip_vit_unet",
                            "remote_clip_unet_masked", "clip_unet"])
    p.add_argument("--rc_checkpoint",
                   default="data/pretrained_models/RemoteCLIP-RN50.pt")
    p.add_argument("--per_class_batch", type=int, default=4,
                   help="How many classes to forward in one batched call "
                        "(trade GPU memory vs speed).")

    # Proposer
    p.add_argument("--default_threshold", type=float, default=0.35)
    p.add_argument("--normalization",     default="percentile",
                   choices=["percentile", "zscore", "minmax", "none"])
    p.add_argument("--percentile_lo",     type=float, default=50.0)
    p.add_argument("--percentile_hi",     type=float, default=99.0)
    p.add_argument("--min_blob_area",     type=int,   default=25)
    p.add_argument("--tile_stride_frac",  type=float, default=0.5)
    p.add_argument("--min_coverage",      type=float, default=0.3)
    p.add_argument("--score_fn",          default="mean",
                   choices=["mean", "max", "sum"])
    p.add_argument("--nms_iou",           type=float, default=0.3)
    p.add_argument("--max_boxes_total",   type=int,   default=None)

    # Terrain
    p.add_argument("--no_terrain",        action="store_true")
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

    # ── Class specs ───────────────────────────────────────────────────
    class_specs = load_class_specs(args.classes_json)
    print(f"[cls]   Loaded {len(class_specs)} class(es): "
          f"{[s.name for s in class_specs]}")

    # ── Configs ───────────────────────────────────────────────────────
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

    # ── Model ─────────────────────────────────────────────────────────
    print(f"[model] Loading '{args.model_name}' ...")
    model = build_model(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        remote_clip_cfg=remote_clip_cfg,
        terrain_cfg=terrain_cfg,
        device=device,
    )

    # ── Image ─────────────────────────────────────────────────────────
    print(f"[image] Loading {args.image}")
    image = load_image_tensor(args.image, device)
    print(f"[image] Shape: {tuple(image.shape)}")

    # ── Inference ─────────────────────────────────────────────────────
    labels = [s.name for s in class_specs]
    print(f"[infer] Running forward(s) for {len(labels)} class(es), "
          f"per_class_batch={args.per_class_batch} ...")
    heatmaps, validity = run_model_multiclass(
        model=model,
        image=image,
        labels=labels,
        device=device,
        per_class_batch=args.per_class_batch,
    )
    for lbl, hm in heatmaps.items():
        print(f"[infer]  {lbl:>20s}: min={hm.min():.3f} max={hm.max():.3f} "
              f"mean={hm.mean():.3f}")

    # ── Proposer ──────────────────────────────────────────────────────
    proposer = MultiClassBBoxProposer(
        class_specs=class_specs,
        default_threshold=args.default_threshold,
        normalization=args.normalization,
        percentile_lo=args.percentile_lo,
        percentile_hi=args.percentile_hi,
        min_blob_area=args.min_blob_area,
        tile_stride_frac=args.tile_stride_frac,
        min_coverage=args.min_coverage,
        score_fn=args.score_fn,
        nms_iou_thresh=args.nms_iou,
        max_boxes_total=args.max_boxes_total,
    )
    print(f"[bbox]  {proposer}")

    result = proposer.propose(heatmaps=heatmaps, validity_mask=validity)

    if not result.boxes:
        print("[bbox]  WARNING — no boxes produced. "
              "Try lowering thresholds, reducing min_coverage, "
              "or widening the percentile range.")
    else:
        print(f"[bbox]  {len(result.boxes)} box(es) returned:")
        per_class_count: Dict[str, int] = {}
        for rank, b in enumerate(result.boxes):
            per_class_count[b.class_name] = per_class_count.get(b.class_name, 0) + 1
            print(f"         #{rank + 1:>2d}  [{b.class_name:>15s}] "
                  f"({b.x1},{b.y1}) → ({b.x2},{b.y2})  score={b.score:.4f}")
        print(f"[bbox]  Per-class counts: {per_class_count}")

    # ── Save ──────────────────────────────────────────────────────────
    annotate_image(
        image_path=args.image,
        result=result,
        class_specs=class_specs,
        out_path=os.path.join(args.out_dir, f"{stem}_multiclass_annotated.png"),
    )
    save_json(
        result=result,
        class_specs=class_specs,
        image_path=args.image,
        out_path=os.path.join(args.out_dir, f"{stem}_multiclass_boxes.json"),
    )
    if args.save_debug:
        save_debug_artifacts(result, class_specs, args.out_dir, stem)

    print("\n[done] Multi-class inference complete.")


if __name__ == "__main__":
    main()
