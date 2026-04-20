"""
prepare_mvrsd.py
================
Download the MVRSD aerial dataset from Kaggle, convert its YOLO-format labels
to the SemanticPlacement dataset format, and generate RemoteCLIP text embeddings.

Pipeline
--------
1. Download MVRSD from Kaggle via kagglehub
2. Read classes.txt → list of military class names
3. For each split (train / val):
     a. Walk data/images/{split}/ → find every image
     b. Parse matching YOLO label file (data/labels/{split}/*.txt)
     c. Convert each YOLO bbox → binary mask → COCO RLE
     d. Build split_records.json  ← consumed by SemanticPlacementTextQueryDataset
4. Generate remote_clip_embeddings.pkl using RemoteCLIPTextEncoder
   (one embedding per class using 5 satellite-domain prompt templates)

Output layout  (under --out_dir, default: data/datasets/mvrsd/)
-------
    data/datasets/mvrsd/
      train/
        train_records.json
        remote_clip_embeddings.pkl
      val/
        val_records.json
        remote_clip_embeddings.pkl

Usage
-----
    # Minimal — generate records only (no RemoteCLIP embeddings)
    python scripts/prepare_mvrsd.py

    # Full — also generate RemoteCLIP embeddings
    python scripts/prepare_mvrsd.py \
        --rc_checkpoint data/pretrained_models/RemoteCLIP-RN50.pt \
        --out_dir       data/datasets/mvrsd/

    # Skip download if already cached
    python scripts/prepare_mvrsd.py --skip_download --kaggle_cache /path/to/cached/
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Repo root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_classes(classes_txt: str) -> List[str]:
    """Return list of class names from YOLO classes.txt (one name per line)."""
    with open(classes_txt, "r") as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes


def yolo_to_bbox(
    cx_n: float, cy_n: float, w_n: float, h_n: float,
    img_w: int, img_h: int,
) -> Tuple[int, int, int, int]:
    """
    Convert normalised YOLO centre-format to absolute pixel bbox.
    Returns (x1, y1, x2, y2) clipped to image bounds.
    """
    cx = cx_n * img_w
    cy = cy_n * img_h
    bw = w_n  * img_w
    bh = h_n  * img_h

    x1 = int(np.clip(cx - bw / 2, 0, img_w - 1))
    y1 = int(np.clip(cy - bh / 2, 0, img_h - 1))
    x2 = int(np.clip(cx + bw / 2, 0, img_w))
    y2 = int(np.clip(cy + bh / 2, 0, img_h))
    return x1, y1, x2, y2


def bbox_to_rle(
    x1: int, y1: int, x2: int, y2: int,
    img_h: int, img_w: int,
) -> Dict:
    """
    Fill a binary mask with 1s inside the bounding box and encode as COCO RLE.
    Returns a dict  {"counts": str, "size": [H, W]}  compatible with
    seeing_unseen.utils.utils.decode_rle_mask().
    """
    from seeing_unseen.utils.utils import binary_mask_to_rle  # lazy import

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return binary_mask_to_rle(mask)


def parse_yolo_label(
    label_path: str,
    img_w: int,
    img_h: int,
    classes: List[str],
) -> List[Dict]:
    """
    Parse a YOLO .txt label file.

    Returns a list of annotation dicts, one per object:
        {
            "object_category": "<class_name>|<class_id>",
            "segmentation":    <COCO RLE dict>,
            "bbox":            [x1, y1, x2, y2],   # pixel coords, for reference
        }
    """
    annotations = []

    if not os.path.isfile(label_path):
        return annotations

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx_n, cy_n, w_n, h_n = map(float, parts[1:5])

            if cls_id >= len(classes):
                continue  # guard against label/classes mismatch

            x1, y1, x2, y2 = yolo_to_bbox(cx_n, cy_n, w_n, h_n, img_w, img_h)

            # Skip degenerate boxes
            if x2 <= x1 or y2 <= y1:
                continue

            rle = bbox_to_rle(x1, y1, x2, y2, img_h, img_w)
            cls_name = classes[cls_id]

            annotations.append(
                {
                    "object_category": f"{cls_name}|{cls_id}",
                    "segmentation": rle,
                    "bbox": [x1, y1, x2, y2],
                }
            )

    return annotations


def build_split_records(
    images_dir: str,
    labels_dir: str,
    classes: List[str],
    split: str,
    img_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff"),
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Walk images_dir, pair with labels_dir, build records list.

    Returns (records, category_counts) where records is a list of:
        {
            "img_path":    str,
            "annotations": List[Dict],
        }
    """
    records = []
    category_counts: Dict[str, int] = defaultdict(int)
    skipped_no_label = 0
    skipped_no_annotations = 0

    image_paths = sorted(
        p for p in Path(images_dir).rglob("*")
        if p.suffix.lower() in img_extensions
    )

    for img_path in tqdm(image_paths, desc=f"[{split}] building records"):
        # Derive label path: same stem, .txt extension, in labels_dir
        label_path = Path(labels_dir) / (img_path.stem + ".txt")

        if not label_path.exists():
            skipped_no_label += 1
            continue

        # Get image dimensions
        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size   # PIL gives (width, height)
        except Exception:
            continue

        annotations = parse_yolo_label(
            str(label_path), img_w, img_h, classes
        )

        if not annotations:
            skipped_no_annotations += 1
            continue

        for ann in annotations:
            cls_name = ann["object_category"].split("|")[0]
            category_counts[cls_name] += 1

        records.append(
            {
                "img_path":    str(img_path.resolve()),
                "annotations": annotations,
            }
        )

    print(
        f"[{split}] {len(records)} records  |  "
        f"no label: {skipped_no_label}  |  "
        f"no annotations: {skipped_no_annotations}"
    )
    return records, dict(category_counts)


def generate_remote_clip_embeddings(
    classes: List[str],
    checkpoint_path: Optional[str],
    out_path: str,
) -> None:
    """
    Generate a RemoteCLIP text embedding for every class and save as pkl.
    The pkl is a dict  {class_name: np.ndarray (D,)}  matching the format
    expected by SemanticPlacementTextQueryDataset.
    """
    from seeing_unseen.models.encoders.remote_clip_text_encoder import (
        RemoteCLIPTextEncoder,
    )

    print(f"\n[embeddings] Generating RemoteCLIP embeddings for {len(classes)} classes...")
    print(f"[embeddings] Classes: {classes}")

    encoder = RemoteCLIPTextEncoder(
        model_name="RN50",
        checkpoint_path=checkpoint_path,
    )

    embeddings = encoder.encode_to_numpy(classes)

    # Also add underscore variants so lookup works regardless of tokenisation
    for cls in list(classes):
        cls_spaced = cls.replace("_", " ")
        if cls_spaced not in embeddings:
            embeddings[cls_spaced] = embeddings[cls]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"[embeddings] Saved {len(embeddings)} embeddings → {out_path}")
    for cls, emb in embeddings.items():
        print(f"             {cls:<30s}  shape={emb.shape}  norm={np.linalg.norm(emb):.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare MVRSD dataset for the RemoteCLIP placement pipeline"
    )
    p.add_argument(
        "--out_dir",
        default="data/datasets/mvrsd",
        help="Root output directory for prepared dataset",
    )
    p.add_argument(
        "--kaggle_dataset",
        default="mrproudysharma/mvrsd-aerial",
        help="Kaggle dataset slug",
    )
    p.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip Kaggle download; use --kaggle_cache as the dataset root",
    )
    p.add_argument(
        "--kaggle_cache",
        default=None,
        help="Path to already-downloaded MVRSD root (use with --skip_download)",
    )
    p.add_argument(
        "--rc_checkpoint",
        default=None,
        help="Path to RemoteCLIP-RN50.pt for embedding generation. "
             "If omitted, only split_records.json files are produced.",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Splits to process (default: train val)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Download dataset
    # ------------------------------------------------------------------
    if args.skip_download:
        if not args.kaggle_cache:
            raise ValueError("--kaggle_cache required when --skip_download is set")
        dataset_root = args.kaggle_cache
        print(f"[download] Skipping download; using cached root: {dataset_root}")
    else:
        try:
            import kagglehub
        except ImportError:
            raise ImportError(
                "kagglehub is required. Install with:\n"
                "  pip install kagglehub[pandas-datasets]"
            )
        print(f"[download] Downloading {args.kaggle_dataset} from Kaggle...")
        dataset_root = kagglehub.dataset_download(args.kaggle_dataset)
        print(f"[download] Downloaded to: {dataset_root}")

    # Locate the MVRSD_dataset/data sub-directory
    data_dir = Path(dataset_root)
    # Handle both direct download and nested path
    for candidate in [
        data_dir / "MVRSD_dataset" / "data",
        data_dir / "data",
        data_dir,
    ]:
        if (candidate / "images").exists():
            data_dir = candidate
            break
    print(f"[dataset]  Data root: {data_dir}")

    # ------------------------------------------------------------------
    # 2. Load class names
    # ------------------------------------------------------------------
    classes_txt = data_dir / "labels" / "classes.txt"
    if not classes_txt.exists():
        # Fallback: search recursively
        found = list(Path(dataset_root).rglob("classes.txt"))
        if not found:
            raise FileNotFoundError(
                f"classes.txt not found under {dataset_root}"
            )
        classes_txt = found[0]

    classes = load_classes(str(classes_txt))
    print(f"[classes]  {len(classes)} classes: {classes}")

    # ------------------------------------------------------------------
    # 3. Build split records
    # ------------------------------------------------------------------
    import json

    os.makedirs(args.out_dir, exist_ok=True)
    all_classes_seen: set = set()

    for split in args.splits:
        images_dir = data_dir / "images" / split
        labels_dir = data_dir / "labels" / split

        if not images_dir.exists():
            print(f"[{split}] images dir not found ({images_dir}), skipping.")
            continue
        if not labels_dir.exists():
            print(f"[{split}] labels dir not found ({labels_dir}), skipping.")
            continue

        records, cat_counts = build_split_records(
            str(images_dir), str(labels_dir), classes, split
        )
        all_classes_seen.update(cat_counts.keys())

        print(f"[{split}] Category distribution:")
        for cls, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
            print(f"         {cls:<30s} {cnt:>6d}")

        # Save records
        split_out_dir = Path(args.out_dir) / split
        split_out_dir.mkdir(parents=True, exist_ok=True)
        records_path = split_out_dir / f"{split}_records.json"
        with open(records_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"[{split}] Saved {len(records)} records → {records_path}")

    # ------------------------------------------------------------------
    # 4. Generate RemoteCLIP embeddings
    # ------------------------------------------------------------------
    # Use only the classes that actually appear in the data
    active_classes = sorted(all_classes_seen) if all_classes_seen else classes

    for split in args.splits:
        split_out_dir = Path(args.out_dir) / split
        if not split_out_dir.exists():
            continue

        emb_path = str(split_out_dir / "remote_clip_embeddings.pkl")

        if args.rc_checkpoint:
            generate_remote_clip_embeddings(
                classes=active_classes,
                checkpoint_path=args.rc_checkpoint,
                out_path=emb_path,
            )
        else:
            # Write a placeholder so the dataset class doesn't crash on import
            # (will use random embeddings — replace with real ones before training)
            print(
                f"\n[embeddings] WARNING — no --rc_checkpoint provided.\n"
                f"             Writing zero-filled placeholder embeddings to {emb_path}.\n"
                f"             Re-run with --rc_checkpoint to generate real embeddings."
            )
            placeholder = {cls: np.zeros(1024, dtype=np.float32) for cls in active_classes}
            # Also add space variants
            for cls in list(active_classes):
                placeholder[cls.replace("_", " ")] = placeholder[cls]

            split_out_dir.mkdir(parents=True, exist_ok=True)
            with open(emb_path, "wb") as f:
                pickle.dump(placeholder, f)
            print(f"[embeddings] Placeholder saved → {emb_path}")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MVRSD dataset preparation complete.")
    print(f"Output root : {args.out_dir}")
    print(f"Splits      : {args.splits}")
    print(f"Classes     : {active_classes}")
    print()
    print("Next steps:")
    print("  1. Update config/baseline/clip_unet.yaml:")
    print(f"       dataset.root_dir: \"{args.out_dir}\"")
    print(f"       dataset.val_dir:  \"{args.out_dir}\"")
    print(f"       dataset.embeddings_file: \"remote_clip_embeddings.pkl\"")
    if not args.rc_checkpoint:
        print()
        print("  2. Generate real embeddings (placeholder used now):")
        print("       python scripts/prepare_mvrsd.py \\")
        print("           --skip_download \\")
        print(f"           --kaggle_cache {dataset_root} \\")
        print("           --rc_checkpoint data/pretrained_models/RemoteCLIP-RN50.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
