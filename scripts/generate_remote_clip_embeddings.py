"""
Generate Remote-Sensing Prompt-Conditioned Text Embeddings
===========================================================
Reads an existing clip_embeddings.pkl (whose keys are raw category strings),
re-encodes each category using RemoteCLIP's text encoder + satellite-domain
prompt templates, and writes a new remote_clip_embeddings.pkl in the same
format {category: np.ndarray (D,)}.

Usage
-----
# With RemoteCLIP checkpoint (recommended):
python scripts/generate_remote_clip_embeddings.py \
    --input  data/datasets/semantic_placement/train/clip_embeddings.pkl \
    --output data/datasets/semantic_placement/train/remote_clip_embeddings.pkl \
    --checkpoint data/pretrained_models/RemoteCLIP-RN50.pt

# Without checkpoint (random weights — shape validation only):
python scripts/generate_remote_clip_embeddings.py \
    --input  data/datasets/semantic_placement/train/clip_embeddings.pkl \
    --output data/datasets/semantic_placement/train/remote_clip_embeddings.pkl

# Custom templates via JSON file:
python scripts/generate_remote_clip_embeddings.py \
    --input  ... --output ... \
    --templates templates.json   # JSON list of format strings with {c}

# Process multiple splits at once:
python scripts/generate_remote_clip_embeddings.py \
    --input data/datasets/semantic_placement/train/clip_embeddings.pkl \
            data/datasets/semantic_placement/val/clip_embeddings.pkl \
    --output data/datasets/semantic_placement/train/remote_clip_embeddings.pkl \
             data/datasets/semantic_placement/val/remote_clip_embeddings.pkl
"""

import argparse
import json
import os
import pickle
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm

from seeing_unseen.models.encoders.remote_clip_text_encoder import (
    REMOTE_SENSING_TEMPLATES,
    RemoteCLIPTextEncoder,
)


def load_pkl(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"  Saved → {path}  ({len(data)} categories)")


def process_file(
    input_path: str,
    output_path: str,
    encoder: RemoteCLIPTextEncoder,
    overwrite: bool = False,
) -> None:
    if os.path.exists(output_path) and not overwrite:
        print(f"  [skip] {output_path} already exists (use --overwrite to regenerate)")
        return

    print(f"\nProcessing: {input_path}")
    old_embeddings = load_pkl(input_path)
    categories = list(old_embeddings.keys())
    print(f"  Categories found: {len(categories)}")

    # Print a few examples to verify
    sample = categories[:5]
    print(f"  Sample categories: {sample}")
    for cat in sample:
        prompts = encoder.formatter.format(cat)
        print(f"    '{cat}' → '{prompts[0]}' ... '{prompts[-1]}'")

    print(f"  Encoding with {len(encoder.templates)} templates × {len(categories)} categories ...")
    new_embeddings = encoder.encode_to_numpy(categories, batch_size=32)

    # Sanity check: embeddings should be unit-norm
    import numpy as np
    norms = [np.linalg.norm(v) for v in new_embeddings.values()]
    print(f"  Embedding norms — mean: {float(np.mean(norms)):.4f}, "
          f"std: {float(np.std(norms)):.6f} (should be ≈1.0)")

    # Compare old vs new embedding for first category (cosine similarity)
    first_cat = categories[0]
    old_emb = np.array(old_embeddings[first_cat], dtype=np.float32)
    new_emb = new_embeddings[first_cat]
    old_norm = old_emb / (np.linalg.norm(old_emb) + 1e-8)
    cos_sim = float(np.dot(old_norm, new_emb))
    print(f"  Cosine sim (old CLIP vs new RemoteCLIP) for '{first_cat}': {cos_sim:.4f}")

    save_pkl(new_embeddings, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Re-encode clip_embeddings.pkl using RemoteCLIP + prompt templates"
    )
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="Path(s) to input clip_embeddings.pkl files"
    )
    parser.add_argument(
        "--output", nargs="+", required=True,
        help="Path(s) to output remote_clip_embeddings.pkl files (same count as --input)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to RemoteCLIP-RN50.pt checkpoint "
             "(download from https://huggingface.co/chendelong/RemoteCLIP)"
    )
    parser.add_argument(
        "--model", type=str, default="RN50",
        help="OpenCLIP architecture: RN50 | ViT-B-32 | ViT-L-14"
    )
    parser.add_argument(
        "--templates", type=str, default=None,
        help="Path to JSON file containing a list of template strings with {c} placeholder. "
             "Defaults to the 5 built-in satellite templates."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output files"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Number of categories to encode per batch"
    )
    args = parser.parse_args()

    if len(args.input) != len(args.output):
        parser.error("--input and --output must have the same number of paths")

    # ------------------------------------------------------------------
    # Load templates
    # ------------------------------------------------------------------
    templates: Optional[List[str]] = None
    if args.templates is not None:
        with open(args.templates) as f:
            templates = json.load(f)
        print(f"Loaded {len(templates)} custom templates from {args.templates}")
    else:
        templates = REMOTE_SENSING_TEMPLATES
        print(f"Using {len(templates)} built-in satellite templates:")
        for t in templates:
            print(f"  • {t}")

    # ------------------------------------------------------------------
    # Build encoder
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Architecture: {args.model}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    else:
        print("Checkpoint: NONE (random weights — for shape testing only)")

    encoder = RemoteCLIPTextEncoder(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        device=device,
        templates=templates,
    )

    # ------------------------------------------------------------------
    # Process each file pair
    # ------------------------------------------------------------------
    for inp, out in zip(args.input, args.output):
        process_file(inp, out, encoder, overwrite=args.overwrite)

    print("\nDone. ✓")
    print("To use the new embeddings, update your config:")
    print("  dataset:")
    print("    embeddings_file: \"remote_clip_embeddings.pkl\"")


if __name__ == "__main__":
    main()
