"""
RemoteCLIP Text Encoder with Satellite-Domain Prompt Templates
==============================================================
Replaces bare-noun CLIP text embeddings (e.g. "chair") with
remote-sensing prompt ensembles (e.g. "an aerial view of a chair").

RemoteCLIP's text encoder was trained on richly annotated remote sensing
captions; framing queries in satellite vernacular significantly improves
alignment with top-down imagery features.

Public API
----------
REMOTE_SENSING_TEMPLATES   : list[str]  — the 5 default prompt templates
PromptFormatter            : stateless utility for expanding categories
RemoteCLIPTextEncoder      : produces ensemble-averaged text embeddings
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from seeing_unseen.core.logger import logger


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

REMOTE_SENSING_TEMPLATES: List[str] = [
    "an aerial view of a {c}",
    "a satellite image of a {c}",
    "a top-down view of a {c}",
    "a remote sensing image of a {c}",
    "a bird's eye view of a {c}",
]


# ---------------------------------------------------------------------------
# PromptFormatter — stateless
# ---------------------------------------------------------------------------

class PromptFormatter:
    """
    Expand a category name into a list of prompted strings.

    Example
    -------
    >>> fmt = PromptFormatter()
    >>> fmt.format("tank")
    ['an aerial view of a tank',
     'a satellite image of a tank',
     'a top-down view of a tank',
     'a remote sensing image of a tank',
     "a bird's eye view of a tank"]
    """

    def __init__(self, templates: Optional[List[str]] = None):
        self.templates = templates if templates is not None else REMOTE_SENSING_TEMPLATES

    def format(self, category: str) -> List[str]:
        """Return one prompted string per template for `category`."""
        return [t.format(c=category) for t in self.templates]

    def format_batch(self, categories: List[str]) -> List[List[str]]:
        """Return a list-of-lists: outer = categories, inner = templates."""
        return [self.format(c) for c in categories]


# ---------------------------------------------------------------------------
# RemoteCLIPTextEncoder
# ---------------------------------------------------------------------------

class RemoteCLIPTextEncoder(torch.nn.Module):
    """
    Encodes object category names into L2-normalised text embeddings using
    RemoteCLIP's text encoder and a configurable prompt-template ensemble.

    Each category is encoded as the **mean of N template embeddings**
    (N = len(templates)), each individually L2-normalised before averaging,
    followed by another L2-normalisation of the mean — consistent with the
    retrieval evaluation in the RemoteCLIP paper.

    Parameters
    ----------
    model_name       : OpenCLIP architecture string, e.g. "RN50"
    checkpoint_path  : path to RemoteCLIP .pt file (None = random weights)
    device           : torch.device
    templates        : list of format strings with a ``{c}`` placeholder
    """

    def __init__(
        self,
        model_name: str = "RN50",
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        templates: Optional[List[str]] = None,
    ):
        super().__init__()

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.formatter = PromptFormatter(templates)
        self.templates = self.formatter.templates

        # ------------------------------------------------------------------
        # Load OpenCLIP model
        # ------------------------------------------------------------------
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "open-clip-torch is required. Install: pip install open-clip-torch>=2.20.0"
            ) from exc

        logger.info(f"RemoteCLIPTextEncoder: architecture={model_name}")
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=False
        )
        self.tokenize = open_clip.tokenize

        # Embedding dimension
        with torch.no_grad():
            dummy = self.tokenize(["test"]).to("cpu")
            self.embed_dim = model.encode_text(dummy).shape[-1]

        # ------------------------------------------------------------------
        # Load RemoteCLIP checkpoint (optional)
        # ------------------------------------------------------------------
        if checkpoint_path is not None:
            import os
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"RemoteCLIP checkpoint not found: '{checkpoint_path}'"
                )
            logger.info(f"Loading RemoteCLIP text weights from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            logger.info(
                f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}"
            )
        else:
            logger.warning(
                "RemoteCLIPTextEncoder: no checkpoint_path — using random weights. "
                "Suitable only for shape testing."
            )

        # Keep the full model but we only use the text tower
        self.text_model = model.to(self.device)
        for param in self.text_model.parameters():
            param.requires_grad = False
        self.text_model.eval()

        logger.info(
            f"RemoteCLIPTextEncoder ready. "
            f"dim={self.embed_dim}, templates={len(self.templates)}"
        )

    # ------------------------------------------------------------------
    # Core: encode a single category string
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_one(self, category: str) -> torch.Tensor:
        """
        Encode one category → L2-normalised ensemble embedding  (D,).
        """
        prompts = self.formatter.format(category)
        tokens = self.tokenize(prompts).to(self.device)

        per_prompt = self.text_model.encode_text(tokens).float()   # (T, D)
        per_prompt = F.normalize(per_prompt, dim=-1)               # normalise each
        ensemble   = per_prompt.mean(dim=0)                        # average
        ensemble   = F.normalize(ensemble, dim=-1)                 # re-normalise
        return ensemble                                            # (D,)

    # ------------------------------------------------------------------
    # Batch encode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        categories: List[str],
        batch_size: int = 64,
    ) -> torch.Tensor:
        """
        Encode a list of categories → (N, D) float32 tensor.

        Parameters
        ----------
        categories : list of category strings
        batch_size : number of categories per forward pass

        Returns
        -------
        embeddings : (N, D) float32, L2-normalised
        """
        results = []
        for i in range(0, len(categories), batch_size):
            chunk = categories[i : i + batch_size]
            # Build a flat list of all prompts for this chunk
            all_prompts: List[str] = []
            for c in chunk:
                all_prompts.extend(self.formatter.format(c))

            T = len(self.templates)  # prompts per category
            tokens = self.tokenize(all_prompts).to(self.device)

            feats = self.text_model.encode_text(tokens).float()    # (len(chunk)*T, D)
            feats = F.normalize(feats, dim=-1)
            feats = feats.view(len(chunk), T, -1).mean(dim=1)      # (len(chunk), D)
            feats = F.normalize(feats, dim=-1)
            results.append(feats.cpu())

        return torch.cat(results, dim=0)   # (N, D)

    def encode_to_numpy(
        self,
        categories: List[str],
        batch_size: int = 64,
    ) -> Dict[str, np.ndarray]:
        """
        Convenience wrapper: returns dict {category: np.ndarray (D,)}.
        Matches the format expected by dataset.py clip_embeddings.pkl.
        """
        embeddings_tensor = self.encode(categories, batch_size)
        return {
            cat: embeddings_tensor[i].numpy()
            for i, cat in enumerate(categories)
        }

    def forward(self, categories: List[str]) -> torch.Tensor:
        """Alias for encode(), returns tensor on self.device."""
        return self.encode(categories).to(self.device)
