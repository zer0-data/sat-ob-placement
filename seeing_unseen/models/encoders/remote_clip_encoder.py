"""
RemoteCLIP Image Encoders
=========================
Drop-in replacements for ResNetCLIPEncoder that use RemoteCLIP weights
(https://github.com/ChenDelong1999/RemoteCLIP) for satellite / remote-sensing imagery.

RemoteCLIP weights are distributed in OpenCLIP format on HuggingFace:
    https://huggingface.co/chendelong/RemoteCLIP

Supported model types:
    RN50      -> RemoteCLIPRN50Encoder    (drop-in for ResNetCLIPEncoder, same output contract)
    ViT-B-32  -> RemoteCLIPViTEncoder     (new class, produces spatial grid features for decoder)
    ViT-L-14  -> RemoteCLIPViTEncoder

Download checkpoints before use:
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="chendelong/RemoteCLIP",
        filename="RemoteCLIP-RN50.pt",          # or RemoteCLIP-ViT-B-32.pt / RemoteCLIP-ViT-L-14.pt
        local_dir="data/pretrained_models",
    )
"""

import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import transforms as T

try:
    import open_clip
except ImportError as e:
    raise ImportError(
        "open-clip-torch is required for RemoteCLIP encoders. "
        "Install it with: pip install open-clip-torch>=2.20.0"
    ) from e

from seeing_unseen.core.logger import logger


# ---------------------------------------------------------------------------
# Shared pre-pool hook for RN50 backbones (same logic as clip_encoder.py)
# ---------------------------------------------------------------------------

def _forward_prepool_rn50(self, x):
    """
    Monkey-patched forward for an OpenCLIP / CLIP RN50 visual backbone.

    Returns
    -------
    x           : (B, 2048, 7, 7) — final spatial feature map before pooling
    im_feats    : list of intermediate feature maps (stem × 4 + layer1-4 = 8 tensors)
                  Index mapping (matches original clip_encoder.py):
                    0-2  : stem conv outputs
                    3    : stem avgpool output
                    4    : layer1 output
                    5    : layer2 output
                    6    : layer3 output
                    7    : layer4 output  ← used as the top-level feature in decoder
    """
    im_feats: List[torch.Tensor] = []

    # open_clip >= 2.20 renamed the stem activations from `relu{1,2,3}`
    # to `act{1,2,3}` (generalised from ReLU to configurable activation).
    # Support both names.
    act1 = getattr(self, "relu1", None) or getattr(self, "act1")
    act2 = getattr(self, "relu2", None) or getattr(self, "act2")
    act3 = getattr(self, "relu3", None) or getattr(self, "act3")

    def stem(x):
        for conv, bn, act in [
            (self.conv1, self.bn1, act1),
            (self.conv2, self.bn2, act2),
            (self.conv3, self.bn3, act3),
        ]:
            x = act(bn(conv(x)))
            im_feats.append(x)
        x = self.avgpool(x)
        im_feats.append(x)
        return x

    x = x.type(self.conv1.weight.dtype)
    x = stem(x)
    for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
        x = layer(x)
        im_feats.append(x)
    return x, im_feats


# ---------------------------------------------------------------------------
# RemoteCLIPRN50Encoder
# ---------------------------------------------------------------------------

class RemoteCLIPRN50Encoder(nn.Module):
    """
    Drop-in replacement for ResNetCLIPEncoder using RemoteCLIP RN50 weights.

    Output contract (identical to ResNetCLIPEncoder with backbone_type="prepool"):
        forward(batch) -> (feat, im_feats)
            feat      : (B, 2048, 7, 7)    top-level spatial feature map
            im_feats  : list[Tensor]  8 intermediate feature maps for skip connections

    For backbone_type="none" (used by the target encoder in CLIPUNetImgQuery):
        forward(batch) -> (feat, [])
            feat      : (B, 2048)          global average-pooled embedding
    """

    # OpenCLIP normalization constants for RN50 (same as openai/clip)
    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD  = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        input_shape: Tuple,
        backbone_type: str = "prepool",
        checkpoint_path: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        input_shape     : (H, W, C) or (C, H, W) — only used to decide whether
                          to add a resize transform before preprocessing.
        backbone_type   : "prepool" → multi-scale spatial features
                          "none"    → global pooled embedding (for target encoder)
        checkpoint_path : path to RemoteCLIP-RN50.pt checkpoint.
                          If None, the backbone is initialised with random weights
                          (useful for shape-only smoke tests).
        """
        super().__init__()

        self.backbone_type = backbone_type

        # ------------------------------------------------------------------
        # 1. Create an OpenCLIP RN50 model (architecture only, no pretrained weights)
        # ------------------------------------------------------------------
        model, _, preprocess = open_clip.create_model_and_transforms(
            "RN50",
            pretrained=False,   # we load RemoteCLIP weights below
        )

        # ------------------------------------------------------------------
        # 2. Load RemoteCLIP checkpoint into the visual encoder
        # ------------------------------------------------------------------
        if checkpoint_path is not None:
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"RemoteCLIP checkpoint not found at '{checkpoint_path}'. "
                    "Download it from https://huggingface.co/chendelong/RemoteCLIP"
                )
            logger.info(f"Loading RemoteCLIP-RN50 weights from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            # Checkpoint may be the full CLIP model or just visual encoder
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            # Filter to visual encoder keys from full model checkpoint.
            # Only keep keys that belong to the visual encoder — exclude
            # text encoder, logit_scale, and other non-visual keys.
            visual_sd = {}
            for k, v in state_dict.items():
                clean_k = k.replace("module.", "")
                if clean_k.startswith("visual."):
                    visual_sd[clean_k.replace("visual.", "", 1)] = v
            # Try loading into model.visual; fall back to full model
            try:
                missing, unexpected = model.visual.load_state_dict(
                    visual_sd, strict=False
                )
                logger.info(
                    f"RemoteCLIP-RN50 visual encoder loaded. "
                    f"Missing: {missing[:5]}, Unexpected: {unexpected[:5]}"
                )
            except Exception:
                # If keys don't match directly (full model ckpt), load full model
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                logger.info(
                    f"RemoteCLIP-RN50 full model loaded. "
                    f"Missing: {missing[:5]}, Unexpected: {unexpected[:5]}"
                )
        else:
            logger.warning(
                "No checkpoint_path provided for RemoteCLIPRN50Encoder — "
                "using random weights (suitable only for shape tests)."
            )

        # ------------------------------------------------------------------
        # 3. Extract visual backbone and configure output shape
        # ------------------------------------------------------------------
        self.backbone = model.visual

        if "none" in backbone_type:
            # Replace attention-pool with global average pool → (B, 2048)
            self.backbone.attnpool = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten(),
            )
            self.output_shape = (2048, 1)
        else:
            # "prepool": expose pre-attnpool spatial features (B, 2048, 7, 7)
            self.output_shape = (2048, 7, 7)
            # Monkey-patch forward to return (spatial_feat, [im_feats])
            bound_method = _forward_prepool_rn50.__get__(
                self.backbone, self.backbone.__class__
            )
            setattr(self.backbone, "forward", bound_method)

        # ------------------------------------------------------------------
        # 4. Freeze backbone
        # ------------------------------------------------------------------
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # ------------------------------------------------------------------
        # 5. Preprocessing transforms (no resize — caller handles that)
        # ------------------------------------------------------------------
        H, W = (input_shape[0], input_shape[1]) if len(input_shape) >= 2 else (224, 224)
        resize_transforms = []
        if H != 224 or W != 224:
            logger.info(
                f"RemoteCLIPRN50Encoder: input {H}×{W} ≠ 224×224; "
                "adding resize+center-crop to 224."
            )
            resize_transforms = [
                T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
            ]
        self.resize_transforms = T.Compose(resize_transforms)

        self.preprocess = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=list(self.MEAN), std=list(self.STD)),
        ])

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode(self, batch: torch.Tensor):
        """Core encode — always runs with frozen backbone."""
        if "none" in self.backbone_type:
            out = self.backbone(batch)
            return out.to(torch.float32), []
        out, im_feats = self.backbone(batch)
        return out.to(torch.float32), [f.to(torch.float32) for f in im_feats]

    def forward(
        self,
        batch: torch.Tensor,
        apply_resize_tfms: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Parameters
        ----------
        batch             : (B, C, H, W) float tensor, values in [0, 255]
        apply_resize_tfms : if True, apply resize+crop before normalising

        Returns
        -------
        feat      : (B, 2048, 7, 7) or (B, 2048) depending on backbone_type
        im_feats  : list of intermediate feature maps (empty for "none" mode)
        """
        if apply_resize_tfms and len(self.resize_transforms.transforms) > 0:
            batch = torch.stack([self.resize_transforms(img) for img in batch])
        batch = self.preprocess(batch)
        return self._encode(batch)

    @property
    def clip_prepool(self) -> bool:
        return "prepool" in self.backbone_type


# ---------------------------------------------------------------------------
# RemoteCLIPViTEncoder
# ---------------------------------------------------------------------------

# Channel widths for each backbone at the intermediate layers we tap:
_VIT_LAYER_DIMS = {
    "ViT-B-32": 768,
    "ViT-L-14": 1024,
}

# For skip connections we grab every quarter of the transformer depth
_VIT_SKIP_FRACTIONS = [0.25, 0.5, 0.75, 1.0]


class RemoteCLIPViTEncoder(nn.Module):
    """
    RemoteCLIP ViT encoder (ViT-B-32 or ViT-L-14).

    Unlike the RN50 encoder, ViT backbones do not naturally expose
    hierarchical spatial feature maps.  We hook into every 1/4-depth
    transformer block and reshape the patch tokens into a (B, C, H, W)
    spatial grid so that a UNet decoder can consume them as skip connections.

    Output contract:
        forward(batch) -> (cls_feat, spatial_feats)
            cls_feat        : (B, D)                   CLS-token embedding
            spatial_feats   : list of 4 × (B, D, H, W) spatial grids
                              (coarse→fine, smallest H*W first)

    Where D = 768 for ViT-B-32, 1024 for ViT-L-14.
    H = W = image_size // patch_size  (e.g. 224//32 = 7 for ViT-B-32).
    """

    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD  = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        model_type: str = "ViT-B-32",
        checkpoint_path: Optional[str] = None,
        image_size: int = 224,
    ):
        """
        Parameters
        ----------
        model_type      : "ViT-B-32" or "ViT-L-14"
        checkpoint_path : path to RemoteCLIP ViT checkpoint.
        image_size      : input resolution fed to encoder (must be 224 for ViT-B-32).
        """
        super().__init__()

        if model_type not in _VIT_LAYER_DIMS:
            raise ValueError(
                f"Unsupported ViT model_type '{model_type}'. "
                f"Choose from {list(_VIT_LAYER_DIMS.keys())}"
            )

        self.model_type = model_type
        self.embed_dim = _VIT_LAYER_DIMS[model_type]

        # ------------------------------------------------------------------
        # 1. Build OpenCLIP ViT model
        # ------------------------------------------------------------------
        model, _, _ = open_clip.create_model_and_transforms(
            model_type, pretrained=False
        )

        # ------------------------------------------------------------------
        # 2. Load RemoteCLIP checkpoint
        # ------------------------------------------------------------------
        if checkpoint_path is not None:
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"RemoteCLIP checkpoint not found at '{checkpoint_path}'."
                )
            logger.info(f"Loading RemoteCLIP-{model_type} weights from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            logger.info(
                f"RemoteCLIP-{model_type} loaded. "
                f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
            )
        else:
            logger.warning(
                f"No checkpoint_path for RemoteCLIPViTEncoder({model_type}); "
                "using random weights."
            )

        self.backbone = model.visual

        # Determine patch grid size
        patch_size = int(model_type.split("-")[-1])  # e.g. 32 or 14
        self.grid_size = image_size // patch_size    # e.g. 7 for 224/32

        # Number of transformer blocks; pick skip layers at ~25%, 50%, 75%, 100%
        depth = len(self.backbone.transformer.resblocks)
        self.skip_layer_indices = [
            max(0, int(f * depth) - 1) for f in _VIT_SKIP_FRACTIONS
        ]
        logger.info(
            f"RemoteCLIPViTEncoder: depth={depth}, "
            f"skip layers={self.skip_layer_indices}, grid={self.grid_size}×{self.grid_size}"
        )

        # ------------------------------------------------------------------
        # 3. Register forward hooks on selected transformer blocks
        # ------------------------------------------------------------------
        self._hook_features: List[torch.Tensor] = []
        self._hooks = []

        for idx in self.skip_layer_indices:
            block = self.backbone.transformer.resblocks[idx]
            hook = block.register_forward_hook(self._make_hook())
            self._hooks.append(hook)

        # ------------------------------------------------------------------
        # 4. Output shape info
        # ------------------------------------------------------------------
        self.output_shape = (self.embed_dim,)   # CLS token dim
        # skip connection shapes: each is (embed_dim, grid_size, grid_size)
        self.skip_shapes = [(self.embed_dim, self.grid_size, self.grid_size)] * 4

        # ------------------------------------------------------------------
        # 5. Freeze
        # ------------------------------------------------------------------
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # ------------------------------------------------------------------
        # 6. Preprocessing
        # ------------------------------------------------------------------
        self.preprocess = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=list(self.MEAN), std=list(self.STD)),
        ])

    def _make_hook(self):
        def hook(module, input, output):
            # output shape: (seq_len, B, embed_dim) in OpenCLIP ViT
            # seq_len = 1 + grid_size^2 (CLS + patch tokens)
            self._hook_features.append(output)
        return hook

    def _patch_tokens_to_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert patch token sequence to (B, C, H, W) spatial grid.

        tokens : (seq_len, B, embed_dim)  — first token is CLS
        """
        # Drop CLS token, transpose to (B, N, C)
        patch_tokens = tokens[1:].permute(1, 0, 2)     # (B, N, C)
        B, N, C = patch_tokens.shape
        H = W = self.grid_size
        assert N == H * W, f"Expected {H*W} patch tokens, got {N}"
        # Reshape to (B, H, W, C) then to (B, C, H, W)
        spatial = patch_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return spatial.contiguous().to(torch.float32)

    def forward(
        self,
        batch: torch.Tensor,
        apply_resize_tfms: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns
        -------
        cls_feat      : (B, embed_dim)
        spatial_feats : list of 4 × (B, embed_dim, grid_H, grid_W)
                        ordered coarse→fine (depth 25% → 100%)
        """
        self._hook_features.clear()

        batch = self.preprocess(batch)

        with torch.no_grad():
            cls_feat = self.backbone(batch)     # triggers hooks
            cls_feat = cls_feat.to(torch.float32)

        # Convert hooked token sequences to spatial grids
        spatial_feats = [
            self._patch_tokens_to_spatial(f) for f in self._hook_features
        ]
        self._hook_features.clear()

        return cls_feat, spatial_feats

    def remove_hooks(self):
        """Call this if you no longer need the encoder, to free hook memory."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
