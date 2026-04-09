from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from seeing_unseen.core.logger import logger
from seeing_unseen.core.registry import registry
from seeing_unseen.models.base import SPModel
from seeing_unseen.models.encoders.clip_encoder import ResNetCLIPEncoder
from seeing_unseen.models.encoders.fusion import FusionConvLat, FusionMult
from seeing_unseen.models.encoders.remote_clip_encoder import (
    RemoteCLIPRN50Encoder,
    RemoteCLIPViTEncoder,
)
from seeing_unseen.models.encoders.remote_clip_text_encoder import (
    REMOTE_SENSING_TEMPLATES,
    RemoteCLIPTextEncoder,
)
from seeing_unseen.models.encoders.resnet import ConvBlock, IdentityBlock
from seeing_unseen.models.encoders.unet import Up


@registry.register_affordance_model(name="clip_unet_img_query")
class CLIPUNetImgQuery(SPModel):
    def __init__(
        self,
        input_shape: tuple,
        target_input_shape: tuple,
        output_dim: int = 1,
        upsample_factor: int = 2,
        bilinear: bool = True,
        batchnorm: bool = True,
    ) -> None:
        super().__init__()

        self.input_shape = input_shape
        self.target_input_shape = target_input_shape
        self.output_dim = output_dim
        self.upsample_factor = upsample_factor
        self.bilinear = bilinear
        self.batchnorm = batchnorm

        self.init_clip()
        self.init_target_encoder()
        self.init_decoder()
        self.train()

        self.activation = nn.Sigmoid()

    def init_clip(self):
        self.clip = ResNetCLIPEncoder(
            input_shape=self.input_shape,
            backbone_type="prepool",
            clip_model="RN50",
        )
        self.clip_out_dim = self.clip.output_shape[0]

    def init_target_encoder(self):
        self.target_encoder = ResNetCLIPEncoder(
            input_shape=self.target_input_shape,
            backbone_type="none",
            clip_model="RN50",
        )
        self.target_encoder_out_dim = self.target_encoder.output_shape[0]

    def init_decoder(self):
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                self.clip_out_dim, 1024, kernel_size=3, padding=1, bias=False
            ),
            nn.ReLU(inplace=True),
        )

        self.up1 = Up(
            self.clip_out_dim, 1024 // self.upsample_factor, self.bilinear
        )
        self.up2 = Up(1024, 512 // self.upsample_factor, self.bilinear)

        self.lin_proj = nn.Linear(self.target_encoder_out_dim, 256)
        self.target_fuser = FusionConvLat(input_dim=256 + 256, output_dim=256)
        self.up3 = Up(512, 256 // self.upsample_factor, self.bilinear)

        self.layer1 = nn.Sequential(
            ConvBlock(
                128,
                [64, 64, 64],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            IdentityBlock(
                64,
                [64, 64, 64],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            nn.UpsamplingBilinear2d(scale_factor=self.upsample_factor),
        )

        self.layer2 = nn.Sequential(
            ConvBlock(
                64,
                [32, 32, 32],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            IdentityBlock(
                32,
                [32, 32, 32],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            nn.UpsamplingBilinear2d(scale_factor=self.upsample_factor),
        )

        self.layer3 = nn.Sequential(
            ConvBlock(
                32,
                [16, 16, 16],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            IdentityBlock(
                16,
                [16, 16, 16],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            nn.UpsamplingBilinear2d(scale_factor=self.upsample_factor),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, self.output_dim, kernel_size=1)
        )

    def forward(self, **kwargs) -> torch.Tensor:
        # receptacle = receptacle.permute(0, 3, 1, 2) # BATCH x CHANNEL x HEIGHT X WIDTH
        # target = target.permute(0, 3, 1, 2) # BATCH x CHANNEL x HEIGHT X WIDTH
        batch = kwargs["batch"]
        target = batch["target_query"]
        receptacle = batch["image"]

        input_shape = receptacle.shape
        x, x_im_feats = self.clip(receptacle)

        target_embedding, _ = self.target_encoder(
            target, apply_resize_tfms=False
        )

        x = self.conv1(x)
        x = self.up1(x, x_im_feats[-2])

        x = self.up2(x, x_im_feats[-3])

        x = self.target_fuser(x, target_embedding, x2_proj=self.lin_proj)
        x = self.up3(x, x_im_feats[-4])

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)
        x = F.interpolate(
            x, size=(input_shape[-2], input_shape[-1]), mode="bilinear"
        )
        return x


@registry.register_affordance_model(name="clip_unet")
class CLIPUNet(CLIPUNetImgQuery):
    def __init__(
        self,
        input_shape: tuple,
        target_input_shape: tuple,
        output_dim: int = 1,
        upsample_factor: int = 2,
        bilinear: bool = True,
        batchnorm: bool = True,
    ) -> None:
        super().__init__(
            input_shape,
            target_input_shape,
            output_dim,
            upsample_factor,
            bilinear,
            batchnorm,
        )

    def init_target_encoder(self):
        self.target_encoder = None
        self.target_encoder_out_dim = 1024

    def init_discriminator(self):
        self.discriminator = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 13 * 18, 1024),
            nn.ReLU(),
        )
        self.discriminator_fc = nn.Linear(1024, 1)
        self.discriminator_out_dim = 1024

    def init_decoder(self):
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                self.clip_out_dim, 1024, kernel_size=3, padding=1, bias=False
            ),
            nn.ReLU(inplace=True),
        )

        self.lang_fuser1 = FusionMult(input_dim=self.clip_out_dim // 2)
        self.lang_fuser2 = FusionMult(input_dim=self.clip_out_dim // 4)
        self.lang_fuser3 = FusionMult(input_dim=self.clip_out_dim // 8)

        self.lang_proj1 = nn.Linear(self.target_encoder_out_dim, 1024)
        self.lang_proj2 = nn.Linear(self.target_encoder_out_dim, 512)
        self.lang_proj3 = nn.Linear(self.target_encoder_out_dim, 256)

        self.up1 = Up(
            self.clip_out_dim, 1024 // self.upsample_factor, self.bilinear
        )
        self.up2 = Up(1024, 512 // self.upsample_factor, self.bilinear)
        self.up3 = Up(512, 256 // self.upsample_factor, self.bilinear)

        self.layer1 = nn.Sequential(
            ConvBlock(
                128,
                [64, 64, 64],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            IdentityBlock(
                64,
                [64, 64, 64],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            nn.UpsamplingBilinear2d(scale_factor=self.upsample_factor),
        )

        self.layer2 = nn.Sequential(
            ConvBlock(
                64,
                [32, 32, 32],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            IdentityBlock(
                32,
                [32, 32, 32],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            nn.UpsamplingBilinear2d(scale_factor=self.upsample_factor),
        )

        self.layer3 = nn.Sequential(
            ConvBlock(
                32,
                [16, 16, 16],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            IdentityBlock(
                16,
                [16, 16, 16],
                kernel_size=3,
                stride=1,
                batchnorm=self.batchnorm,
            ),
            nn.UpsamplingBilinear2d(scale_factor=self.upsample_factor),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, self.output_dim, kernel_size=1)
        )

    def forward_encoder(self, image, query):
        x, x_im_feats = self.clip(image)

        x = self.conv1(x)

        x = self.lang_fuser1(x, query, x2_proj=self.lang_proj1)
        x = self.up1(x, x_im_feats[-2])

        x = self.lang_fuser2(x, query, x2_proj=self.lang_proj2)
        x = self.up2(x, x_im_feats[-3])

        x = self.lang_fuser3(x, query, x2_proj=self.lang_proj3)
        x = self.up3(x, x_im_feats[-4])
        return x

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        disc_out = None
        output = {}
        batch = kwargs["batch"]
        discriminator_only = kwargs.get("discriminator_only", False)
        target = batch["target_query"]
        receptacle = batch["image"]

        input_shape = receptacle.shape

        x = self.forward_encoder(receptacle, target)

        if not discriminator_only:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.conv2(x)
            x = F.interpolate(
                x, size=(input_shape[-2], input_shape[-1]), mode="bilinear"
            )
            output["affordance"] = x
        return output


# =============================================================================
# RemoteCLIP drop-in models
# =============================================================================

@registry.register_affordance_model(name="remote_clip_unet_img_query")
class RemoteCLIPUNetImgQuery(CLIPUNetImgQuery):
    """
    CLIPUNetImgQuery with RemoteCLIP RN50 image encoder (visual-query variant).

    Drop-in swap: only init_clip() and init_target_encoder() are overridden.
    The entire UNet decoder is inherited unchanged.

    Config keys consumed (under model.remote_clip in clip_unet.yaml):
        model_type       : "RN50" (default)
        checkpoint_path  : path to RemoteCLIP-RN50.pt
    """

    def __init__(
        self,
        input_shape: tuple,
        target_input_shape: tuple,
        output_dim: int = 1,
        upsample_factor: int = 2,
        bilinear: bool = True,
        batchnorm: bool = True,
        remote_clip_cfg: Optional[dict] = None,
    ) -> None:
        self._remote_clip_cfg = remote_clip_cfg or {}
        super().__init__(
            input_shape, target_input_shape, output_dim,
            upsample_factor, bilinear, batchnorm,
        )

    def init_clip(self):
        checkpoint_path = self._remote_clip_cfg.get("checkpoint_path", None)
        self.clip = RemoteCLIPRN50Encoder(
            input_shape=self.input_shape,
            backbone_type="prepool",
            checkpoint_path=checkpoint_path,
        )
        self.clip_out_dim = self.clip.output_shape[0]   # 2048
        logger.info("RemoteCLIPUNetImgQuery: using RemoteCLIP RN50 image encoder.")

    def init_target_encoder(self):
        checkpoint_path = self._remote_clip_cfg.get("checkpoint_path", None)
        self.target_encoder = RemoteCLIPRN50Encoder(
            input_shape=self.target_input_shape,
            backbone_type="none",
            checkpoint_path=checkpoint_path,
        )
        self.target_encoder_out_dim = self.target_encoder.output_shape[0]   # 2048


@registry.register_affordance_model(name="remote_clip_unet")
class RemoteCLIPUNet(CLIPUNet):
    """
    CLIPUNet (text-query variant) with RemoteCLIP RN50 image encoder.

    Drop-in swap: only init_clip() is overridden.
    The language-fusion layers and all decoder blocks are inherited from CLIPUNet.

    Config keys consumed (under model.remote_clip in clip_unet.yaml):
        model_type          : "RN50" (default)
        checkpoint_path     : path to RemoteCLIP-RN50.pt
        text_templates      : optional list of template strings (default: 5 satellite templates)

    On-the-fly text encoding
    ------------------------
    If batch["target_query"] is a list of strings (e.g. during zero-shot inference
    without a pre-built pkl), the model encodes them on-the-fly using the
    RemoteCLIPTextEncoder with satellite-domain prompt templates.
    If it is a pre-computed float tensor (normal training path), it is used directly.
    """

    def __init__(
        self,
        input_shape: tuple,
        target_input_shape: tuple,
        output_dim: int = 1,
        upsample_factor: int = 2,
        bilinear: bool = True,
        batchnorm: bool = True,
        remote_clip_cfg: Optional[dict] = None,
    ) -> None:
        self._remote_clip_cfg = remote_clip_cfg or {}
        super().__init__(
            input_shape, target_input_shape, output_dim,
            upsample_factor, bilinear, batchnorm,
        )
        self._init_text_encoder()

    def init_clip(self):
        checkpoint_path = self._remote_clip_cfg.get("checkpoint_path", None)
        self.clip = RemoteCLIPRN50Encoder(
            input_shape=self.input_shape,
            backbone_type="prepool",
            checkpoint_path=checkpoint_path,
        )
        self.clip_out_dim = self.clip.output_shape[0]   # 2048
        logger.info("RemoteCLIPUNet: using RemoteCLIP RN50 image encoder.")

    def _init_text_encoder(self):
        """Build the on-the-fly text encoder (used only when target_query is strings)."""
        checkpoint_path = self._remote_clip_cfg.get("checkpoint_path", None)
        templates = self._remote_clip_cfg.get("text_templates", REMOTE_SENSING_TEMPLATES)
        self._text_encoder = RemoteCLIPTextEncoder(
            model_name="RN50",
            checkpoint_path=checkpoint_path,
            templates=templates,
        )
        logger.info(
            f"RemoteCLIPUNet: text encoder ready with {len(templates)} templates — "
            f"{templates[0]!r} ... {templates[-1]!r}"
        )

    def _resolve_text_query(
        self,
        target_query,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Normalise target_query to a float tensor (B, D).

        Accepts:
          - torch.Tensor (float32, B×D)  — pre-computed pkl embedding → used as-is
          - list[str]                    → encoded on-the-fly with RemoteCLIP + templates
        """
        if isinstance(target_query, (list, tuple)) and isinstance(target_query[0], str):
            # On-the-fly encoding path
            emb = self._text_encoder.encode(list(target_query)).to(device)
            return emb
        # Pre-computed tensor path
        return target_query.to(device)

    def forward_encoder(self, image, query):
        # Override to resolve text query before calling parent
        query = self._resolve_text_query(query, image.device)
        return super().forward_encoder(image, query)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        batch = kwargs["batch"]
        batch["target_query"] = self._resolve_text_query(
            batch["target_query"], batch["image"].device
        )
        return super().forward(**kwargs)


@registry.register_affordance_model(name="remote_clip_vit_unet")
class RemoteCLIPViTUNet(SPModel):
    """
    UNet decoder paired with a RemoteCLIP ViT encoder (ViT-B-32 or ViT-L-14).

    Because ViT produces 768- or 1024-channel spatial grids (not 2048), the
    decoder channel widths are adapted.  The text query (target_query) is
    fused via multiplicative FusionMult at three decoder stages, mirroring
    the CLIPUNet design.

    Config keys consumed (under model.remote_clip):
        model_type       : "ViT-B-32" or "ViT-L-14" (default "ViT-B-32")
        checkpoint_path  : path to RemoteCLIP ViT checkpoint
    """

    def __init__(
        self,
        input_shape: tuple,
        target_input_shape: tuple,
        output_dim: int = 1,
        upsample_factor: int = 2,
        bilinear: bool = True,
        batchnorm: bool = True,
        remote_clip_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()

        cfg = remote_clip_cfg or {}
        self.input_shape = input_shape
        self.target_input_shape = target_input_shape
        self.output_dim = output_dim
        self.upsample_factor = upsample_factor
        self.bilinear = bilinear
        self.batchnorm = batchnorm

        model_type = cfg.get("model_type", "ViT-B-32")
        checkpoint_path = cfg.get("checkpoint_path", None)

        # ------------------------------------------------------------------
        # ViT image encoder
        # ------------------------------------------------------------------
        self.vit = RemoteCLIPViTEncoder(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
        )
        D = self.vit.embed_dim          # 768 or 1024
        self.clip_out_dim = D

        logger.info(
            f"RemoteCLIPViTUNet: model={model_type}, embed_dim={D}, "
            f"grid={self.vit.grid_size}×{self.vit.grid_size}"
        )

        # ------------------------------------------------------------------
        # Text query embedding dim (same size as ViT CLS for simplicity)
        # We reuse the ViT CLS token as the scene-level feature; the text
        # query is encoded separately via open_clip.encode_text when called.
        # target_encoder_out_dim mirrors this.
        # ------------------------------------------------------------------
        self.target_encoder_out_dim = D

        # ------------------------------------------------------------------
        # Decoder (channel widths scaled to D instead of 2048)
        # ------------------------------------------------------------------
        half = D // 2       # 384 or 512
        qtr  = D // 4       # 192 or 256
        eth  = D // 8       # 96  or 128

        self.conv1 = nn.Sequential(
            nn.Conv2d(D, D, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.lang_proj1 = nn.Linear(D, D)
        self.lang_proj2 = nn.Linear(D, half)
        self.lang_proj3 = nn.Linear(D, qtr)

        self.lang_fuser1 = FusionMult(input_dim=D)
        self.lang_fuser2 = FusionMult(input_dim=half)
        self.lang_fuser3 = FusionMult(input_dim=qtr)

        self.up1 = Up(D,    half // upsample_factor, bilinear)
        self.up2 = Up(half, qtr  // upsample_factor, bilinear)
        self.up3 = Up(qtr,  eth  // upsample_factor, bilinear)

        self.layer1 = nn.Sequential(
            ConvBlock(eth // 2, [64, 64, 64], kernel_size=3, stride=1, batchnorm=batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=upsample_factor),
        )
        self.layer2 = nn.Sequential(
            ConvBlock(64, [32, 32, 32], kernel_size=3, stride=1, batchnorm=batchnorm),
            IdentityBlock(32, [32, 32, 32], kernel_size=3, stride=1, batchnorm=batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=upsample_factor),
        )
        self.layer3 = nn.Sequential(
            ConvBlock(32, [16, 16, 16], kernel_size=3, stride=1, batchnorm=batchnorm),
            IdentityBlock(16, [16, 16, 16], kernel_size=3, stride=1, batchnorm=batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=upsample_factor),
        )
        self.conv2 = nn.Sequential(nn.Conv2d(16, output_dim, kernel_size=1))
        self.activation = nn.Sigmoid()

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        batch = kwargs["batch"]
        receptacle = batch["image"]
        target = batch["target_query"]      # (B, D) text embedding or category token

        input_shape = receptacle.shape

        # ViT encoder — returns (cls_feat, [4 × spatial grids])
        cls_feat, spatial_feats = self.vit(receptacle)
        # spatial_feats: [layer25%, layer50%, layer75%, layer100%]
        # Use the last two as skip connections (finest features)
        skip_deep   = spatial_feats[-1]     # (B, D, 7, 7) — deepest
        skip_mid    = spatial_feats[-2]     # (B, D, 7, 7)
        skip_shallow = spatial_feats[-3]    # (B, D, 7, 7)

        x = self.conv1(skip_deep)

        x = self.lang_fuser1(x, target, x2_proj=self.lang_proj1)
        x = self.up1(x, skip_mid)

        x = self.lang_fuser2(x, target, x2_proj=self.lang_proj2)
        x = self.up2(x, skip_shallow)

        x = self.lang_fuser3(x, target, x2_proj=self.lang_proj3)
        x = self.up3(x, spatial_feats[-4])

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        x = F.interpolate(x, size=(input_shape[-2], input_shape[-1]), mode="bilinear")

        return {"affordance": x}
