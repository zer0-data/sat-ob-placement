from seeing_unseen.models import base, clip_unet

# `llava` submodule requires the LLaVA package and is only used by the
# paper's original VQA pipeline — not by RemoteCLIPUNet training/inference.
# Import `seeing_unseen.models.llava` explicitly if you need it.
