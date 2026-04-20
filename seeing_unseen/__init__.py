from seeing_unseen import core, dataset, models, terrain, trainer, utils

# `third_party/` (Detic, Inpaint-Anything, LLaVA) is intentionally not eagerly
# imported — it pulls heavyweight dependencies that aren't needed for the
# RemoteCLIP affordance training / inference path. Import explicitly when used.
