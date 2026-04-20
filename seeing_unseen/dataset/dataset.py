import os
import random

# Ignore warnings
import warnings
from collections import defaultdict
from typing import Any

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Canonical image size for collation. The transform pipeline may further
# rescale/crop this during training augmentations, but every __getitem__
# must return the same shape so torch.stack() works across a batch when
# the underlying dataset has heterogeneous native resolutions (e.g. MVRSD).
# This matches RemoteCLIPUNet's input_shape=(3, 480, 640) — see
# scripts/infer_placement.py and config/baseline/clip_unet_mvrsd.yaml.
_COLLATE_H, _COLLATE_W = 480, 640

from seeing_unseen.core.logger import logger
from seeing_unseen.core.registry import registry
from seeing_unseen.utils.utils import (
    decode_rle_mask,
    load_image,
    load_json,
    load_pickle,
)

warnings.filterwarnings("ignore")


def collate_fn(batch):
    observations = defaultdict(list)

    for sample in batch:
        for key, val in sample.items():
            observations[key].append(val)

    observations_batch = {}
    for key, val in observations.items():
        if "target_category" in key:
            observations_batch[key] = val
            continue
        observations_batch[key] = torch.stack(val)

        # NHWC -> NCHW for augmentations
        if len(observations_batch[key].shape) == 4:
            observations_batch[key] = observations_batch[key].permute(
                0, 3, 1, 2
            )
    return observations_batch


@registry.register_dataset(name="semantic_placement")
class SemanticPlacementTextQueryDataset(Dataset):
    def __init__(
        self,
        split: str,
        root_dir: str,
        trfms: str = "none",
        load_original_image: bool = False,
        load_depth: bool = False,
        embeddings_file: str = "clip_embeddings.pkl",
    ) -> None:
        super().__init__()

        self.split = split
        self.root_dir = f"{root_dir}/{split}"

        metadata_file = f"{self.root_dir}/{split}_records.json"

        if os.path.exists(metadata_file):
            self.metadata = load_json(metadata_file)
        else:
            self.metadata = []

        clip_embeddings_file = f"{self.root_dir}/{embeddings_file}"
        self.clip_embeddings = load_pickle(clip_embeddings_file)

        if load_original_image:
            self.original_records = load_json(
                f"{root_dir}/original_records.json"
            )

        self.transforms = registry.get_transforms(trfms)()
        self.load_original_image = load_original_image
        self.load_depth = load_depth

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index) -> Any:
        if torch.is_tensor(index):
            index = index.tolist()

        record = self.metadata[index]

        # Load + resize image to a fixed collation size so that
        # torch.stack() succeeds across a batch even when the source
        # dataset (e.g. MVRSD) has variable native resolutions.
        pil_img = load_image(record["img_path"])
        if pil_img.size != (_COLLATE_W, _COLLATE_H):
            pil_img = pil_img.resize(
                (_COLLATE_W, _COLLATE_H), resample=Image.BILINEAR
            )
        image = np.array(pil_img)
        native_w, native_h = pil_img.size  # after resize: (_COLLATE_W, _COLLATE_H)

        annotation = random.choice(record["annotations"])
        target_mask = np.array(decode_rle_mask(annotation["segmentation"]))
        target_mask = np.where(target_mask > 0, 1, 0).astype(np.uint8)
        # Resize mask with nearest-neighbour to the same canonical size
        if target_mask.shape != (_COLLATE_H, _COLLATE_W):
            target_mask = np.array(
                Image.fromarray(target_mask).resize(
                    (_COLLATE_W, _COLLATE_H), resample=Image.NEAREST
                )
            )

        target_category = (
            annotation["object_category"].split("|")[0].replace("_", " ")
        )
        target_query = torch.tensor(self.clip_embeddings[target_category])

        sample = {
            "image": torch.tensor(image),
            "target_query": target_query,
            "mask": torch.tensor(target_mask),
            "target_category": target_category,
            "original_image": torch.tensor(image),
        }
        if self.load_original_image and self.split == "train":
            original_image_record = random.choice(
                self.original_records[self.split]
            )
            original_image = np.array(load_image(original_image_record))

            sample["original_image"] = torch.tensor(original_image)

        if record.get("depth_img_path") is not None:
            depth = np.load("{}.npy".format(record["depth_img_path"]))
            sample["depth"] = torch.tensor(depth)

        if annotation.get("receptacle_mask") is not None:
            receptacle_masks = []
            receptacle_id = 0
            ann_receptacle_masks = annotation.get("receptacle_mask")
            if isinstance(ann_receptacle_masks, list):
                for idx, rle_receptacle_mask in enumerate(ann_receptacle_masks):
                    receptacle_masks.append(
                        (np.array(decode_rle_mask(rle_receptacle_mask)) / 255)
                        * (receptacle_id + 1)
                    )
                    receptacle_id += 1
            else:
                receptacle_masks.append(
                    np.array(decode_rle_mask(annotation["receptacle_mask"]))
                )
                receptacle_id = 1
            receptacle_mask = np.add.reduce(receptacle_masks)
            if receptacle_id == 1:
                receptacle_mask = np.where(receptacle_mask > 0, 1, 0)
            sample["receptacle_mask"] = torch.tensor(receptacle_mask)
        return sample


@registry.register_dataset(name="semantic_multi_placement_text_query")
class SemanticMultiPlacementTextQueryDataset(Dataset):
    def __init__(
        self,
        split: str,
        root_dir: str,
        trfms: str = "none",
        load_original_image: bool = False,
        embeddings_file: str = "clip_embeddings.pkl",
    ) -> None:
        super().__init__()

        self.split = split
        self.root_dir = f"{root_dir}/{split}"

        metadata_file = f"{self.root_dir}/{split}_records.json"
        self.metadata = load_json(metadata_file)

        clip_embeddings_file = f"{self.root_dir}/{embeddings_file}"
        self.clip_embeddings = load_pickle(clip_embeddings_file)

        if load_original_image:
            self.original_records = load_json(
                f"{root_dir}/original_records.json"
            )

        self.transforms = registry.get_transforms(trfms)()
        self.add_original_image = load_original_image
        self.total_samples = len(self.metadata)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index) -> Any:
        record = self.metadata[index]

        image = np.array(load_image(record["img_path"]))

        num_masks_per_category = defaultdict(int)
        for annotation in record["annotations"]:
            num_masks_per_category[
                annotation["object_category"].split("|")[0]
            ] += 1

        categories = list(num_masks_per_category.keys())
        target_category = random.choice(categories)
        target_query = torch.tensor(self.clip_embeddings[target_category])

        target_masks = []
        receptacle_masks = []
        receptacle_id = 0
        for annotation in record["annotations"]:
            if annotation["object_category"].split("|")[0] == target_category:
                target_masks.append(
                    np.array(decode_rle_mask(annotation["segmentation"]))
                )
                if annotation.get("receptacle_mask") is not None:
                    ann_receptacle_masks = annotation.get("receptacle_mask")
                    if isinstance(ann_receptacle_masks, list):
                        for idx, rle_receptacle_mask in enumerate(
                            ann_receptacle_masks
                        ):
                            receptacle_masks.append(
                                (
                                    np.array(
                                        decode_rle_mask(rle_receptacle_mask)
                                    )
                                    / 255
                                )
                                * (receptacle_id + 1)
                            )
                            receptacle_id += 1
                    else:
                        receptacle_masks.append(
                            np.array(
                                decode_rle_mask(annotation["receptacle_mask"])
                            )
                        )
                        receptacle_id = 1

        num_masks = random.choice(list(range(1, len(target_masks) + 1)))
        target_mask = np.add.reduce(random.sample(target_masks, num_masks))
        target_mask = np.where(target_mask > 0, 1, 0)

        if len(receptacle_masks) > 0:
            receptacle_mask = np.add.reduce(receptacle_masks)
            if receptacle_id == 1:
                receptacle_mask = np.where(receptacle_mask > 0, 1, 0)
        else:
            receptacle_mask = None

        sample = {
            "image": torch.tensor(image),
            "target_query": target_query,
            "mask": torch.tensor(target_mask),
            "target_category": target_category,
            "original": torch.tensor(image),
        }

        if receptacle_mask is not None:
            sample["receptacle_mask"] = torch.tensor(receptacle_mask)

        if record.get("depth_img_path") is not None:
            depth = np.load("{}.npy".format(record["depth_img_path"]))
            sample["depth"] = torch.tensor(depth)

        return sample
