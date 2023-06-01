import json
import logging
import os
import os.path as osp
import pickle
import random

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from resources.consts import DATASET_ROOT, IMG_NORMALIZE_MEAN, IMG_NORMALIZE_STD
from torch.utils.data import Dataset
from utils.common_utils import NearBBoxResizedSafeCrop
from utils.training_utils import tensor2img


class BaistDataset(Dataset):
    def __init__(self, metadata_root, transforms, partition, target_frames=None, limit=None):
        super(BaistDataset, self).__init__()

        self.sharp_image_paths = []
        self.blur_image_paths = []
        self.blur_anno = []
        self.sequence_ids = []
        self.metadata = []

        self.logger = logging.getLogger("base")

        with open(metadata_root, "r") as f:
            metadata = json.load(f)
            self.dataset_name = metadata["name"]
            self.transforms = transforms

            self.logger.info(f"{partition}: {self.dataset_name}")

            if limit is None:
                limit = len(metadata["data"])

            cnt = {"keep": 0, "reverse": 0, "random": 0, "ignore": 0}

            for data in metadata["data"]:
                if data["partition"] not in ["train", "val", "test"]:
                    raise ValueError(f'Unrecognized partition {data["partition"]}')

                if data["partition"] != partition:
                    continue

                sequence_id = data["id"]
                order = data["order"]

                self.sharp_image_paths.append(
                    [osp.join(DATASET_ROOT, self.dataset_name, data[f"frame{t:03d}_path"]) for t in target_frames]
                )

                self.blur_image_paths.append(osp.join(DATASET_ROOT, self.dataset_name, data["blur_path"]))

                self.blur_anno.append(osp.join(DATASET_ROOT, self.dataset_name, data["bbox"]))

                self.sequence_ids.append(sequence_id)
                self.metadata.append(data)

                if order == "keep":
                    pass
                elif order == "reverse":
                    self.sharp_image_paths[-1].reverse()
                elif order == "ignore":
                    pass
                elif order == "random":
                    if random.randint(0, 1):
                        self.sharp_image_paths[-1].reverse()
                else:
                    raise ValueError(f"Unrecognized order {order}")
                cnt[order] += 1

                if len(self.sequence_ids) == limit:
                    break

            for k, v in cnt.items():
                self.logger.info(f"Number of sequence with order {k}: {v}")

    def __getitem__(self, idx):
        imgs = []
        for frame_path in self.sharp_image_paths[idx]:
            imgs.append(cv2.imread(frame_path))

        blur_img = cv2.imread(self.blur_image_paths[idx])
        anno_file = pickle.load(open(self.blur_anno[idx], "rb"))
        blur_anno = anno_file["bbox"]

        data = {
            f"frame{i + 1}": self.transforms["cropbbox"](image=imgs[i], prev_img=imgs[i - 1], bbox=blur_anno)
            for i in range(len(imgs))
        }
        data["image"] = self.transforms["cropbbox"](image=blur_img, prev_img=blur_img, bbox=blur_anno)

        # print(video_root, type(imgs['frame_7']))
        # if self.transforms['cropbbox'] is not None:
        #     data = self.transforms['cropbbox'](**data)
        if self.transforms["geometry"] is not None:
            data = self.transforms["geometry"](**data)
        if self.transforms["texture"] is not None:
            data = self.transforms["texture"](**data)
        if self.transforms["normalization"] is not None:
            data = self.transforms["normalization"](**data)

        blur_img = data["image"]
        imgs = [data[f"frame{t + 1}"] for t in range(len(imgs))]
        gt = torch.stack(imgs)

        return {
            "blur_img": blur_img,
            "sharp_frames": imgs,
            "mid_img": imgs[len(imgs) // 2],
            "gt": gt,
            "meta": self.metadata[idx],
        }

    def debug(self, save_path):
        for i in range(min(len(self), 2)):
            data = self[i]
            for t, v in enumerate(data["sharp_frames"]):
                dst = osp.join(
                    save_path,
                    f"train_samples/{i:04d}/frame{t}.png",
                )
                os.makedirs(osp.dirname(dst), exist_ok=True)
                cv2.imwrite(dst, tensor2img(v))

            dst = osp.join(
                save_path,
                f"train_samples/{i:04d}/blur.png",
            )
            os.makedirs(osp.dirname(dst), exist_ok=True)
            cv2.imwrite(dst, tensor2img(data["blur_img"]))

    def __len__(self):
        return len(self.blur_image_paths)


class BaistTrainDataset(BaistDataset):
    def __init__(self, metadata_root, target_frames, limit=None):
        if target_frames is None:
            target_frames = range(1, 8)
        additional_targets = {f"frame{t}": "image" for t in target_frames}

        geo_transform = A.Compose(
            [
                A.augmentations.geometric.transforms.Affine(
                    translate_percent={"x": [-0.05, 0.05], "y": [-0.05, 0.05]}, rotate=[-5, 5], shear=[-7, 7], p=0.6
                ),
                # A.augmentations.crops.transforms.RandomCrop(256, 256)
                # A.augmentations.geometric.resize.Resize(256,256)
            ],
            additional_targets=additional_targets,
        )

        texture_transform = A.Compose(
            [
                A.ColorJitter(brightness=0.4, contrast=0.3, p=0.6),
                # A.GaussNoise(p=0.4),
            ],
            additional_targets=additional_targets,
        )

        normalization_transform = A.Compose(
            [
                A.Normalize(mean=IMG_NORMALIZE_MEAN, std=IMG_NORMALIZE_STD),
                ToTensorV2(),
            ],
            additional_targets=additional_targets,
        )
        cropbbox_transform = NearBBoxResizedSafeCrop(height=192, width=192, max_ratio=0.1)

        transforms = {
            "cropbbox": cropbbox_transform,
            "geometry": geo_transform,
            "texture": texture_transform,
            "normalization": normalization_transform,
        }

        super().__init__(
            metadata_root=metadata_root,
            transforms=transforms,
            partition="train",
            target_frames=target_frames,
            limit=limit,
        )


class BaistValDataset(BaistDataset):
    def __init__(self, metadata_root, target_frames, limit=None):
        if target_frames is None:
            target_frames = range(1, 8)
        additional_targets = {f"frame{t}": "image" for t in target_frames}

        geo_transform = None

        texture_transform = None

        normalization_transform = A.Compose(
            [
                A.Normalize(mean=IMG_NORMALIZE_MEAN, std=IMG_NORMALIZE_STD),
                ToTensorV2(),
            ],
            additional_targets=additional_targets,
        )

        cropbbox_transform = NearBBoxResizedSafeCrop(height=192, width=160, max_ratio=0.1)

        transforms = {
            "cropbbox": cropbbox_transform,
            "geometry": geo_transform,
            "texture": texture_transform,
            "normalization": normalization_transform,
        }

        super().__init__(
            metadata_root=metadata_root,
            transforms=transforms,
            partition="val",
            target_frames=target_frames,
            limit=limit,
        )


class BaistTestDataset(BaistDataset):
    def __init__(self, metadata_root, target_frames, limit=None):
        if target_frames is None:
            target_frames = range(1, 8)
        additional_targets = {f"frame{t}": "image" for t in target_frames}

        geo_transform = None

        texture_transform = None

        normalization_transform = A.Compose(
            [
                A.Normalize(mean=IMG_NORMALIZE_MEAN, std=IMG_NORMALIZE_STD),
                ToTensorV2(),
            ],
            additional_targets=additional_targets,
        )

        cropbbox_transform = NearBBoxResizedSafeCrop(height=192, width=160, max_ratio=0.1)

        transforms = {
            "cropbbox": cropbbox_transform,
            "geometry": geo_transform,
            "texture": texture_transform,
            "normalization": normalization_transform,
        }

        super().__init__(
            metadata_root=metadata_root,
            transforms=transforms,
            partition="test",
            target_frames=target_frames,
            limit=limit,
        )
