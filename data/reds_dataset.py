from torch.utils.data import Dataset
from resources.const import REDS_DATASET_ROOT_DIR, IMG_NORMALIZE_MEAN, IMG_NORMALIZE_STD
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os.path as osp


class REDSTrainDataset(Dataset):
    def __init__(self, target_frames, use_flow, limit, **kwargs):
        if target_frames is None:
            target_frames = range(1, 8)
        additional_targets = {f'frame_{t}': 'image' for t in target_frames}

        geo_transform = A.compose(
            [
                A.augmentations.geometric.transforms.Affine(
                    translate_percent={'x': [-0.05, 0.05], 'y': [-0.05, 0.05]},
                    rotate=[-5, 5],
                    shear=[-7, 7],
                    p=0.6
                ),
                A.augmentations.crops.transforms.RandomCrop(256, 256)
            ],
            additional_targets=additional_targets
        )

        texture_transform = A.Compose(
            [
                A.ColorJitter(brightness=0.4, contrast=0.3, p=0.6),
                A.GaussNoise(p=0.4),
            ],
            additional_targets=additional_targets
        )

        normalization_transform = A.Compose(
            [
                A.Normalize(mean=IMG_NORMALIZE_MEAN, std=IMG_NORMALIZE_STD),
                ToTensorV2(),
            ],
            additional_targets=additional_targets
        )

        transforms = {
            'geometry': geo_transform,
            'texture': texture_transform,
            'normalization': normalization_transform
        }

        super().__init__(
            root_dir=osp.join(REDS_DATASET_ROOT_DIR, 'train'),
            transforms=transforms,
            target_frames=target_frames,
            limit=kwargs['train_limit'],
        )


class REDSTrainDataset(Dataset):
    def __init__(self, target_frames, use_flow, limit, **kwargs):
        if target_frames is None:
            target_frames = range(1, 8)
        additional_targets = {f'frame_{t}': 'image' for t in target_frames}

        geo_transform = A.compose(
            [
                A.augmentations.geometric.transforms.Affine(
                    translate_percent={'x': [-0.05, 0.05], 'y': [-0.05, 0.05]},
                    rotate=[-5, 5],
                    shear=[-7, 7],
                    p=0.6
                ),
                A.augmentations.crops.transforms.RandomCrop(256, 256)
            ],
            additional_targets=additional_targets
        )

        texture_transform = A.Compose(
            [
                A.ColorJitter(brightness=0.4, contrast=0.3, p=0.6),
                A.GaussNoise(p=0.4),
            ],
            additional_targets=additional_targets
        )

        normalization_transform = A.Compose(
            [
                A.Normalize(mean=IMG_NORMALIZE_MEAN, std=IMG_NORMALIZE_STD),
                ToTensorV2(),
            ],
            additional_targets=additional_targets
        )

        transforms = {
            'geometry': geo_transform,
            'texture': texture_transform,
            'normalization': normalization_transform
        }

        super().__init__(
            root_dir=osp.join(REDS_DATASET_ROOT_DIR, 'train'),
            transforms=transforms,
            target_frames=target_frames,
            limit=kwargs['train_limit'],
        )
