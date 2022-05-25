""" This file contains the augmentation pipelines for the labeled/unlabeled training and testing data.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Augmentations:
    def __init__(self, args):
        self.args = args
        self.mean, self.std = {
            "melanoma": ([0.7116, 0.5834, 0.5337], [0.1471, 0.1646, 0.1795]),
            "pneumothorax": (0.5380, 0.2641),
            "breastCancer": (0.3298, 0.2218),
            "multiorgan": (0.1935, 0.1889),
            "brats": (0.0775, 0.1539),
            "hippocampus": (0.2758, 0.1628),
        }[args.dataset]

    def a_train_transforms_weak(self):
        return A.Compose(
            [
                # A.RandomScale(scale_limit=(0.5, 2), p=1),
                A.PadIfNeeded(self.args.crop_size, self.args.crop_size),
                A.RandomCrop(self.args.crop_size, self.args.crop_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(self.mean, self.std),
                ToTensorV2(),
            ]
        )

    def a_train_transforms_strong(self):
        return A.Compose(
            [
                # A.RandomScale(scale_limit=(0.5, 2), p=1),
                A.PadIfNeeded(self.args.crop_size, self.args.crop_size),
                A.RandomCrop(self.args.crop_size, self.args.crop_size),
                A.HorizontalFlip(p=0.5),
                A.GaussianBlur(p=0.5),
                A.ColorJitter(p=0.8),
                A.CoarseDropout(),  # cutout
                A.Normalize(self.mean, self.std),
                ToTensorV2(),
            ]
        )

    def a_val_transforms(self):
        return A.Compose(
            [
                A.PadIfNeeded(self.args.crop_size, self.args.crop_size),
                A.Resize(self.args.crop_size, self.args.crop_size),
                A.Normalize(self.mean, self.std),
                ToTensorV2(),
            ]
        )
