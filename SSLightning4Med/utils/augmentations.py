""" This file contains the Augmentation pipelines for the training and testing data.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Augmentations:
    def __init__(self, args):
        self.args = args

    def a_train_transforms_labeled(self):
        return A.Compose(
            [
                A.LongestMaxSize(self.args.base_size),
                A.RandomScale(scale_limit=[0, 5, 2], p=1),
                A.RandomCrop(self.args.crop_size, self.args.crop_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

    def a_train_transforms_unlabeled(self):
        return A.Compose(
            [
                A.LongestMaxSize(self.args.base_size),
                A.RandomScale(scale_limit=[0, 5, 2], p=1),
                A.RandomCrop(self.args.crop_size, self.args.crop_size),
                A.HorizontalFlip(p=0.5),
                A.GaussianBlur(p=0.5),
                A.ColorJitter(p=0.8),
                A.CoarseDropout(),  # cutout
                A.Normalize(  # imagenet normalize
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    def a_val_transforms(self):
        return A.Compose([A.LongestMaxSize(self.args.base_size), A.Normalize(), ToTensorV2()])
