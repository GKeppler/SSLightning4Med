import math
from typing import Optional

import albumentations as A
import pytorch_lightning as pl
import yaml
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2
from numpy import ndarray
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader

from SSLightning4Med.models.dataset import BaseDataset


class SemiDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        batch_size: int,
        split_yaml_path: str,
        test_yaml_path: str,
        pseudo_mask_path: str,
        batch_size_unlabeled: Optional[int] = None,
        color_map: Optional[ndarray] = None,
        mode: Optional[str] = "train",
        num_workers: Optional[int] = 0,
        oversample: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.split_yaml_path = split_yaml_path
        self.test_yaml_path = test_yaml_path
        self.pseudo_mask_path = pseudo_mask_path
        self.batch_size_unlabeled = batch_size if batch_size_unlabeled is None else batch_size_unlabeled
        self.color_map = color_map
        self.mode = mode
        self.num_workers = num_workers
        self.oversample = oversample
        self.setup_split()

    def setup_split(self):
        with open(self.split_yaml_path, "r") as file:
            split_dict = yaml.load(file, Loader=yaml.FullLoader)
            self.train_id_dict = split_dict["val_split_0"]
        # testset
        with open(self.test_yaml_path, "r") as file:
            self.test_id_list = yaml.load(file, Loader=yaml.FullLoader)

    def train_dataloader(self):  # type: ignore
        transforms = self.base_transform() if self.train_transforms is None else self.train_transforms
        transforms_unlabeled = (
            self.base_transform() if self.train_transforms_unlabeled is None else self.train_transforms_unlabeled
        )
        # supervised
        dataset_labeled = BaseDataset(
            root_dir=self.root_dir,
            id_list=(
                self.train_id_dict["labeled"]
                * math.ceil(len(self.train_id_dict["unlabeled"]) / len(self.train_id_dict["labeled"]))
                if self.oversample is True
                else self.train_id_dict["labeled"]
            ),
            transform=transforms,
            color_map=self.color_map,
        )
        if self.mode == "train":
            loader_labeled = DataLoader(
                dataset_labeled,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=True,
            )
            return {"labeled": loader_labeled}
        elif self.mode == "pseudo_train":
            loader_labeled = DataLoader(
                dataset_labeled,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=True,
            )
            # unsupervised
            pseudolabeled_dataset = BaseDataset(
                root_dir=self.root_dir,
                id_list=self.train_id_dict["unlabeled"],
                transform=transforms_unlabeled,
                pseudo_mask_path=self.pseudo_mask_path,
                color_map=self.color_map,
            )
            loader_pseudolabeled = DataLoader(
                pseudolabeled_dataset,
                batch_size=self.batch_size_unlabeled,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=True,
            )
            combined_loaders = CombinedLoader(
                {
                    "labeled": loader_labeled,
                    "pseudolabeled": loader_pseudolabeled,
                },
                "max_size_cycle",
            )
            return combined_loaders
        elif self.mode == "semi_train":
            loader_labeled = DataLoader(
                dataset_labeled,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=True,
            )
            # unsupervised
            unlabeled_dataset = BaseDataset(
                root_dir=self.root_dir,
                id_list=self.train_id_dict["unlabeled"],
                transform=transforms_unlabeled,
                color_map=self.color_map,
            )
            loader_unlabeled = DataLoader(
                unlabeled_dataset,
                batch_size=self.batch_size_unlabeled,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            combined_loaders = CombinedLoader(
                {
                    "labeled": loader_labeled,
                    "unlabeled": loader_unlabeled,
                },
                "max_size_cycle",
            )
            return combined_loaders
        elif self.mode == "fixmatch_train":
            loader_labeled = DataLoader(
                dataset_labeled,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=True,
            )
            # unsupervised
            unlabeled_dataset_wa = BaseDataset(
                root_dir=self.root_dir,
                id_list=self.train_id_dict["unlabeled"],
                transform=transforms,
                color_map=self.color_map,
            )
            loader_unlabeled_wa = DataLoader(
                unlabeled_dataset_wa,
                batch_size=self.batch_size_unlabeled,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=False,
            )
            unlabeled_dataset_sa = BaseDataset(
                root_dir=self.root_dir,
                id_list=self.train_id_dict["unlabeled"],
                transform=transforms_unlabeled,
                color_map=self.color_map,
            )
            loader_unlabeled_sa = DataLoader(
                unlabeled_dataset_sa,
                batch_size=self.batch_size_unlabeled,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=False,
            )
            combined_loaders = CombinedLoader(
                {
                    "labeled": loader_labeled,
                    "unlabeled_wa": loader_unlabeled_wa,
                    "unlabeled_sa": loader_unlabeled_sa,
                },
                "max_size_cycle",
            )
            return combined_loaders
        else:
            raise ValueError("Wrong dataloader mode for training")

    def val_dataloader(self) -> DataLoader:
        transforms = self.base_transform() if self.val_transforms is None else self.val_transforms

        val_dataset = BaseDataset(
            root_dir=self.root_dir, transform=transforms, id_list=self.train_id_dict["val"], color_map=self.color_map
        )
        return DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        transforms = self.base_transform() if self.val_transforms is None else self.val_transforms
        test_dataset = BaseDataset(
            root_dir=self.root_dir, transform=transforms, id_list=self.test_id_list, color_map=self.color_map
        )
        return DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self) -> DataLoader:
        transforms = self.base_transform() if self.val_transforms is None else self.val_transforms
        predict_dataset = BaseDataset(
            root_dir=self.root_dir,
            id_list=self.train_id_dict["unlabeled"],
            transform=transforms,
            color_map=self.color_map,
        )
        return DataLoader(
            predict_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def base_transform(self) -> Compose:
        """Standard transform.
        Just the ImageNet Normalization.
        """
        return A.Compose(
            [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
