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
        unlabeled_batch_size: Optional[int] = None,
        color_map: Optional[ndarray] = None,
        mode: Optional[str] = "train",
        num_workers: Optional[int] = 0,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.split_yaml_path = split_yaml_path
        self.test_yaml_path = test_yaml_path
        self.pseudo_mask_path = pseudo_mask_path
        self.unlabeled_batch_size = unlabeled_batch_size if unlabeled_batch_size is not None else batch_size
        self.color_map = color_map
        self.mode = mode
        self.num_workers = num_workers
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
            id_list=self.train_id_dict["labeled"],
            transform=transforms,
            color_map=self.color_map,
        )
        loader_labeled = DataLoader(
            dataset_labeled,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        if self.mode == "train":
            return {"labeled": loader_labeled}
        elif self.mode == "pseudo_train":
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
                batch_size=self.unlabeled_batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            combined_loaders = CombinedLoader(
                {
                    "pseudolabeled": loader_pseudolabeled,
                    "labeled": loader_labeled,
                },
                "max_size_cycle",
            )
            return combined_loaders
        elif self.mode == "semi_train":
            unlabeled_dataset = BaseDataset(
                root_dir=self.root_dir,
                id_list=self.train_id_dict["unlabeled"],
                transform=transforms_unlabeled,
                color_map=self.color_map,
            )

            loader_unlabeled = DataLoader(
                unlabeled_dataset,
                batch_size=self.unlabeled_batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            return {"unlabeled": loader_unlabeled, "labeled": loader_labeled}
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
        transforms = self.base_transform()
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
