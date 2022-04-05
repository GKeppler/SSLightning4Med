from typing import Optional

import pytorch_lightning as pl
import yaml
from albumentations.core.composition import Compose
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
        train_transforms: Compose,
        train_transforms_unlabeled: Optional[Compose] = None,
        test_transforms: Optional[Compose] = None,
        val_transforms: Optional[Compose] = None,
        unlabeled_batch_size: Optional[int] = None,
        color_map: Optional[ndarray] = None,
        mode: str = "train",
        num_workers: Optional[int] = 4,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.split_yaml_path = split_yaml_path
        self.test_yaml_path = test_yaml_path
        self.pseudo_mask_path = pseudo_mask_path
        self.train_transforms = train_transforms
        self.train_transforms_unlabeled = (
            train_transforms_unlabeled if train_transforms_unlabeled is not None else train_transforms
        )
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.unlabeled_batch_size = unlabeled_batch_size if unlabeled_batch_size is not None else batch_size
        self.color_map = color_map
        self.mode = mode
        self.num_workers = num_workers
        self.init_datasets()

    def init_datasets(self) -> None:
        with open(self.split_yaml_path, "r") as file:
            split_dict = yaml.load(file, Loader=yaml.FullLoader)
        val_split_0 = split_dict["val_split_0"]
        # supervised
        self.labeled_dataset = BaseDataset(
            root_dir=self.root_dir,
            id_list=val_split_0["labeled"],
            transform=self.train_transforms,
            color_map=self.color_map,
        )
        self.pseudolabeled_dataset = BaseDataset(
            root_dir=self.root_dir,
            id_list=val_split_0["unlabeled"],
            transform=self.train_transforms_unlabeled,
            pseudo_mask_path=self.pseudo_mask_path,
            color_map=self.color_map,
        )
        # unsupervised
        self.unlabeled_dataset = BaseDataset(
            root_dir=self.root_dir,
            id_list=val_split_0["unlabeled"],
            transform=self.train_transforms_unlabeled,
            color_map=self.color_map,
        )
        self.predict_dataset = BaseDataset(
            root_dir=self.root_dir,
            id_list=val_split_0["unlabeled"],
            transform=None,
            color_map=self.color_map,
        )

        self.val_dataset = BaseDataset(
            root_dir=self.root_dir, transform=self.val_transforms, id_list=val_split_0["val"], color_map=self.color_map
        )
        # testset
        with open(self.test_yaml_path, "r") as file:
            test_list = yaml.load(file, Loader=yaml.FullLoader)

        self.test_dataset = BaseDataset(
            root_dir=self.root_dir, transform=self.test_transforms, id_list=test_list, color_map=self.color_map
        )

    def train_dataloader(self):  # type: ignore
        loader_labeled = DataLoader(self.labeled_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        if self.mode == "train":
            return {"labeled": loader_labeled}
        elif self.mode == "pseudo_train":
            combined_loaders = CombinedLoader(
                {
                    "pseudolabeled": DataLoader(
                        self.pseudolabeled_dataset, batch_size=self.unlabeled_batch_size, num_workers=self.num_workers
                    ),
                    "labeled": loader_labeled,
                },
                "max_size_cycle",
            )
            return combined_loaders
        elif self.mode == "semi_train":
            loader_unlabeled = DataLoader(
                self.unlabeled_dataset,
                batch_size=self.unlabeled_batch_size,
                num_workers=self.num_workers,
            )
            return {"unlabeled": loader_unlabeled, "labeled": loader_labeled}
        else:
            raise ValueError("Wrong dataloader mode for training")

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            num_workers=self.num_workers,
        )
