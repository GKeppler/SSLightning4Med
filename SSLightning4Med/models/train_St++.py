import os
from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
import yaml
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch import Tensor

from SSLightning4Med.models.base_model import BaseModel
from SSLightning4Med.models.data_module import SemiDataModule
from SSLightning4Med.utils import base_parse_args, meanIOU


class STPlusPlusModel(BaseModel):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("LightiningModel")
        parser = super(STPlusPlusModel, STPlusPlusModel).add_model_specific_args(parser)
        parser.add_argument("--method", default="St++")
        parser.add_argument(
            "--plus",
            dest="plus",
            default=True,
            help="whether to use ST++",
        )
        parser.add_argument("--use-tta", default=True, help="whether to use Test Time Augmentation")
        return parent_parser

    def __init__(self, args: Any) -> None:
        super(STPlusPlusModel, self).__init__(args)
        self.checkpoints: List[torch.nn.Module] = []
        self.id_to_reliability: List[Tuple] = []
        self.previous_best: float = 0.0
        self.args = args
        self.mode = "label"

    def base_forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        x = self.model(x)
        x = F.interpolate(x, (h, w), mode="bilinear", align_corners=True)
        return x

    def forward(self, x: Tensor, tta: bool = False) -> Optional[Tensor]:
        if not tta:
            return self.base_forward(x)

        h, w = x.shape[-2:]
        # scales = [0.5, 0.75, 1.0]
        # to avoid cuda out of memory
        scales = [0.5, 0.75, 1.0, 1.5, 2.0]

        final_result = None

        for scale in scales:
            cur_h, cur_w = int(h * scale), int(w * scale)
            cur_x = F.interpolate(x, size=(cur_h, cur_w), mode="bilinear", align_corners=True)

            out = F.softmax(self.base_forward(cur_x), dim=1)
            out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
            final_result = out if final_result is None else (final_result + out)

            out = F.softmax(self.base_forward(cur_x.flip(3)), dim=1).flip(3)
            out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
            final_result += out

        return final_result

    def training_step(self, batch: Dict[str, Tuple[Tensor, Tensor, str]]) -> Tensor:
        img, mask, _ = batch["labeled"]
        # combine batches
        if "pseudolabeled" in batch:
            img_pseudo, mask_pseudo, _ = batch["pseudolabeled"]
            # torch.unsqueeze(mask, dim=-1).shape
            img = torch.cat((img, img_pseudo), 0)
            mask = torch.cat((mask, mask_pseudo), 0)
        pred = self(img)
        loss = self.criterion(pred, mask.long())
        return loss

    def training_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        if (self.current_epoch + 1) in [
            self.args.epochs // 3,
            self.args.epochs * 2 // 3,
            self.args.epochs,
        ]:
            self.checkpoints.append(deepcopy(self.model))

    def validation_step(self, batch: Tuple[Tensor, Tensor, str], batch_idx: int) -> Dict[str, float]:
        img, mask, id = batch
        pred = self(img)
        self.metric.add_batch(torch.argmax(pred, dim=1).cpu().numpy(), mask.cpu().numpy())
        val_acc = self.metric.evaluate()[-1]
        self.log("mIOU", val_acc)
        return {"mIOU": val_acc}

    def validation_epoch_end(self, outputs: List[Dict[str, float]]) -> Dict[str, Union[Dict[str, float], float]]:
        mIOU = outputs[-1]["mIOU"]
        log = {"mean mIOU": mIOU}
        if mIOU > self.previous_best:
            if self.previous_best != 0:
                os.remove(
                    os.path.join(
                        self.args.save_path,
                        "%s_mIOU%.2f.pth" % (self.args.model, self.previous_best),
                    )
                )
            self.previous_best = mIOU
            torch.save(
                self.state_dict(),
                os.path.join(
                    self.args.save_path,
                    "%s_mIOU%.2f.pth" % (self.args.model, mIOU),
                ),
            )
        return {"log": log, "mIOU": mIOU}

    def predict_step(self, batch: List[Union[Tensor, Tuple[str]]], batch_idx: int) -> None:
        img, mask, id = batch
        if self.mode == "label":
            pred = self(img, tta=True)
            pred = torch.argmax(pred, dim=1).cpu()
            pred = (pred * 255).squeeze(0).numpy().astype(np.uint8)
            cv2.imwrite(
                "%s/%s" % (self.args.pseudo_mask_path, os.path.basename(id[0].split(" ")[1])),
                pred,
            )
        if self.mode == "select_reliable":
            preds = []
            for model in self.checkpoints:
                preds.append(torch.argmax(model(img), dim=1).cpu().numpy())
            mIOU = []
            for i in range(len(preds) - 1):
                metric = meanIOU(self.args.n_class)
                metric.add_batch(preds[i], preds[-1])
                mIOU.append(metric.evaluate()[-1])
            reliability = sum(mIOU) / len(mIOU)
            self.id_to_reliability.append((id[0], reliability))

    def on_predict_epoch_end(self) -> None:
        if self.mode == "select_reliable":
            labeled_ids = []
            with open(self.args.split_file_path, "r") as file:
                split_dict = yaml.load(file, Loader=yaml.FullLoader)
                labeled_ids = split_dict[self.args.val_split]["labeled"]
                val_ids = labeled_ids = split_dict[self.args.val_split]["val"]

            yaml_dict = dict()
            yaml_dict[self.args.val_split] = dict(
                val=val_ids,
                labeled=labeled_ids,
                unlabeled=[i[0] for i in self.id_to_reliability[: len(self.id_to_reliability) // 2]],  # reliable ids
            )
            # save to yaml
            if not os.path.exists(self.args.reliable_id_path):
                os.makedirs(self.args.reliable_id_path)

            with open(os.path.join(self.args.reliable_id_path, "reliable_ids.yaml"), "w+") as outfile:
                yaml.dump(yaml_dict, outfile, default_flow_style=False)


if __name__ == "__main__":

    args = base_parse_args(STPlusPlusModel)

    seed_everything(123, workers=True)

    # Declare an augmentation pipelines
    a_train_transforms_labeled = A.Compose(
        [
            A.LongestMaxSize(args.base_size),
            A.RandomScale(scale_limit=[0, 5, 2], p=1),
            A.RandomCrop(args.crop_size, args.crop_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    a_val_transforms = A.Compose([A.LongestMaxSize(args.base_size), A.Normalize(), ToTensorV2()])

    a_train_transforms_unlabeled = A.Compose(
        [
            A.LongestMaxSize(args.base_size),
            A.RandomScale(scale_limit=[0, 5, 2], p=1),
            A.RandomCrop(args.crop_size, args.crop_size),
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

    dataModule = SemiDataModule(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        split_yaml_path=args.split_file_path,
        test_yaml_path=args.test_file_path,
        pseudo_mask_path=args.pseudo_mask_path,
        test_transforms=a_val_transforms,
        val_transforms=a_val_transforms,
        train_transforms=a_train_transforms_labeled,
        train_transforms_unlabeled=a_train_transforms_unlabeled,
        mode="train",
    )

    model = STPlusPlusModel(args)

    # saves a file like: my/path/sample-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("./", f"{args.save_path}"),
        filename=f"{args.model}" + "-{epoch:02d}-{mIOU:.2f}",
        mode="max",
        save_weights_only=True,
    )
    if args.use_wandb:
        wandb.init(project="SSLightning4Med", entity="gkeppler")
        wandb_logger = WandbLogger(project="SSLightning4Med")
        wandb.define_metric("Pictures")
        wandb.define_metric("loss")
        wandb.define_metric("mIOU")
        wandb.config.update(args)

    dev_run = False  # not working when predicting with best_model checkpoint
    trainer = pl.Trainer.from_argparse_args(
        args,
        fast_dev_run=dev_run,
        max_epochs=args.epochs,
        log_every_n_steps=2,
        logger=wandb_logger if args.use_wandb else TensorBoardLogger("./tb_logs"),
        callbacks=[checkpoint_callback],
        # gpus=[0],
        accelerator="cpu",
        # profiler="pytorch"
    )
    # <====================== Supervised training with labeled images (SupOnly) ======================>

    trainer.fit(model=model, datamodule=dataModule)

    # <====================== Test supervised model on testset (SupOnly) ======================>
    trainer.test(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)

    if not args.plus:
        # <============================= Pseudolabel all unlabeled images =============================>

        trainer.predict(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)

        # <======================== Re-training on labeled and unlabeled images ========================>

        model = STPlusPlusModel(args)

        # increase max epochs to double the amount
        trainer.fit_loop.max_epochs = args.epochs * 2
        dataModule.mode = "pseudo_train"
        trainer.fit(model=model, datamodule=dataModule)
    else:
        # <===================================== Select Reliable IDs =====================================>
        model.mode = "select_reliable"
        trainer.predict(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)
        # <================================ Pseudo label reliable images =================================>
        dataModule.split_yaml_path = os.path.join(args.reliable_id_path, "reliable_ids.yaml")
        dataModule.init_datasets()
        model.mode = "label"
        trainer.predict(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)
        # <================================== The 1st stage re-training ==================================>

        model = STPlusPlusModel(args)

        # increase max epochs to double the amount
        trainer.fit_loop.max_epochs = args.epochs * 3
        dataModule.mode = "pseudo_train"
        trainer.fit(model=model, datamodule=dataModule)
        # <=============================== Pseudo label all images ================================>

        dataModule.split_yaml_path = args.split_file_path
        dataModule.init_datasets()
        model.mode = "label"
        trainer.predict(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)

        # <================================== The 2nd stage re-training ==================================>
        model = STPlusPlusModel(args)

        # increase max epochs to double the amount
        trainer.fit_loop.max_epochs = args.epochs * 4
        trainer.fit(model=model, datamodule=dataModule)

    # <====================== Test supervised model on testset (Semi) ======================>
    trainer.test(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)
