import os
from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch import Tensor

import wandb
from SSLightning4Med.models.base_model import BaseModel
from SSLightning4Med.models.data_module import SemiDataModule
from SSLightning4Med.utils.augmentations import Augmentations
from SSLightning4Med.utils.utils import base_parse_args, get_color_map


class STPlusPlusModel(BaseModel):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("LightiningModel")
        parser = super(STPlusPlusModel, STPlusPlusModel).add_model_specific_args(parser)
        parser.add_argument("--method", default="SupervisedOnly", type=str)
        return parent_parser

    def __init__(self, args: Any) -> None:
        super(STPlusPlusModel, self).__init__(args)
        self.args = args

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        return x

    def training_step(self, batch: Dict[str, Tuple[Tensor, Tensor, str]]) -> Tensor:
        img, mask, _ = batch["labeled"]
        pred = self(img)
        if self.args.n_class == 2:
            # BCEloss
            mask = mask.float().squeeze()
            pred = pred.squeeze()
        else:
            # CEloss
            mask = mask.long()
        loss = self.criterion(pred, mask)
        self.log("train_loss", loss, sync_dist=True, on_epoch=True, on_step=True)
        return {"loss": loss}

    def validation_step(self, batch: Tuple[Tensor, Tensor, str], batch_idx: int) -> Dict[str, float]:
        img, mask, id = batch
        pred = self(img)
        self.metric.add_batch(torch.argmax(pred, dim=1).cpu().numpy(), mask.cpu().numpy())
        val_acc = self.metric.evaluate()[-1]
        return {"mIOU": val_acc}

    def validation_epoch_end(self, outputs: List[Dict[str, float]]) -> Dict[str, Union[Dict[str, float], float]]:
        mIOU = outputs[-1]["mIOU"]
        self.log("val_mIoU", mIOU)
        self.set_metrics()


if __name__ == "__main__":

    args = base_parse_args(STPlusPlusModel)
    seed_everything(123, workers=True)
    augs = Augmentations(args)
    color_map = get_color_map(args.dataset)
    dataModule = SemiDataModule(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        split_yaml_path=args.split_file_path,
        test_yaml_path=args.test_file_path,
        pseudo_mask_path=args.pseudo_mask_path,
        test_transforms=augs.a_val_transforms(),
        val_transforms=augs.a_val_transforms(),
        train_transforms=augs.a_train_transforms_labeled(),
        train_transforms_unlabeled=augs.a_train_transforms_unlabeled(),
        mode="train",
        color_map=color_map,
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
        wandb.config.update(args)

    dev_run = False  # not working when predicting with best_model checkpoint
    trainer = pl.Trainer.from_argparse_args(
        args,
        fast_dev_run=dev_run,
        max_epochs=args.epochs,
        log_every_n_steps=2,
        logger=wandb_logger if args.use_wandb else TensorBoardLogger("./tb_logs"),
        callbacks=[checkpoint_callback],
        gpus=[0],
        # accelerator="cpu",
        # profiler="pytorch",
    )
    # <====================== Supervised training with labeled images (SupOnly) ======================>

    trainer.fit(model=model, datamodule=dataModule)

    # <====================== Test supervised model on testset (SupOnly) ======================>
    trainer.test(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)
