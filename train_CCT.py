import os
from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple
import wandb


import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data.data_module import SemiDataModule
from model.base_module import BaseModule
from model.unet import UNet_CCT
from utils import base_parse_args, mulitmetrics, sigmoid_rampup, wandb_image_mask


class CCTModule(BaseModule):
    """
    this is the implementation of the CCT SSL approach
    - custom unet with aux decoders 1-3
    """

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("LightiningModel")
        parser = super(CCTModule, CCTModule).add_model_specific_args(parser)
        parser.add_argument("--method", default="CCT")
        return parent_parser

    def __init__(self, args: Any):
        super(CCTModule, self).__init__(args)
        self.ce_loss = CrossEntropyLoss()
        self.model = UNet_CCT(in_chns=3, class_num=args.n_class)
        self.consistency = 0.1
        self.test_metrics = mulitmetrics(num_classes=args.n_class)

    def training_step(self, batch: Dict[str, Tuple[Tensor, Tensor, str]]) -> Tensor:
        batch_unusupervised = batch["unlabeled"]
        batch_supervised = batch["labeled"]

        # calculate all outputs based on the different decoders
        image_batch, label_batch, _ = batch_supervised

        outputs, outputs_aux1, outputs_aux2, outputs_aux3 = self.model(image_batch)
        outputs_soft = torch.softmax(outputs, dim=1)
        outputs_aux1_soft = torch.softmax(outputs_aux1, dim=1)
        outputs_aux2_soft = torch.softmax(outputs_aux2, dim=1)
        outputs_aux3_soft = torch.softmax(outputs_aux3, dim=1)
        # calc losses for labeled batch
        label_batch = label_batch.long()
        loss_ce = self.ce_loss(outputs, label_batch)
        loss_ce_aux1 = self.ce_loss(outputs_aux1, label_batch)
        loss_ce_aux2 = self.ce_loss(outputs_aux2, label_batch)
        loss_ce_aux3 = self.ce_loss(outputs_aux3, label_batch)

        supervised_loss = (loss_ce + loss_ce_aux1 + loss_ce_aux2 + loss_ce_aux3) / 4

        # calucalate unsupervised loss
        image_batch, _, _ = batch_unusupervised
        outputs, outputs_aux1, outputs_aux2, outputs_aux3 = self.model(image_batch)
        # warmup unsup loss to avoid inital noise
        consistency_weight = self.consistency * sigmoid_rampup(self.current_epoch)

        consistency_loss_aux1 = torch.mean((outputs_soft - outputs_aux1_soft) ** 2)
        consistency_loss_aux2 = torch.mean((outputs_soft - outputs_aux2_soft) ** 2)
        consistency_loss_aux3 = torch.mean((outputs_soft - outputs_aux3_soft) ** 2)

        consistency_loss = (consistency_loss_aux1 + consistency_loss_aux2 + consistency_loss_aux3) / 3
        loss = supervised_loss + consistency_loss * consistency_weight
        return loss

    def test_step(self, batch, batch_idx):  # type: ignore
        img, mask, id = batch
        pred = self.model(img)[0]
        pred = torch.argmax(pred, dim=1).cpu()
        self.test_metrics.add_batch(pred.numpy(), mask.cpu().numpy())
        return {"Test Pictures": wandb_image_mask(img, mask, pred, self.args.n_class)}

    def configure_optimizers(self) -> List:
        optimizer = SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        return [optimizer]


if __name__ == "__main__":
    args = base_parse_args(CCTModule)
    seed_everything(123, workers=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("./", f"{args.save_path}"),
        filename=f"{args.model}" + "-{epoch:02d}-{val_acc:.2f}",
        mode="max",
        save_weights_only=True,
    )
    if args.use_wandb:
        wandb.init(project="ST++", entity="gkeppler")
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

    # Declare an augmentation pipelines
    a_train_transforms = A.Compose(
        [
            A.LongestMaxSize(args.base_size),
            A.RandomScale(scale_limit=[0, 5, 2], p=1),
            A.RandomCrop(args.crop_size, args.crop_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    dataModule = SemiDataModule(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        split_yaml_path=args.split_file_path,
        test_yaml_path=args.test_file_path,
        pseudo_mask_path=args.pseudo_mask_path,
        train_transforms=a_train_transforms,
        mode="semi_train",
    )

    model = CCTModule(args)
    trainer.fit(model=model, datamodule=dataModule)
    trainer.test(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)
