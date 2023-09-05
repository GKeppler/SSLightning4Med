import os
from argparse import ArgumentParser
from typing import List

import cv2
import numpy as np
import pytorch_lightning as pl
from torch import optim
from torchmetrics import JaccardIndex

from SSLightning4Med.nets.deeplabv3plus import DeepLabV3Plus
from SSLightning4Med.nets.seg_former import SegFormer

# from SSLightning4Med.nets.small_unet import SmallUnet2
from SSLightning4Med.nets.unet import UNet, UNet_CCT, small_UNet, small_UNet_CCT
from SSLightning4Med.utils.losses import CELoss, DiceLoss
from SSLightning4Med.utils.utils import (
    get_color_map,
    getOneHot,
    meanIOU,
    mulitmetrics,
    wandb_image_mask,
)

net_zoo = {
    "DeepLabV3Plus": (DeepLabV3Plus, None),
    "Unet": (UNet, UNet_CCT),
    "smallUnet": (small_UNet, small_UNet_CCT),
    "SegFormer": (SegFormer, None),
}


class BaseModule(pl.LightningModule):
    """The base module is used to train the models."""

    def __init__(self, args) -> None:  # type: ignore
        super(BaseModule, self).__init__()
        self.previous_best = 0.0
        self.lr = args.lr
        self.epochs = args.epochs
        self.n_class = args.n_class
        self.use_wandb = args.use_wandb
        self.args = args
        self.oneHot = getOneHot(args.n_class)
        if args.method == "CCT" or args.method == "PseudoCCT" or args.method == "St++CCT":
            self.net = net_zoo[args.net][1]
        else:
            self.net = net_zoo[args.net][0]
        self.net = self.net(in_chns=args.n_channel, n_class=args.n_class if args.n_class > 2 else 1)

        # https://discuss.pytorch.org/t/how-to-use-bce-loss-and-crossentropyloss-correctly/89049
        loss = {"Dice": DiceLoss, "CE": CELoss}[args.loss]
        self.criterion = loss(self.n_class)
        self.val_criterion = loss(self.n_class)
        self.val_IoU = JaccardIndex(
            # ignore_index=0,
            num_classes=self.n_class,
        )
        self.color_map = get_color_map(args.dataset)
        self.set_metrics()

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser) -> ArgumentParser:
        """
        A rule of thumb here is to double the learning rate as you double the batch size.
        """
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--optimizer", type=str, default="Adam", choices=["SGD", "AdamWOneCycle", "Adam"])
        parser.add_argument("--net", type=str, choices=list(net_zoo.keys()), default="Unet")
        parser.add_argument(
            "--method",
            default="Supervised",
            choices=["CCT", "St++", "Bolt", "Supervised", "MeanTeacher", "FixMatch", "PseudoCCT", "St++CCT"],
        )

        # For St++ Model
        parser.add_argument(
            "--plus",
            dest="plus",
            default=True,
            help="whether to use ST++",
        )
        parser.add_argument("--use-tta", default=False, help="whether to use Test Time Augmentation")
        return parser

    def set_metrics(self):
        self.metric = meanIOU(num_classes=self.n_class)
        self.predict_metric = meanIOU(num_classes=self.n_class)
        self.test_metrics = mulitmetrics(num_classes=self.n_class)

    def forward(self, x):  # type: ignore
        return self.net(x)

    def on_train_start(self):
        self.logger.log_hyperparams(self.args),

    def training_step(self, batch, batch_idx):  # type: ignore
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):  # type: ignore
        img, mask, id = batch
        logits = self(img)
        val_loss = self.val_criterion(logits, mask)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)
        pred = self.oneHot(logits)
        self.val_IoU(pred.to(device=self.device), mask)
        self.log("val_mIoU", self.val_IoU, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):  # type: ignore
        img, mask, id = batch
        pred = self.net(img)
        pred = self.oneHot(pred).cpu()
        self.test_metrics.add_batch(pred.numpy(), mask.cpu().numpy())
        caption = f"{self.args.method}, {self.args.dataset}, {self.args.split}, {self.args.shuffle},id:{id[0]}"
        image = wandb_image_mask(img, mask, pred, self.n_class, caption=caption)
        pred = pred.squeeze(0).numpy().astype(np.uint8)
        pred = np.array(self.color_map)[pred]
        cv2.imwrite(
            "%s/%s" % (self.args.test_mask_path, os.path.basename(id[0].split(" ")[1])),
            cv2.cvtColor(pred.astype(np.uint8), cv2.COLOR_BGR2RGB),
        )
        return image

    # for 3D-testing: https://github.com/HiLab-git/SSL4MIS/blob/master/code/test_2D_fully.py
    def test_epoch_end(self, outputs) -> None:  # type: ignore
        overall_acc, meanIOU, meanDSC, medpy_dc, medpy_jc, medpy_hd, medpy_asd = self.test_metrics.evaluate()
        self.log("test overall_acc", overall_acc)
        self.log("test mIOU", meanIOU)
        self.log("test mDICE", meanDSC)
        self.log("test medpy_dc", medpy_dc)
        self.log("test medpy_jc", medpy_jc)
        self.log("test medpy_hd", medpy_hd)
        self.log("test medpy_asd", medpy_asd)

        # save first images
        if self.use_wandb:
            self.logger.experiment.log({"Test Pictures": outputs[:10]})
        # reset metric
        self.test_metrics = mulitmetrics(num_classes=self.n_class)

    def configure_optimizers(self) -> List:
        if self.args.optimizer == "SGD":
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=1e-4,
            )
            return [optimizer]
        elif self.args.optimizer == "Adam":
            """
            loss collapses on hippocampus dataset
            """
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
            return [optimizer]
        elif self.args.optimizer == "AdamWOneCycle":
            """
            Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates
            """
            optimizer = optim.AdamW(self.parameters(), lr=self.lr)
            lr_scheduler = {
                "scheduler": optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.lr,
                    steps_per_epoch=sum(
                        [len(v) for k, v in self.trainer.datamodule.train_dataloader().items()]
                    ),  # len(self.trainer.datamodule.train_dataloader()["labeled"]),
                    epochs=self.epochs,
                    anneal_strategy="linear",
                    final_div_factor=30,
                ),
                "name": "learning_rate",
                "interval": "step",
                "frequency": 1,
            }
            return [optimizer], [lr_scheduler]

    @staticmethod
    def pipeline(get_datamodule, get_trainer, args):
        raise NotImplementedError
