from argparse import ArgumentParser
from typing import List

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import SGD
from torchmetrics import IoU

from SSLightning4Med.models.data_module import SemiDataModule

# from SSLightning4Med.nets.small_unet import SmallUnet2
from SSLightning4Med.nets.unet import UNet, UNet_CCT, small_UNet, small_UNet_CCT
from SSLightning4Med.utils.losses import CE_loss
from SSLightning4Med.utils.utils import meanIOU, mulitmetrics, wandb_image_mask

net_zoo = {"unet": (UNet, UNet_CCT), "smallUnet": (small_UNet, small_UNet_CCT)}


class BaseModule(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--net", type=str, choices=list(net_zoo.keys()), default="unet")
        parser.add_argument("--method", default="Supervised", choices=["CCT", "St++", "Bolt", "Supervised"])

        # For St++ Model
        parser.add_argument(
            "--plus",
            dest="plus",
            default=True,
            help="whether to use ST++",
        )
        parser.add_argument("--use-tta", default=True, help="whether to use Test Time Augmentation")
        return parser

    def __init__(self, args) -> None:  # type: ignore
        super(BaseModule, self).__init__()
        self.previous_best = 0.0
        self.lr = args.lr
        self.n_class = args.n_class
        self.use_wandb = args.use_wandb
        if args.method == "CCT":
            self.net = net_zoo[args.net][1]
        else:
            self.net = net_zoo[args.net][0]
        self.net = self.net(in_chns=3, class_num=args.n_class if args.n_class > 2 else 1)

        # https://discuss.pytorch.org/t/how-to-use-bce-loss-and-crossentropyloss-correctly/89049
        self.criterion = CE_loss(self.n_class)
        self.val_IoU = IoU(
            # ignore_index=0,
            num_classes=self.n_class,
        )
        self.set_metrics()

    def set_metrics(self):
        self.metric = meanIOU(num_classes=self.n_class)
        self.predict_metric = meanIOU(num_classes=self.n_class)
        self.test_metrics = mulitmetrics(num_classes=self.n_class)

    def forward(self, x):  # type: ignore
        return self.net(x)

    def training_step(self, batch, batch_idx):  # type: ignore
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):  # type: ignore
        img, mask, id = batch
        pred = self(img)
        self.val_IoU(torch.argmax(pred, dim=1), mask)
        self.log("val_mIoU", self.val_IoU, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):  # type: ignore
        img, mask, id = batch
        pred = self.net(img)
        pred = torch.argmax(pred, dim=1).cpu()
        self.test_metrics.add_batch(pred.numpy(), mask.cpu().numpy())
        return wandb_image_mask(img, mask, pred, self.n_class)

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
        optimizer = SGD(
            self.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=1e-4,
        )
        # scheduler = torch.optim.ReduceLROnPlateau(optimizer, mode='min')
        return [optimizer]  # , [scheduler]

    @staticmethod
    def pipeline(dataModule: SemiDataModule, trainer: pl.Trainer, checkpoint_callback: ModelCheckpoint, args) -> None:
        raise NotImplementedError
