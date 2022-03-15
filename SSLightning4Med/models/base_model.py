from argparse import ArgumentParser
from typing import List

import pytorch_lightning as pl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from SSLightning4Med.nets.unet import SmallUNet, UNet, UNet_CCT
from SSLightning4Med.utils import meanIOU, mulitmetrics, wandb_image_mask

model_zoo = {"unet": UNet, "smallUnet": SmallUNet, "unet_cct": UNet_CCT}


class BaseModel(pl.LightningModule):
    def __init__(self, args) -> None:  # type: ignore
        super(BaseModel, self).__init__()
        self.model = model_zoo[args.model](in_chns=3, class_num=args.n_class)
        self.criterion = CrossEntropyLoss()
        self.previous_best = 0.0
        self.args = args
        self.set_metrics()

    def set_metrics(self):
        self.metric = meanIOU(num_classes=self.args.n_class)
        self.predict_metric = meanIOU(num_classes=self.args.n_class)
        self.test_metrics = mulitmetrics(num_classes=self.args.n_class)

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--model", type=str, choices=list(model_zoo.keys()), default="smallUnet")
        return parser

    def forward(self, x):  # type: ignore
        return self.model(x)

    def training_step(self, batch, batch_idx):  # type: ignore
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):  # type: ignore
        img, mask, id = batch
        pred = self(img)
        self.metric.add_batch(torch.argmax(pred, dim=1).cpu().numpy(), mask.cpu().numpy())
        val_acc = self.metric.evaluate()[-1]
        self.log("mIOU", val_acc)

    def test_step(self, batch, batch_idx):  # type: ignore
        img, mask, id = batch
        pred = self.model(img)
        pred = torch.argmax(pred, dim=1).cpu()
        self.test_metrics.add_batch(pred.numpy(), mask.cpu().numpy())
        return {"Test Picture": wandb_image_mask(img, mask, pred, self.args.n_class)}

    def test_epoch_end(self, outputs) -> None:  # type: ignore
        overall_acc, mIOU, mDICE = self.test_metrics.evaluate()
        self.log("accuracy", overall_acc)
        self.log("mIOU", mIOU)
        self.log("mDICE", mDICE)
        # save first images
        # self.log("Test Pictures", outputs["Test Picture"][:10])
        # reset metric
        self.test_metrics = mulitmetrics(num_classes=self.args.n_class)

    def configure_optimizers(self) -> List:
        optimizer = SGD(
            self.parameters(),
            lr=self.args.lr,
            momentum=0.9,
            weight_decay=1e-4,
        )
        # scheduler = torch.optim.ReduceLROnPlateau(optimizer, mode='min')
        return [optimizer]  # , [scheduler]
