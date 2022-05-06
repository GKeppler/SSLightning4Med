from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
from pl_bolts.models.self_supervised import SwAV
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor

from SSLightning4Med.models.base_module import BaseModule
from SSLightning4Med.models.data_module import SemiDataModule
from SSLightning4Med.nets.deeplabv3plus import DeepLabV3Plus


class BoltModule(BaseModule):
    def __init__(self, args: Any) -> None:
        super(BoltModule, self).__init__(args)
        self.backbone = SwAV.model
        self.net = DeepLabV3Plus(self.backbone, args.n_class)
        self.args = args

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        return x

    def training_step(self, batch: Dict[str, Tuple[Tensor, Tensor, str]]) -> Tensor:
        batch_supervised = batch["labeled"]

        image_batch, mask_batch, _ = batch_supervised
        if self.trainer.current_epoch < self.args.warmup_epochs:
            with torch.no_grad():
                (f1, f2) = self.backbone(image_batch)
                features = f1
        else:
            (f1, f2) = self.backbone(image_batch)
            features = f1
        pred = self.finetune_layer(features)

        loss = self.criterion(pred, mask_batch)
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        return {"loss": loss}

    @staticmethod
    def pipeline(dataModule: SemiDataModule, trainer: pl.Trainer, checkpoint_callback: ModelCheckpoint, args) -> None:
        model = BoltModule(args)
        trainer.fit(model=model, datamodule=dataModule)
        trainer.test(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)
