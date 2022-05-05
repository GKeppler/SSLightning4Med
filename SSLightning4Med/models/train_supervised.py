from typing import Any, Dict, Tuple

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor

from SSLightning4Med.models.base_model import BaseModule
from SSLightning4Med.models.data_module import SemiDataModule


class SupervisedModule(BaseModule):
    def __init__(self, args: Any) -> None:
        super(SupervisedModule, self).__init__(args)
        self.args = args

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        return x

    def training_step(self, batch: Dict[str, Tuple[Tensor, Tensor, str]]) -> Tensor:
        img, mask, _ = batch["labeled"]
        pred = self(img)
        loss = self.criterion(pred, mask)
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        return {"loss": loss}

    @staticmethod
    def pipeline(dataModule: SemiDataModule, trainer: pl.Trainer, checkpoint_callback: ModelCheckpoint, args) -> None:
        model = SupervisedModule(args)
        trainer.fit(model=model, datamodule=dataModule)
        trainer.test(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)
