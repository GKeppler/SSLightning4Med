from copy import deepcopy
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor

from SSLightning4Med.models.base_model import BaseModule
from SSLightning4Med.models.data_module import SemiDataModule
from SSLightning4Med.models.train_CCT import consistency_loss
from SSLightning4Med.utils.utils import sigmoid_rampup


def update_ema_variables(model, ema_model, alpha, global_step):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        if param.requires_grad:
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


class MeanTeacherModule(BaseModule):
    def __init__(self, args: Any) -> None:
        super(MeanTeacherModule, self).__init__(args)
        self.consistency = 0.1
        self.ema = EMA(self.net, 0.99)
        self.ema.register()
        # old
        self.net_ema = deepcopy(self.net)
        for param in self.net_ema.parameters():
            param.detach_()

    def training_step(self, batch: Dict[str, Tuple[Tensor, Tensor, str]]) -> Tensor:
        # supervised
        batch_supervised = batch["labeled"]
        labeled_image_batch, label_batch, _ = batch_supervised
        outputs = self.net(labeled_image_batch)
        supervised_loss = self.criterion(outputs, label_batch)
        # unsupervised
        batch_unusupervised = batch["unlabeled"]
        unlabeled_image_batch, _, _ = batch_unusupervised

        self.ema.apply_shadow()
        with torch.no_grad():
            ema_output = self.net(unlabeled_image_batch)
        self.ema.restore()
        outputs_unsup = self.net(unlabeled_image_batch)
        consistency_weight = self.consistency * sigmoid_rampup(self.current_epoch)
        if self.global_step < 1000:
            unsupervised_loss = 0.0
        else:
            unsupervised_loss = consistency_loss(outputs_unsup, ema_output)

        loss = supervised_loss + consistency_weight * unsupervised_loss
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        return {"loss": loss}

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update()
        # update_ema_variables(self.net, self.net_ema, 0.99, self.global_step)

    @staticmethod
    def pipeline(dataModule: SemiDataModule, trainer: pl.Trainer, checkpoint_callback: ModelCheckpoint, args) -> None:
        model = MeanTeacherModule(args)
        dataModule.mode = "semi_train"
        trainer.fit(model=model, datamodule=dataModule)
        trainer.test(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)


class EMA:
    """
    Implementation from https://fyubang.com/2019/06/01/ema/
    """

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def load(self, ema_model):
        for name, param in ema_model.named_parameters():
            self.shadow[name] = param.data.clone()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
