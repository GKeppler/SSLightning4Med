from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor

from SSLightning4Med.models.base_model import BaseModule, net_zoo
from SSLightning4Med.models.data_module import SemiDataModule
from SSLightning4Med.models.train_CCT import consistency_loss
from SSLightning4Med.utils.utils import sigmoid_rampup


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


class MeanTeacherModule(BaseModule):
    def __init__(self, args: Any) -> None:
        super(MeanTeacherModule, self).__init__(args)
        self.consistency = 0.1
        self.net_ema = net_zoo[args.net][0]
        self.net_ema = self.net_ema(in_chns=args.n_channel, n_class=args.n_class if args.n_class > 2 else 1)
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
        noise = torch.clamp(torch.randn_like(unlabeled_image_batch) * 0.1, -0.2, 0.2)
        ema_inputs = unlabeled_image_batch + noise
        with torch.no_grad():
            ema_output = self.net_ema(ema_inputs)
            # ema_output_soft = torch.softmax(ema_output, dim=1)

        outputs_unsup = self.net(unlabeled_image_batch)
        # outputs_unsup_soft = torch.softmax(outputs_unsup, dim=1)

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
        update_ema_variables(self.net, self.net_ema, 0.99, self.global_step)

    @staticmethod
    def pipeline(dataModule: SemiDataModule, trainer: pl.Trainer, checkpoint_callback: ModelCheckpoint, args) -> None:
        model = MeanTeacherModule(args)
        dataModule.mode = "semi_train"
        trainer.fit(model=model, datamodule=dataModule)
        trainer.test(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)
