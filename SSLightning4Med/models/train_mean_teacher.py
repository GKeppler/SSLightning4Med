from copy import deepcopy
from typing import Any, Dict, Tuple

import torch
from torch import Tensor, nn

from SSLightning4Med.models.base_module import BaseModule
from SSLightning4Med.models.train_CCT import consistency_loss
from SSLightning4Med.utils.utils import consistency_weight


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


class Bn_Controller:
    """update BatchNorm only for labeled data if
    labeled data and unlabeled data are forwarded separatel.
    Record the BatchNorm statistics before the forward propagation of unlabeled data and
    restore them after the propagation is done.
    """

    def __init__(self):
        """
        freeze_bn and unfreeze_bn must appear in pairs
        """
        self.backup = {}

    def freeze_bn(self, model):
        assert self.backup == {}
        for name, m in model.named_modules():
            if isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                self.backup[name + ".running_mean"] = m.running_mean.data.clone()
                self.backup[name + ".running_var"] = m.running_var.data.clone()
                self.backup[name + ".num_batches_tracked"] = m.num_batches_tracked.data.clone()

    def unfreeze_bn(self, model):
        for name, m in model.named_modules():
            if isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                m.running_mean.data = self.backup[name + ".running_mean"]
                m.running_var.data = self.backup[name + ".running_var"]
                m.num_batches_tracked.data = self.backup[name + ".num_batches_tracked"]
        self.backup = {}


class MeanTeacherModule(BaseModule):
    def __init__(self, args: Any) -> None:
        super(MeanTeacherModule, self).__init__(args)
        self.bn_controller = Bn_Controller()
        self.net_ema = deepcopy(self.net)
        for param in self.net_ema.parameters():
            param.detach_()
        self.ramp_up = 0.1  # from CCT paper
        self.cons_w_unsup = consistency_weight(final_w=30, rampup_ends=self.ramp_up * self.epochs)  # from CCT paper

    def training_step(self, batch: Dict[str, Tuple[Tensor, Tensor, str]]) -> Tensor:
        # supervised
        batch_supervised = batch["labeled"]
        labeled_image_batch, label_batch, _ = batch_supervised
        outputs = self.net(labeled_image_batch)
        supervised_loss = self.criterion(outputs, label_batch)
        # unsupervised
        batch_unusupervised = batch["unlabeled"]
        unlabeled_image_batch, _, _ = batch_unusupervised

        with torch.no_grad():
            self.bn_controller.freeze_bn(self.net_ema)
            ema_output = self.net_ema(unlabeled_image_batch)
            self.bn_controller.unfreeze_bn(self.net_ema)

        self.bn_controller.freeze_bn(self.net)
        outputs_unsup = self.net(unlabeled_image_batch)
        self.bn_controller.unfreeze_bn(self.net)

        if self.global_step < 100:
            unsupervised_loss = 0.0
        else:
            unsupervised_loss = consistency_loss(outputs_unsup, ema_output)

        loss = supervised_loss + unsupervised_loss * self.cons_w_unsup(self.current_epoch)
        self.log("supervised_loss", supervised_loss, on_epoch=True, on_step=False)
        self.log("unsupervised_loss", unsupervised_loss, on_epoch=True, on_step=False)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return {"loss": loss}

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        update_ema_variables(self.net, self.net_ema, 0.99, self.global_step)

    @staticmethod
    def pipeline(get_datamodule, get_trainer, args):
        dataModule = get_datamodule(args)
        trainer, checkpoint_callback = get_trainer(args)
        model = MeanTeacherModule(args)
        dataModule.mode = "semi_train"
        trainer.fit(model=model, datamodule=dataModule)
        trainer.test(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)
