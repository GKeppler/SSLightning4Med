from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor

from SSLightning4Med.models.base_module import BaseModule
from SSLightning4Med.models.data_module import SemiDataModule
from SSLightning4Med.utils.utils import sigmoid_rampup


class FixmatchModule(BaseModule):
    def __init__(self, args: Any) -> None:
        super(FixmatchModule, self).__init__(args)
        self.args = args
        self.consistency = 0.1

    def training_step(self, batch: Dict[str, Tuple[Tensor, Tensor, str]]) -> Tensor:
        # supervised
        batch_supervised = batch["labeled"]
        labeled_image_batch, label_batch, _ = batch_supervised
        outputs = self.net(labeled_image_batch)
        supervised_loss = self.criterion(outputs, label_batch)
        # unsupervised
        batch_unusupervised_wa = batch["unlabeled_wa"]
        unlabeled_image_batch_wa, _, id_wa = batch_unusupervised_wa
        batch_unusupervised_sa = batch["unlabeled_sa"]
        unlabeled_image_batch_sa, _, id_sa = batch_unusupervised_sa
        assert id_sa == id_wa
        threshold = 0.95

        output_wa = self.net(unlabeled_image_batch_wa)
        output_sa = self.net(unlabeled_image_batch_sa)

        pseudo_label = (
            torch.softmax(output_wa.detach(), dim=1) if self.args.n_class == 2 else output_wa.detach().sigmoid()
        )
        max_probs, targets_u = torch.max(pseudo_label, dim=1)

        mask = max_probs.ge(threshold).float()

        unsupervised_loss = (F.cross_entropy(output_sa, targets_u, reduction="none") * mask).mean()
        # warmup unsup loss to avoid inital noise
        consistency_weight = self.consistency * sigmoid_rampup(self.current_epoch)
        loss = supervised_loss + unsupervised_loss * consistency_weight
        self.log("supervised_loss", supervised_loss, on_epoch=True, on_step=True)
        self.log("unsupervised_loss", unsupervised_loss, on_epoch=True, on_step=True)
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        return {"loss": loss}

    def pipeline(dataModule: SemiDataModule, trainer: pl.Trainer, checkpoint_callback: ModelCheckpoint, args) -> None:
        model = FixmatchModule(args)
        dataModule.mode = "fixmatch_train"
        trainer.fit(model=model, datamodule=dataModule)
        trainer.test(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)
