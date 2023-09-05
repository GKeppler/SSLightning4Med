import os
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from SSLightning4Med.models.base_module import BaseModule
from SSLightning4Med.utils.utils import consistency_weight, wandb_image_mask


def consistency_loss(logits_w1, logits_w2):
    assert logits_w1.size() == logits_w2.size()
    if logits_w1.shape[1] == 1:
        return F.mse_loss(torch.sigmoid(logits_w1), torch.sigmoid(logits_w2))
    else:
        return F.mse_loss(torch.softmax(logits_w1, dim=-1), torch.softmax(logits_w2, dim=-1), reduction="mean")


class CCTModule(BaseModule):
    """Pl CCT training module. The trainng pileline is included.
        This is the implementation of the CCT SSL approach
    -   custom unet with aux decoders 1-3
        Args:
            BaseModule (pl.LightningModule): The base module.
    """

    def __init__(self, args: Any):
        super(CCTModule, self).__init__(args)
        self.ramp_up = 0.1  # from CCT paper
        self.cons_w_unsup = consistency_weight(final_w=30, rampup_ends=self.ramp_up * self.epochs)  # from CCT paper

    def training_step(self, batch: Dict[str, Tuple[Tensor, Tensor, str]]) -> Tensor:
        batch_unusupervised = batch["unlabeled"]
        batch_supervised = batch["labeled"]

        # calculate all outputs based on the different decoders
        image_batch, label_batch, _ = batch_supervised

        outputs, outputs_aux1, outputs_aux2, outputs_aux3 = self.net(image_batch)
        # calc losses for labeled batch
        label_batch = label_batch.long()
        loss_ce = self.criterion(outputs, label_batch)
        loss_ce_aux1 = self.criterion(outputs_aux1, label_batch)
        loss_ce_aux2 = self.criterion(outputs_aux2, label_batch)
        loss_ce_aux3 = self.criterion(outputs_aux3, label_batch)

        supervised_loss = (loss_ce + loss_ce_aux1 + loss_ce_aux2 + loss_ce_aux3) / 4

        # calucalate unsupervised loss
        image_batch, _, _ = batch_unusupervised
        outputs, outputs_aux1, outputs_aux2, outputs_aux3 = self.net(image_batch)

        consistency_loss_aux1 = consistency_loss(outputs, outputs_aux1)
        consistency_loss_aux2 = consistency_loss(outputs, outputs_aux2)
        consistency_loss_aux3 = consistency_loss(outputs, outputs_aux3)

        unsupervised_loss = (consistency_loss_aux1 + consistency_loss_aux2 + consistency_loss_aux3) / 3
        # warmup unsup loss to avoid inital noise
        loss = supervised_loss + unsupervised_loss * self.cons_w_unsup(self.current_epoch)
        self.log("supervised_loss", supervised_loss, on_epoch=True, on_step=False)
        self.log("unsupervised_loss", unsupervised_loss, on_epoch=True, on_step=False)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):  # type: ignore
        img, mask, id = batch
        logits = self(img)[0]
        val_loss = self.val_criterion(logits, mask)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)
        pred = self.oneHot(logits)
        self.val_IoU(pred.to(device=self.device), mask)
        self.log("val_mIoU", self.val_IoU, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):  # type: ignore
        img, mask, id = batch
        pred = self.net(img)[0]
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

    @staticmethod
    def pipeline(get_datamodule, get_trainer, args):
        dataModule = get_datamodule(args)
        trainer, checkpoint_callback = get_trainer(args)
        model = CCTModule(args)
        dataModule.mode = "semi_train"
        trainer.fit(model=model, datamodule=dataModule)
        trainer.test(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)
