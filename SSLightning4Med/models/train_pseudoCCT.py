from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from SSLightning4Med.models.base_module import BaseModule
from SSLightning4Med.utils.utils import consistency_weight, wandb_image_mask


class pseudoCCTModule(BaseModule):
    def __init__(self, args: Any) -> None:
        super(pseudoCCTModule, self).__init__(args)
        self.args = args
        self.ramp_up = 0.1  # from CCT paper
        self.cons_w_unsup = consistency_weight(
            final_w=1, rampup_ends=self.ramp_up * self.epochs  # from Fixmatch paper
        )
        self.threshold = 0.95
        self.cons_criterion = F.binary_cross_entropy_with_logits if self.args.n_class == 2 else F.cross_entropy

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
        output_wa, output_sa1, output_sa2, output_sa3 = self.net(image_batch)

        pseudo_label = (
            output_wa.detach().sigmoid() if self.args.n_class == 2 else torch.softmax(output_wa.detach(), dim=1)
        )
        if self.args.n_class == 2:
            t = torch.Tensor([0.5], device=self.device)  # threshold
            targets_u = (pseudo_label > t).float() * 1
            max_probs = pseudo_label
        else:
            max_probs, targets_u = torch.max(pseudo_label, dim=1)
        mask = max_probs.ge(self.threshold).float()
        unsupervised_loss1 = (self.cons_criterion(output_sa1, targets_u, reduction="none") * mask).mean()
        unsupervised_loss2 = (self.cons_criterion(output_sa2, targets_u, reduction="none") * mask).mean()
        unsupervised_loss3 = (self.cons_criterion(output_sa3, targets_u, reduction="none") * mask).mean()

        unsupervised_loss = (unsupervised_loss1 + unsupervised_loss2 + unsupervised_loss3) / 3
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
        return wandb_image_mask(img, mask, pred, self.n_class)

    @staticmethod
    def pipeline(get_datamodule, get_trainer, args):
        dataModule = get_datamodule(args)
        trainer, checkpoint_callback = get_trainer(args)
        model = pseudoCCTModule(args)
        dataModule.mode = "semi_train"
        trainer.fit(model=model, datamodule=dataModule)
        trainer.test(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)
