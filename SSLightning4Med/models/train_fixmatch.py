from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from SSLightning4Med.models.base_module import BaseModule
from SSLightning4Med.utils.utils import consistency_weight


class FixmatchModule(BaseModule):
    """Pl Fixmatch training module. The trainng pileline is included.
    Args:
        BaseModule (pl.LightningModule): The base module.
    """

    def __init__(self, args: Any) -> None:
        super(FixmatchModule, self).__init__(args)
        self.args = args
        self.ramp_up = 0.1  # from CCT paper
        self.cons_w_unsup = consistency_weight(
            final_w=1, rampup_ends=self.ramp_up * self.epochs  # from Fixmatch paper
        )
        self.threshold = 0.95
        self.cons_criterion = F.binary_cross_entropy_with_logits if self.args.n_class == 2 else F.cross_entropy

    def training_step(self, batch: Dict[str, Tuple[Tensor, Tensor, str]]) -> Tensor:
        # supervised
        batch_supervised = batch["labeled"]
        labeled_image_batch, label_batch, _ = batch_supervised
        outputs = self.net(labeled_image_batch)
        supervised_loss = self.criterion(outputs, label_batch)
        # unsupervised
        # in the fixmach paper unsupervised batch size is recommended 7x to supervised batch size
        batch_unusupervised_wa = batch["unlabeled_wa"]
        unlabeled_image_batch_wa, _, id_wa = batch_unusupervised_wa
        batch_unusupervised_sa = batch["unlabeled_sa"]
        unlabeled_image_batch_sa, _, id_sa = batch_unusupervised_sa
        assert id_sa == id_wa

        output_wa = self.net(unlabeled_image_batch_wa)
        output_sa = self.net(unlabeled_image_batch_sa)

        pseudo_label = (
            output_wa.detach().sigmoid() if self.args.n_class == 2 else torch.softmax(output_wa.detach(), dim=1)
        )
        if self.args.n_class == 2:
            t = torch.Tensor([0.5]).to(self.device)  # threshold
            targets_u = (pseudo_label > t).float() * 1
            max_probs = pseudo_label
        else:
            max_probs, targets_u = torch.max(pseudo_label, dim=1)
        mask = max_probs.ge(self.threshold).float()

        unsupervised_loss = (self.cons_criterion(output_sa, targets_u, reduction="none") * mask).mean()
        # warmup unsup loss to avoid inital noise
        loss = supervised_loss + unsupervised_loss * self.cons_w_unsup(self.current_epoch)
        self.log("supervised_loss", supervised_loss, on_epoch=True, on_step=False)
        self.log("unsupervised_loss", unsupervised_loss, on_epoch=True, on_step=False)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return {"loss": loss}

    @staticmethod
    def pipeline(get_datamodule, get_trainer, args):
        dataModule = get_datamodule(args)
        trainer, checkpoint_callback = get_trainer(args)
        model = FixmatchModule(args)
        dataModule.mode = "fixmatch_train"
        trainer.fit(model=model, datamodule=dataModule)
        trainer.test(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)
