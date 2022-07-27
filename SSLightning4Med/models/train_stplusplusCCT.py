import os
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from torchmetrics import JaccardIndex

from SSLightning4Med.models.base_module import BaseModule
from SSLightning4Med.utils.utils import get_color_map, wandb_image_mask


def consistency_loss(logits_w1, logits_w2):
    assert logits_w1.size() == logits_w2.size()
    if logits_w1.shape[1] == 1:
        return F.mse_loss(torch.sigmoid(logits_w1), torch.sigmoid(logits_w2))
    else:
        return F.mse_loss(torch.softmax(logits_w1, dim=-1), torch.softmax(logits_w2, dim=-1), reduction="mean")


class STPlusPlusCCTModule(BaseModule):
    def __init__(self, args: Any) -> None:
        super(STPlusPlusCCTModule, self).__init__(args)
        self.checkpoints: List[torch.nn.Module] = []
        self.id_to_reliability: List[Tuple] = []
        self.previous_best: float = 0.0
        self.args = args
        self.mode = "label"
        self.color_map = get_color_map(args.dataset)

    def training_step(self, batch: Dict[str, Tuple[Tensor, Tensor, str]]) -> Tensor:
        img, mask, _ = batch["labeled"]
        outputs, outputs_aux1, outputs_aux2, outputs_aux3 = self.net(img)
        # calc losses for labeled batch
        label_batch = mask.long()
        loss_ce = self.criterion(outputs, label_batch)
        loss_ce_aux1 = self.criterion(outputs_aux1, mask)
        loss_ce_aux2 = self.criterion(outputs_aux2, mask)
        loss_ce_aux3 = self.criterion(outputs_aux3, mask)

        # combine batches
        if "pseudolabeled" in batch:
            img_pseudo, mask_pseudo, _ = batch["pseudolabeled"]
            # torch.unsqueeze(mask, dim=-1).shape
            outputs, outputs_aux1, outputs_aux2, outputs_aux3 = self.net(img_pseudo)
            loss_ce_aux1 += self.criterion(outputs_aux1, mask_pseudo)
            loss_ce_aux2 += self.criterion(outputs_aux2, mask_pseudo)
            loss_ce_aux3 += self.criterion(outputs_aux3, mask_pseudo)

        loss = (
            (loss_ce + loss_ce_aux1 + loss_ce_aux2 + loss_ce_aux3) / 7
            if "pseudolabeled" in batch
            else (loss_ce + loss_ce_aux1 + loss_ce_aux2 + loss_ce_aux3) / 4
        )

        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        self.checkpoints.append(deepcopy(self.net).cpu())

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

    def on_predict_start(self) -> None:
        if self.mode == "select_reliable":
            self.checkpoints[len(self.checkpoints) // 3].to(self.device)
            self.checkpoints[len(self.checkpoints) * 2 // 3].to(self.device)
            self.checkpoints[-1].to(self.device)

    def predict_step(self, batch: List[Union[Tensor, Tuple[str]]], batch_idx: int) -> None:
        img, mask, id = batch
        if self.mode == "label":
            pred = self(img)[0]
            pred = self.oneHot(pred)
            pred = pred.squeeze(0).cpu().numpy().astype(np.uint8)
            pred = np.array(self.color_map)[pred]
            cv2.imwrite(
                "%s/%s" % (self.args.pseudo_mask_path, os.path.basename(id[0].split(" ")[1])),
                cv2.cvtColor(pred.astype(np.uint8), cv2.COLOR_BGR2RGB),
            )
        if self.mode == "select_reliable":
            preds = []
            # for model in self.checkpoints:
            #     preds.append(self.oneHot(model(img)).to(torch.long))
            pred = self.checkpoints[len(self.checkpoints) // 3](img)[0]
            preds.append(self.oneHot(pred).to(torch.long))
            pred = self.checkpoints[len(self.checkpoints) * 2 // 3](img)[0]
            preds.append(self.oneHot(pred).to(torch.long))
            pred = self.checkpoints[-1](img)[0]
            preds.append(self.oneHot(pred).to(torch.long))
            mIOU = []
            for i in range(len(preds) - 1):
                metric = JaccardIndex(
                    # ignore_index=0,
                    num_classes=self.n_class,
                ).to(device=self.device)
                mIOU.append(metric(preds[i].to(device=self.device), preds[-1].to(device=self.device)))
            reliability = sum(mIOU) / len(mIOU)
            self.id_to_reliability.append((id[0], reliability.cpu().item()))

    def on_predict_epoch_end(self, results: List[Any]) -> None:
        if self.mode == "select_reliable":
            labeled_ids = []
            val_ids = []
            with open(self.args.split_file_path, "r") as file:
                split_dict = yaml.load(file, Loader=yaml.FullLoader)
                labeled_ids = split_dict[self.args.val_split]["labeled"]
                val_ids = split_dict[self.args.val_split]["val"]

            self.id_to_reliability.sort(key=lambda elem: elem[1], reverse=True)
            yaml_dict = dict()
            yaml_dict[self.args.val_split] = dict(
                val=val_ids,
                labeled=labeled_ids,
                unlabeled=[i[0] for i in self.id_to_reliability[: len(self.id_to_reliability) // 2]],  # reliable ids
            )
            # save to yaml
            if not os.path.exists(self.args.reliable_id_path):
                os.makedirs(self.args.reliable_id_path)

            with open(os.path.join(self.args.reliable_id_path, "reliable_ids.yaml"), "w+") as outfile:
                yaml.dump(yaml_dict, outfile, default_flow_style=False)
        self.set_metrics()

    @staticmethod
    def pipeline(get_datamodule, get_trainer, args):
        dataModule = get_datamodule(args)
        trainer, checkpoint_callback = get_trainer(args)

        model = STPlusPlusCCTModule(args)
        # <====================== Supervised training with labeled images (SupOnly) ======================>
        # trainer.tune(model, dataModule)
        print(
            "\n================> Total stage 1/%i: "
            "Supervised training on labeled images (SupOnly)" % (6 if args.plus else 3)
        )
        trainer.fit(model=model, datamodule=dataModule)

        # <====================== Test supervised model on testset (SupOnly) ======================>
        # trainer.test(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)

        if not args.plus:
            # <============================= Pseudolabel all unlabeled images =============================>
            print("\n\n\n================> Total stage 2/3: Pseudo labeling all unlabeled images")
            trainer.predict(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)

            # <======================== Re-training on labeled and unlabeled images ========================>
            print("\n\n\n================> Total stage 3/3: Re-training on labeled and unlabeled images")
            model = STPlusPlusCCTModule(args)
            # increase max epochs to double the amount
            trainer.fit_loop.epoch_progress.reset()
            dataModule.mode = "pseudo_train"
            trainer.fit(model=model, datamodule=dataModule)
        else:
            # <===================================== Select Reliable IDs =====================================>
            print("\n\n\n================> Total stage 2/6: Select reliable images for the 1st stage re-training")
            model.mode = "select_reliable"
            trainer.predict(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)

            # <================================ Pseudo label reliable images =================================>
            print("\n\n\n================> Total stage 3/6: Pseudo labeling reliable images")
            dataModule.split_yaml_path = os.path.join(args.reliable_id_path, "reliable_ids.yaml")
            dataModule.setup_split()
            model.mode = "label"
            trainer.predict(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)

            # <================================== The 1st stage re-training ==================================>
            print(
                "\n\n\n================> Total stage 4/6: The 1st stage re-training\
                     on labeled and reliable unlabeled images"
            )
            model = STPlusPlusCCTModule(args)
            trainer, checkpoint_callback = get_trainer(args)
            dataModule.mode = "pseudo_train"
            trainer.fit(model=model, datamodule=dataModule)

            # <=============================== Pseudo label all images ================================>
            print("\n\n\n================> Total stage 5/6: Pseudo labeling all images")
            dataModule.split_yaml_path = args.split_file_path
            dataModule.setup_split()
            model.mode = "label"
            trainer.predict(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)

            # <================================== The 2nd stage re-training ==================================>
            print(
                "\n\n\n================> Total stage 6/6: The 2nd stage re-training \
                    on labeled and all unlabeled images"
            )
            model = STPlusPlusCCTModule(args)
            trainer, checkpoint_callback = get_trainer(args)
            trainer.fit(model=model, datamodule=dataModule)

        # <====================== Test supervised model on testset (Semi) ======================>
        print("\n\n\n================> Test supervised model on testset (Re-trained)")
        trainer.test(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)
