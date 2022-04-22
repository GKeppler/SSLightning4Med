import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor

from SSLightning4Med.models.base_model import BaseModule
from SSLightning4Med.models.data_module import SemiDataModule
from SSLightning4Med.utils.utils import get_color_map, meanIOU


class STPlusPlusModule(BaseModule):
    def __init__(self, args: Any) -> None:
        super(STPlusPlusModule, self).__init__(args)
        self.checkpoints: List[torch.nn.Module] = []
        self.id_to_reliability: List[Tuple] = []
        self.previous_best: float = 0.0
        self.args = args
        self.mode = "label"
        self.color_map = get_color_map(args.dataset)

    def base_forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        x = self.net(x)
        x = F.interpolate(x, (h, w), mode="bilinear", align_corners=True)
        return x

    def forward(self, x: Tensor, tta: bool = False) -> Optional[Tensor]:
        if not tta:
            return self.base_forward(x)

        h, w = x.shape[-2:]
        # scales = [0.5, 0.75, 1.0]
        # to avoid cuda out of memory
        scales = [0.5, 0.75, 1.0, 1.5, 2.0]

        final_result = None

        for scale in scales:
            cur_h, cur_w = int(h * scale), int(w * scale)
            cur_x = F.interpolate(x, size=(cur_h, cur_w), mode="bilinear", align_corners=True)

            out = F.softmax(self.base_forward(cur_x), dim=1)
            out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
            final_result = out if final_result is None else (final_result + out)

            out = F.softmax(self.base_forward(cur_x.flip(3)), dim=1).flip(3)
            out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
            final_result += out

        return final_result

    def training_step(self, batch: Dict[str, Tuple[Tensor, Tensor, str]]) -> Tensor:
        img, mask, _ = batch["labeled"]
        # combine batches
        if "pseudolabeled" in batch:
            img_pseudo, mask_pseudo, _ = batch["pseudolabeled"]
            # torch.unsqueeze(mask, dim=-1).shape
            img = torch.cat((img, img_pseudo), 0)
            mask = torch.cat((mask, mask_pseudo), 0)
        pred = self(img)
        loss = self.criterion(pred, mask)
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        if (self.current_epoch + 1) in [
            self.args.epochs // 3,
            self.args.epochs * 2 // 3,
            self.args.epochs,
        ]:
            self.checkpoints.append(deepcopy(self.net))

    def validation_step(self, batch: Tuple[Tensor, Tensor, str], batch_idx: int) -> Dict[str, float]:
        img, mask, id = batch
        pred = self(img)
        self.val_IoU(pred, mask)
        # self.log("val_loss", self.criterion(pred, mask), on_epoch=True)
        self.log("val_mIoU", self.val_IoU, on_step=False, on_epoch=True)

    def validation_epoch_end(self, outputs: List[Dict[str, float]]) -> Dict[str, Union[Dict[str, float], float]]:
        mIOU = self.val_IoU.compute()
        if mIOU > self.previous_best:
            if self.previous_best != 0:
                os.remove(
                    os.path.join(
                        self.args.save_path,
                        "%s_mIOU%.2f.pth" % (self.args.net, self.previous_best),
                    )
                )
            self.previous_best = mIOU
            torch.save(
                self.state_dict(),
                os.path.join(
                    self.args.save_path,
                    "%s_mIOU%.2f.pth" % (self.args.net, mIOU),
                ),
            )
        self.set_metrics()

    def predict_step(self, batch: List[Union[Tensor, Tuple[str]]], batch_idx: int) -> None:
        img, mask, id = batch
        if self.mode == "label":
            pred = self(img, tta=True)
            pred = self.oneHot(pred.cpu())
            pred = pred.squeeze(0).numpy().astype(np.uint8)
            pred = np.array(self.color_map)[pred]
            cv2.imwrite(
                "%s/%s" % (self.args.pseudo_mask_path, os.path.basename(id[0].split(" ")[1])),
                cv2.cvtColor(pred.astype(np.uint8), cv2.COLOR_BGR2RGB),
            )
        if self.mode == "select_reliable":
            preds = []
            for model in self.checkpoints:
                preds.append(torch.argmax(model(img), dim=1).cpu().numpy())
            mIOU = []
            for i in range(len(preds) - 1):
                metric = meanIOU(self.args.n_class)
                metric.add_batch(preds[i], preds[-1])
                mIOU.append(metric.evaluate()[-1])
            reliability = sum(mIOU) / len(mIOU)
            self.id_to_reliability.append((id[0], reliability))

    def on_predict_epoch_end(self, results: List[Any]) -> None:
        if self.mode == "select_reliable":
            labeled_ids = []
            with open(self.args.split_file_path, "r") as file:
                split_dict = yaml.load(file, Loader=yaml.FullLoader)
                labeled_ids = split_dict[self.args.val_split]["labeled"]
                val_ids = labeled_ids = split_dict[self.args.val_split]["val"]

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
    def pipeline(dataModule: SemiDataModule, trainer: pl.Trainer, checkpoint_callback: ModelCheckpoint, args) -> None:

        model = STPlusPlusModule(args)
        # <====================== Supervised training with labeled images (SupOnly) ======================>
        # trainer.tune(model, dataModule)
        trainer.fit(model=model, datamodule=dataModule)

        # <====================== Test supervised model on testset (SupOnly) ======================>
        trainer.test(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)

        if not args.plus:
            # <============================= Pseudolabel all unlabeled images =============================>

            trainer.predict(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)

            # <======================== Re-training on labeled and unlabeled images ========================>

            model = STPlusPlusModule(args)

            # increase max epochs to double the amount
            trainer.fit_loop.max_epochs = args.epochs * 2
            dataModule.mode = "pseudo_train"
            trainer.fit(model=model, datamodule=dataModule)
        else:
            # <===================================== Select Reliable IDs =====================================>
            model.mode = "select_reliable"
            trainer.predict(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)
            # <================================ Pseudo label reliable images =================================>
            dataModule.split_yaml_path = os.path.join(args.reliable_id_path, "reliable_ids.yaml")
            model.mode = "label"
            trainer.predict(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)
            # <================================== The 1st stage re-training ==================================>

            model = STPlusPlusModule(args)

            # increase max epochs to double the amount
            trainer.fit_loop.max_epochs = args.epochs * 2
            dataModule.mode = "pseudo_train"
            trainer.fit(model=model, datamodule=dataModule)
            # <=============================== Pseudo label all images ================================>

            dataModule.split_yaml_path = args.split_file_path
            model.mode = "label"
            trainer.predict(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)

            # <================================== The 2nd stage re-training ==================================>
            model = STPlusPlusModule(args)

            # increase max epochs to double the amount
            trainer.fit_loop.max_epochs = args.epochs * 3
            trainer.fit(model=model, datamodule=dataModule)

        # <====================== Test supervised model on testset (Semi) ======================>
        trainer.test(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)


# if __name__ == "__main__":
#     module = STPlusPlusModule

#     args = base_parse_args(module)
#     seed_everything(123, workers=True)
#     augs = Augmentations(args)
#     color_map = get_color_map(args.dataset)
#     dataModule = SemiDataModule(
#         root_dir=args.data_root,
#         batch_size=args.batch_size,
#         split_yaml_path=args.split_file_path,
#         test_yaml_path=args.test_file_path,
#         pseudo_mask_path=args.pseudo_mask_path,
#         mode="train",
#         color_map=color_map,
#         num_workers=args.n_workers,
#     )

#     dataModule.val_transforms = augs.a_val_transforms()
#     dataModule.train_transforms = augs.a_train_transforms_labeled()
#     dataModule.train_transforms_unlabeled = augs.a_train_transforms_unlabeled()


#     # saves a file like: my/path/sample-epoch=02-val_loss=0.32.ckpt
#     checkpoint_callback = ModelCheckpoint(
#         dirpath=os.path.join("./", f"{args.save_path}"),
#         filename=f"{args.net}" + "-{epoch:02d}-{val_mIoU:.2f}",
#         mode="max",
#         save_weights_only=True,
#     )
#     if args.use_wandb:
#         wandb.init(project="SSLightning4Med", entity="gkeppler")
#         wandb_logger = WandbLogger(project="SSLightning4Med")
#         wandb.config.update(args)

#     dev_run = False  # not working when predicting with best_model checkpoint
#     trainer = pl.Trainer.from_argparse_args(
#         args,
#         fast_dev_run=dev_run,
#         max_epochs=args.epochs,
#         log_every_n_steps=2,
#         logger=wandb_logger if args.use_wandb else TensorBoardLogger("./tb_logs"),
#         callbacks=[checkpoint_callback],
#         gpus=[0],
#         #precision=16,
#         # accelerator="cpu",
#         # profiler="pytorch",
#         auto_lr_find=True,
#     )
#     module.pipeline(dataModule, trainer, checkpoint_callback, args)
