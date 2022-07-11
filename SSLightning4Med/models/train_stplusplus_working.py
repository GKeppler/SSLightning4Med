import os
from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch import Tensor
from torchmetrics import JaccardIndex

import wandb
from SSLightning4Med.models.base_module import BaseModule
from SSLightning4Med.models.data_module import SemiDataModule
from SSLightning4Med.utils.augmentations import Augmentations
from SSLightning4Med.utils.utils import get_color_map


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
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        if (self.current_epoch + 1) in [
            self.args.epochs // 3,
            self.args.epochs * 2 // 3,
            self.args.epochs,
        ]:
            self.checkpoints.append(deepcopy(self.net))

    def predict_step(self, batch: List[Union[Tensor, Tuple[str]]], batch_idx: int) -> None:
        img, mask, id = batch
        if self.mode == "label":
            pred = self(img, tta=self.args.use_tta)
            pred = self.oneHot(pred)
            pred = pred.squeeze(0).cpu().numpy().astype(np.uint8)
            pred = np.array(self.color_map)[pred]
            cv2.imwrite(
                "%s/%s" % (self.args.pseudo_mask_path, os.path.basename(id[0].split(" ")[1])),
                cv2.cvtColor(pred.astype(np.uint8), cv2.COLOR_BGR2RGB),
            )
        if self.mode == "select_reliable":
            preds = []
            for model in self.checkpoints:
                preds.append(self.oneHot(model(img)).to(torch.long))
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
    def pipeline(dataModule: SemiDataModule, trainer: pl.Trainer, checkpoint_callback: ModelCheckpoint, args) -> None:

        model = STPlusPlusModule(args)
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
            model = STPlusPlusModule(args)
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
            model = STPlusPlusModule(args)
            # increase max epochs to double the amount
            trainer.fit_loop.epoch_progress.reset()
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
            model = STPlusPlusModule(args)
            # increase max epochs to double the amount
            trainer.fit_loop.epoch_progress.reset()
            trainer.fit(model=model, datamodule=dataModule)

        # <====================== Test supervised model on testset (Semi) ======================>
        print("\n\n\n================> Test supervised model on testset (Re-trained)")
        trainer.test(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)


def base_parse_args(LightningModule) -> Any:  # type: ignore
    parser = ArgumentParser()
    # basic settings
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["melanoma", "pneumothorax", "breastCancer", "multiorgan", "brats", "hippocampus", "zebrafish"],
        default="multiorgan",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batch-size-unlabeled", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--base-size", type=int, default=None)
    parser.add_argument("--n-class", type=int, default=None)
    parser.add_argument("--n-channel", type=int, default=None)
    parser.add_argument("--n-workers", type=int, default=10)
    parser.add_argument("--loss", type=str, default="Dice", choices=["CE", "Dice"])
    parser.add_argument("--early-stopping", default=True)
    parser.add_argument("--val-split", type=str, default="val_split_0")

    # semi-supervised settings
    parser.add_argument("--split", type=str, default="1_30")
    parser.add_argument("--shuffle", type=int, default=0)
    # these are derived from the above split, shuffle and dataset. They dont need to be set
    parser.add_argument(
        "--split-file-path", type=str, default=None
    )  # "dataset/splits/melanoma/1_30/split_0/split_sample.yaml")
    parser.add_argument("--test-file-path", type=str, default=None)  # "dataset/splits/melanoma/test_sample.yaml")
    parser.add_argument("--pseudo-mask-path", type=str, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--reliable-id-path", type=str, default=None)
    parser.add_argument("--use-wandb", default=False, help="whether to use WandB for logging")
    parser.add_argument("--wandb-project", type=str, default="SSLightning4Med")
    parser.add_argument("--lsdf", default=True, help="whether to use the LSDF storage")
    # add model specific args
    parser = LightningModule.add_model_specific_args(parser)
    # add all the availabele trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    # parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    if args.method is None:
        raise ValueError("no methodname in model_specific_args specified    .")
    if args.data_root is None:
        if args.lsdf:
            args.data_root = {
                "melanoma": "/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/melanoma256",
                "breastCancer": "/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/breastCancer256",
                "pneumothorax": "/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/pneumothorax",
                "multiorgan": "/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/multiorgan",
                "brats": "/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/brats",
                "hippocampus": "/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/hippocampus",
                "zebrafish": "/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/zebrafish256",
            }[args.dataset]
        else:
            args.data_root = {
                "melanoma": "/home/kit/stud/uwdus/Masterthesis/data/melanoma256",
                "breastCancer": "/home/kit/stud/uwdus/Masterthesis/data/breastCancer256",
                "pneumothorax": "/home/kit/stud/uwdus/Masterthesis/data/pneumothorax",
                "multiorgan": "/home/kit/stud/uwdus/Masterthesis/data/multiorgan",
                "brats": "/home/kit/stud/uwdus/Masterthesis/data/brats",
                "hippocampus": "/home/kit/stud/uwdus/Masterthesis/data/hippocampus",
                "zebrafish": "/home/kit/stud/uwdus/Masterthesis/data/zebrafish256",
            }[args.dataset]

    if args.epochs is None:
        args.epochs = 100
    if args.crop_size is None:
        args.crop_size = {
            "melanoma": 256,
            "breastCancer": 256,
            "pneumothorax": 256,
            "multiorgan": 256,
            "brats": 224,
            "hippocampus": 32,
            "zebrafish": 256,
        }[args.dataset]
    if args.base_size is None:
        args.base_size = {
            "melanoma": 512,
            "breastCancer": 512,
            "pneumothorax": 512,
            "multiorgan": 512,
            "brats": 240,
            "hippocampus": 50,
            "zebrafish": 480,
        }[args.dataset]
    if args.n_class is None:
        args.n_class = {
            "melanoma": 2,
            "breastCancer": 2,
            "pneumothorax": 2,
            "multiorgan": 14,
            "brats": 4,
            "hippocampus": 3,
            "zebrafish": 4,
        }[args.dataset]
    if args.n_channel is None:
        args.n_channel = {
            "melanoma": 3,
            "breastCancer": 3,
            "pneumothorax": 1,
            "multiorgan": 1,
            "brats": 1,
            "hippocampus": 1,
            "zebrafish": 3,
        }[args.dataset]

    if args.batch_size_unlabeled is None:
        args.batch_size_unlabeled = args.batch_size

    if args.split_file_path is None:
        args.split_file_path = (
            f"SSLightning4Med/data/splits/{args.dataset}/{args.split}/split_{args.shuffle}/split.yaml"
        )
    if args.test_file_path is None:
        args.test_file_path = f"SSLightning4Med/data/splits/{args.dataset}/test.yaml"
    if args.pseudo_mask_path is None:
        args.pseudo_mask_path = f"{args.data_root}/pseudo_masks/{args.method}/{args.split}/split_{args.shuffle}"
    if args.save_path is None:
        args.save_path = f"{args.data_root}/modelcheckpoints/{args.method}/{args.split}/split_{args.shuffle}"
    if args.reliable_id_path is None:
        args.reliable_id_path = f"{args.data_root}/reliable_ids/{args.method}/{args.split}/split_{args.shuffle}"

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    if not os.path.exists(args.pseudo_mask_path):
        os.makedirs(args.pseudo_mask_path, exist_ok=True)
    return args


def load_Stuff(args):
    augs = Augmentations(args)
    color_map = get_color_map(args.dataset)
    dataModule = SemiDataModule(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        batch_size_unlabeled=args.batch_size_unlabeled,
        split_yaml_path=args.split_file_path,
        test_yaml_path=args.test_file_path,
        pseudo_mask_path=args.pseudo_mask_path,
        mode="train",
        color_map=color_map,
        num_workers=args.n_workers,
    )

    dataModule.val_transforms = augs.a_val_transforms()
    dataModule.train_transforms = augs.a_train_transforms_weak()
    dataModule.train_transforms_unlabeled = (
        augs.a_train_transforms_strong_stplusplus()
        if args.method == "St++" or args.method == "Fixmatch"
        else augs.a_train_transforms_weak()
    )

    # saves a file like: my/path/sample-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor="val_mIoU",
        dirpath=os.path.join(f"{args.save_path}"),
        filename=f"{args.net}" + "-{epoch:02d}-{val_mIoU:.3f}",
        mode="max",
        save_weights_only=True,
    )
    checkpoint_callback2 = ModelCheckpoint(
        dirpath=os.path.join(f"{args.save_path}"),
        filename=f"{args.net}" + "-{epoch:02d}-{val_mIoU:.3f}",
        mode="max",
        save_weights_only=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # early Stopping
    early_stopping = EarlyStopping(
        monitor="val_mIoU",
        patience=50,
        mode="max",
        verbose=True,
        min_delta=0.01,
    )

    if args.use_wandb:
        wandb.init(project=args.wandb_project, entity="gkeppler")
        wandb_logger = WandbLogger(project=args.wandb_project)
        wandb.config.update(args)

    # define Trainer
    dev_run = False  # not working when predicting with best_model checkpoint
    callbacks = [checkpoint_callback, checkpoint_callback2, lr_monitor]
    if args.early_stopping:
        callbacks.append(early_stopping)
    trainer = pl.Trainer.from_argparse_args(
        args,
        fast_dev_run=dev_run,
        max_epochs=args.epochs,
        logger=wandb_logger if args.use_wandb else TensorBoardLogger("./tb_logs"),
        callbacks=callbacks,
        gpus=[0],
        precision=16,
        log_every_n_steps=2,
        # accelerator="cpu",
        # profiler="simple",
        # auto_lr_find=True,
        # track_grad_norm=True,
        # detect_anomaly=True,
        # overfit_batches=1,
    )
    return dataModule, trainer, checkpoint_callback, args


if __name__ == "__main__":
    args = base_parse_args(BaseModule)
    model = STPlusPlusModule(args)
    dataModule, trainer, checkpoint_callback, args = load_Stuff(args)
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
        model = STPlusPlusModule(args)
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
        model = STPlusPlusModule(args)
        dataModule, trainer, checkpoint_callback, args = load_Stuff(args)
        dataModule.split_yaml_path = os.path.join(args.reliable_id_path, "reliable_ids.yaml")
        dataModule.setup_split()
        # increase max epochs to double the amount
        # trainer.fit_loop.epoch_progress.reset()
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
        model = STPlusPlusModule(args)
        # increase max epochs to double the amount
        dataModule, trainer, checkpoint_callback, args = load_Stuff(args)
        dataModule.split_yaml_path = args.split_file_path
        dataModule.setup_split()
        dataModule.mode = "pseudo_train"
        # trainer.fit_loop.epoch_progress.reset()
        trainer.fit(model=model, datamodule=dataModule)

    # <====================== Test supervised model on testset (Semi) ======================>
    print("\n\n\n================> Test supervised model on testset (Re-trained)")
    trainer.test(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)
