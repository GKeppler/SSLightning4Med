import os
from argparse import ArgumentParser
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import wandb
from SSLightning4Med.models.base_module import BaseModule
from SSLightning4Med.models.data_module import SemiDataModule
from SSLightning4Med.models.data_module_st import SemiDataModuleST

# from SSLightning4Med.models.train_bolt import BoltModule
from SSLightning4Med.models.train_CCT import CCTModule
from SSLightning4Med.models.train_fixmatch import FixmatchModule
from SSLightning4Med.models.train_mean_teacher import MeanTeacherModule
from SSLightning4Med.models.train_pseudoCCT import pseudoCCTModule
from SSLightning4Med.models.train_stplusplus import STPlusPlusModule
from SSLightning4Med.models.train_stplusplusCCT import STPlusPlusCCTModule
from SSLightning4Med.models.train_supervised import SupervisedModule
from SSLightning4Med.utils.augmentations import Augmentations
from SSLightning4Med.utils.utils import get_color_map


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
    parser.add_argument("--split-file-path", type=str, default=None)
    parser.add_argument("--test-file-path", type=str, default=None)
    parser.add_argument("--pseudo-mask-path", type=str, default=None)
    parser.add_argument("--test-mask-path", type=str, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--reliable-id-path", type=str, default=None)
    parser.add_argument(
        "--oversample", type=bool, default=False, help="use oversampling of labeled data for supervised learning"
    )
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
                "multiorgan": "/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/multiorgan256",
                "brats": "/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/brats",
                "hippocampus": "/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/hippocampus32",
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
    if args.test_mask_path is None:
        args.test_mask_path = f"{args.data_root}/test_masks/{args.method}/{args.split}/split_{args.shuffle}"
    if args.pseudo_mask_path is None:
        args.pseudo_mask_path = f"{args.data_root}/pseudo_masks/{args.method}/{args.split}/split_{args.shuffle}"
    if args.save_path is None:
        args.save_path = f"{args.data_root}/modelcheckpoints/{args.method}/{args.split}/split_{args.shuffle}"
    if args.reliable_id_path is None:
        args.reliable_id_path = f"{args.data_root}/reliable_ids/{args.method}/{args.split}/split_{args.shuffle}"

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    if not os.path.exists(args.test_mask_path):
        os.makedirs(args.test_mask_path, exist_ok=True)
    if not os.path.exists(args.pseudo_mask_path):
        os.makedirs(args.pseudo_mask_path, exist_ok=True)
    return args


def get_trainer(args):
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
    return trainer, checkpoint_callback


def get_datamodule(args):
    # Define DataModule with Augmentations
    augs = Augmentations(args)
    color_map = get_color_map(args.dataset)
    if args.method == "St++":
        dataModule = SemiDataModuleST
    else:
        dataModule = SemiDataModule
    dataModule = dataModule(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        batch_size_unlabeled=args.batch_size_unlabeled,
        split_yaml_path=args.split_file_path,
        test_yaml_path=args.test_file_path,
        pseudo_mask_path=args.pseudo_mask_path,
        mode="train",
        color_map=color_map,
        num_workers=args.n_workers,
        oversample=args.oversample,
    )

    dataModule.val_transforms = augs.a_val_transforms()
    dataModule.pred_transforms = augs.a_pred_transforms()
    dataModule.train_transforms = augs.a_train_transforms_weak()
    dataModule.train_transforms_unlabeled = (
        augs.a_train_transforms_strong_stplusplus()
        if args.method == "St++" or args.method == "Fixmatch"
        else augs.a_train_transforms_weak()
    )
    return dataModule


if __name__ == "__main__":
    args = base_parse_args(BaseModule)
    seed_everything(123, workers=True)
    # define Module based on methods
    module = {
        "St++": STPlusPlusModule,
        "CCT": CCTModule,
        "Supervised": SupervisedModule,
        "MeanTeacher": MeanTeacherModule,
        "FixMatch": FixmatchModule,
        "PseudoCCT": pseudoCCTModule,
        "St++CCT": STPlusPlusCCTModule,
        # "Bolt": BoltModule,
    }[args.method]
    module.pipeline(get_datamodule, get_trainer, args)
