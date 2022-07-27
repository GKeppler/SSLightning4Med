import os
from glob import glob

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import wandb
from SSLightning4Med.models.base_module import BaseModule
from SSLightning4Med.models.data_module import SemiDataModule

# from SSLightning4Med.models.train_bolt import BoltModule
from SSLightning4Med.models.train_CCT import CCTModule
from SSLightning4Med.models.train_fixmatch import FixmatchModule
from SSLightning4Med.models.train_mean_teacher import MeanTeacherModule
from SSLightning4Med.models.train_stplusplus import STPlusPlusModule
from SSLightning4Med.models.train_stplusplusCCT import STPlusPlusCCTModule
from SSLightning4Med.models.train_supervised import SupervisedModule
from SSLightning4Med.train import base_parse_args
from SSLightning4Med.utils.augmentations import Augmentations
from SSLightning4Med.utils.utils import get_color_map


def main(args):
    seed_everything(123, workers=True)
    # Define DataModule with Augmentations
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
    # get filenames at dirpath os.path.join(f"{args.save_path}"),

    # load checkpoint with highest val_mIoU

    if args.use_wandb:
        # https://pytorch-lightning.readthedocs.io/en/1.5.0/extensions/generated/pytorch_lightning.loggers.WandbLogger.html
        wandb.init(project=args.wandb_project, entity="gkeppler")
        wandb_logger = WandbLogger(project=args.wandb_project)
        wandb.config.update(args)

    files = glob(os.path.join(f"{args.save_path}", "*.ckpt"))
    # get checkpoint with highest val_mIoU
    if len(files) != 0:
        max_mIoU = max([float(f.split("val_mIoU")[-1][1:5]) for f in files])
        max_mIoU_file = [f for f in files if float(f.split("val_mIoU")[-1][1:5]) == max_mIoU][0]
        max_epoch = max([int(f.split("epoch")[-1].split("-")[0][1:]) for f in files])
        max_epoch_file = [f for f in files if int(f.split("epoch")[-1].split("-")[0][1:]) == max_epoch][0]

        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=wandb_logger if args.use_wandb else TensorBoardLogger("./tb_logs"),
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
        # define Module based on methods
        module = {
            "St++": STPlusPlusModule,
            "CCT": CCTModule,
            "Supervised": SupervisedModule,
            "MeanTeacher": MeanTeacherModule,
            "FixMatch": FixmatchModule,
            "St++CCT": STPlusPlusCCTModule,
            # "Bolt": BoltModule,
        }[args.method]
        model = module(args)
        # trainer.test(datamodule=dataModule, model=model, ckpt_path=os.path.join(f"{args.save_path}", max_epoch_file))
        trainer.test(datamodule=dataModule, model=model, ckpt_path=os.path.join(f"{args.save_path}", max_mIoU_file))


if __name__ == "__main__":
    args = base_parse_args(BaseModule)
    main(args)
