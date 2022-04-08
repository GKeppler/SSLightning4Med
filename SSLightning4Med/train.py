import os

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import wandb
from SSLightning4Med.models.base_model import BaseModule
from SSLightning4Med.models.data_module import SemiDataModule
from SSLightning4Med.models.train_bolt import BoltModule
from SSLightning4Med.models.train_CCT import CCTModule
from SSLightning4Med.models.train_stplusplus import STPlusPlusModule
from SSLightning4Med.models.train_supervised import SupervisedModule
from SSLightning4Med.utils.augmentations import Augmentations
from SSLightning4Med.utils.utils import base_parse_args, get_color_map

if __name__ == "__main__":
    args = base_parse_args(BaseModule)
    seed_everything(123, workers=True)

    # Define DataModule with Augmentations
    augs = Augmentations(args)
    color_map = get_color_map(args.dataset)
    dataModule = SemiDataModule(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        split_yaml_path=args.split_file_path,
        test_yaml_path=args.test_file_path,
        pseudo_mask_path=args.pseudo_mask_path,
        mode="train",
        color_map=color_map,
        num_workers=args.n_workers,
    )

    dataModule.val_transforms = augs.a_val_transforms()
    dataModule.train_transforms = augs.a_train_transforms_labeled()
    dataModule.train_transforms_unlabeled = augs.a_train_transforms_unlabeled()

    # saves a file like: my/path/sample-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("./", f"{args.save_path}"),
        filename=f"{args.net}" + "-{epoch:02d}-{val_mIoU:.2f}",
        mode="max",
        save_weights_only=True,
    )
    if args.use_wandb:
        # https://pytorch-lightning.readthedocs.io/en/1.5.0/extensions/generated/pytorch_lightning.loggers.WandbLogger.html
        wandb.init(project="SSLightning4Med", entity="gkeppler")
        wandb_logger = WandbLogger(project="SSLightning4Med")
        wandb.config.update(args)

    # define Trainer
    dev_run = False  # not working when predicting with best_model checkpoint
    trainer = pl.Trainer.from_argparse_args(
        args,
        fast_dev_run=dev_run,
        max_epochs=args.epochs,
        log_every_n_steps=2,
        logger=wandb_logger if args.use_wandb else TensorBoardLogger("./tb_logs"),
        callbacks=[checkpoint_callback],
        gpus=[0],
        # precision=16,
        # accelerator="cpu",
        # profiler="pytorch",
        # auto_lr_find=True,
        # track_grad_norm=True,
    )
    # define Module based on methods
    module = {
        "St++": STPlusPlusModule,
        "CCT": CCTModule,
        "Supervised": SupervisedModule,
        "Bolt": BoltModule,
    }[args.method]

    module.pipeline(dataModule, trainer, checkpoint_callback, args)
