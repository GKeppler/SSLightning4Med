from typing import Any, Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor

from SSLightning4Med.models.base_model import BaseModule
from SSLightning4Med.models.data_module import SemiDataModule


class SupervisedModule(BaseModule):
    def __init__(self, args: Any) -> None:
        super(SupervisedModule, self).__init__(args)
        self.args = args

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        return x

    def training_step(self, batch: Dict[str, Tuple[Tensor, Tensor, str]]) -> Tensor:
        img, mask, _ = batch["labeled"]
        pred = self(img)
        if self.args.n_class == 2:
            # BCEloss
            mask = mask.float().squeeze()
            pred = pred.squeeze()
        else:
            # CEloss
            mask = mask.long()
        loss = self.criterion(pred, mask)
        self.log("train_loss", loss, sync_dist=True, on_epoch=True, on_step=True)
        return {"loss": loss}

    def validation_step(self, batch: Tuple[Tensor, Tensor, str], batch_idx: int) -> Dict[str, float]:
        img, mask, id = batch
        pred = self(img)
        self.metric.add_batch(torch.argmax(pred, dim=1).cpu().numpy(), mask.cpu().numpy())
        val_acc = self.metric.evaluate()[-1]
        return {"mIOU": val_acc}

    def validation_epoch_end(self, outputs: List[Dict[str, float]]) -> Dict[str, Union[Dict[str, float], float]]:
        mIOU = outputs[-1]["mIOU"]
        self.log("val_mIoU", mIOU)
        self.set_metrics()

    @staticmethod
    def pipeline(dataModule: SemiDataModule, trainer: pl.Trainer, checkpoint_callback: ModelCheckpoint, args) -> None:
        model = SupervisedModule(args)
        trainer.fit(model=model, datamodule=dataModule)
        trainer.test(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)


# if __name__ == "__main__":

#     args = base_parse_args(SupervisedModule)
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

#     model = SupervisedModule(args)

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
#         # accelerator="cpu",
#         # profiler="pytorch",
#     )
#     SupervisedModule.pipeline(dataModule, trainer, args)
