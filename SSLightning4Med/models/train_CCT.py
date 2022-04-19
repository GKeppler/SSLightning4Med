from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor
from torch.optim import SGD

from SSLightning4Med.models.base_model import BaseModule
from SSLightning4Med.models.data_module import SemiDataModule
from SSLightning4Med.utils.utils import sigmoid_rampup, wandb_image_mask


class CCTModule(BaseModule):
    """
    this is the implementation of the CCT SSL approach
    - custom unet with aux decoders 1-3
    """

    def __init__(self, args: Any):
        super(CCTModule, self).__init__(args)
        self.consistency = 0.1

    def training_step(self, batch: Dict[str, Tuple[Tensor, Tensor, str]]) -> Tensor:
        batch_unusupervised = batch["unlabeled"]
        batch_supervised = batch["labeled"]

        # calculate all outputs based on the different decoders
        image_batch, label_batch, _ = batch_supervised

        outputs, outputs_aux1, outputs_aux2, outputs_aux3 = self.net(image_batch)
        outputs_soft = torch.softmax(outputs, dim=1)
        outputs_aux1_soft = torch.softmax(outputs_aux1, dim=1)
        outputs_aux2_soft = torch.softmax(outputs_aux2, dim=1)
        outputs_aux3_soft = torch.softmax(outputs_aux3, dim=1)
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
        # warmup unsup loss to avoid inital noise
        consistency_weight = self.consistency * sigmoid_rampup(self.current_epoch)

        consistency_loss_aux1 = torch.mean((outputs_soft - outputs_aux1_soft) ** 2)
        consistency_loss_aux2 = torch.mean((outputs_soft - outputs_aux2_soft) ** 2)
        consistency_loss_aux3 = torch.mean((outputs_soft - outputs_aux3_soft) ** 2)

        consistency_loss = (consistency_loss_aux1 + consistency_loss_aux2 + consistency_loss_aux3) / 3
        loss = supervised_loss + consistency_loss * consistency_weight
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):  # type: ignore
        img, mask, id = batch
        pred = self(img)[0]
        self.val_IoU(pred, mask)
        self.log("val_mIoU", self.val_IoU, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):  # type: ignore
        img, mask, id = batch
        pred = self.net(img)[0]
        pred = self.oneHot(pred.cpu())
        self.test_metrics.add_batch(pred.numpy(), mask.cpu().numpy())
        return wandb_image_mask(img, mask, pred, self.n_class)

    def configure_optimizers(self) -> List:
        optimizer = SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        return [optimizer]

    @staticmethod
    def pipeline(dataModule: SemiDataModule, trainer: pl.Trainer, checkpoint_callback: ModelCheckpoint, args) -> None:
        model = CCTModule(args)
        dataModule.mode = "semi_train"
        trainer.fit(model=model, datamodule=dataModule)
        trainer.test(datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path)


# if __name__ == "__main__":
#     args = base_parse_args(CCTModule)
#     seed_everything(123, workers=True)

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
#         precision=16,
#         # accelerator="cpu",
#         # profiler="pytorch"
#     )

#     augs = Augmentations(args)
#     color_map = get_color_map(args.dataset)
#     dataModule = SemiDataModule(
#         root_dir=args.data_root,
#         batch_size=args.batch_size,
#         split_yaml_path=args.split_file_path,
#         test_yaml_path=args.test_file_path,
#         pseudo_mask_path=args.pseudo_mask_path,
#         mode="semi_train",
#         color_map=color_map,
#     )

#     dataModule.val_transforms = augs.a_val_transforms()
#     dataModule.train_transforms = augs.a_train_transforms_labeled()
#     dataModule.train_transforms_unlabeled = augs.a_train_transforms_unlabeled()

#     CCTModule.pipeline(dataModule, trainer, args)
