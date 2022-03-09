
from model.base_module import BaseModule
import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import wandb

import albumentations as A
from albumentations.pytorch import ToTensorV2
from data.data_module import SemiDataModule
from copy import deepcopy

import torch.nn.functional as F
import os
import cv2
import numpy as np
import yaml
from utils import meanIOU


class STPlusPlusModule(BaseModule):
    def __init__(self, args):
        super(STPlusPlusModule, self).__init__(args)
        self.checkpoints = []
        self.id_to_reliability = []
        self.args = args
        self.mode = "label"

    def base_forward(self, x):
        h, w = x.shape[-2:]
        x = self.model(x)
        x = F.interpolate(x, (h, w), mode="bilinear", align_corners=True)
        return x

    def forward(self, x, tta=False):
        if not tta:
            return self.base_forward(x)

        else:
            h, w = x.shape[-2:]
            # scales = [0.5, 0.75, 1.0]
            # to avoid cuda out of memory
            scales = [0.5, 0.75, 1.0, 1.5, 2.0]

            final_result = None

            for scale in scales:
                cur_h, cur_w = int(h * scale), int(w * scale)
                cur_x = F.interpolate(
                    x, size=(cur_h, cur_w), mode="bilinear", align_corners=True
                )

                out = F.softmax(self.base_forward(cur_x), dim=1)
                out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
                final_result = out if final_result is None else (final_result + out)

                out = F.softmax(self.base_forward(cur_x.flip(3)), dim=1).flip(3)
                out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
                final_result += out

            return final_result

    def training_step(self, batch, batch_idx):
        img, mask, _ = batch["labeled"]   
        #combine batches
        if "pseudolabeled" in batch:
            img_pseudo, mask_pseudo, _ = batch["pseudolabeled"] 
            #torch.unsqueeze(mask, dim=-1).shape
            img = torch.cat((img, img_pseudo), 0)
            mask = torch.cat((mask, mask_pseudo), 0)
        pred = self(img)
        loss = self.criterion(pred, mask.long())
        return loss

    def training_epoch_end(self, outputs) -> None:
        if (
            (self.current_epoch + 1) in [self.args.epochs // 3, self.args.epochs * 2 // 3, self.args.epochs]
        ):
            self.checkpoints.append(deepcopy(model))


    def validation_step(self, batch, batch_idx):
        img, mask, id = batch
        pred = self(img)
        self.metric.add_batch(
            torch.argmax(pred, dim=1).cpu().numpy(), mask.cpu().numpy()
        )
        val_acc = self.metric.evaluate()[-1]
        self.log("mIOU", val_acc)
        return {"val_acc": val_acc}

    def validation_epoch_end(self, outputs):
        val_acc = outputs[-1]["val_acc"]
        log = {"mean mIOU": val_acc * 100}
        mIOU = val_acc * 100.0
        if mIOU > self.previous_best:
            if self.previous_best != 0:
                os.remove(
                    os.path.join(
                        self.args.save_path,
                        "%s_%s_mIOU%.2f.pth"
                        % (self.args.model, self.backbone_name, self.previous_best),
                    )
                )
            self.previous_best = mIOU
            torch.save(
                self.state_dict(),
                os.path.join(
                    self.args.save_path,
                    "%s_%s_mIOU%.2f.pth" % (self.args.model, self.backbone_name, mIOU),
                ),
            )
        return {"log": log, "val_acc": val_acc}


    def predict_step(self, batch, batch_idx: int):
        img, mask, id = batch 
        if self.mode == "label":
            pred = self(img)
            pred = torch.argmax(pred, dim=1).cpu()
            pred = (pred * 255).squeeze(0).numpy().astype(np.uint8)
            cv2.imwrite(
                "%s/%s"% (self.args.pseudo_mask_path, os.path.basename(id[0].split(" ")[1])),
                pred
                )
            return [pred, mask, id]
        if self.mode == "select_reliable":
            preds = []
            for model in self.checkpoints:
                preds.append(torch.argmax(model(img), dim=1).cpu().numpy())
            mIOU = []
            for i in range(len(preds) - 1):
                metric = meanIOU(args.n_class)
                metric.add_batch(preds[i], preds[-1])
                mIOU.append(metric.evaluate()[-1])
            reliability = sum(mIOU) / len(mIOU)
            self.id_to_reliability.append((id[0], reliability))

    
    def on_predict_epoch_end(self, results) -> None:
        if self.mode == "select_reliable":
            labeled_ids = []
            with open(args.split_file_path, "r") as file:
                split_dict = yaml.load(file, Loader=yaml.FullLoader)
                labeled_ids = split_dict[args.val_split]["labeled"]
                val_ids = labeled_ids = split_dict[args.val_split]["val"]

            yaml_dict = dict()
            yaml_dict[args.val_split] = dict(
                val=val_ids,
                labeled=labeled_ids,
                unlabeled=[i[0] for i in self.id_to_reliability[: len(self.id_to_reliability) // 2]], #reliable ids
            )
            # save to yaml
            if not os.path.exists(self.args.reliable_id_path):
                os.makedirs(self.args.reliable_id_path)

            with open(
                os.path.join(self.args.reliable_id_path, "reliable_ids.yaml"), "w+"
            ) as outfile:
                yaml.dump(yaml_dict, outfile, default_flow_style=False)

def parse_args():
    parser = argparse.ArgumentParser(description="ST and ST++ Framework")

    # basic settings
    parser.add_argument(
        "--data-root",
        type=str,
        default="/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/BreastCancer",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["melanoma", "pneumothorax", "breastCancer"],
        default="breastCancer",
    )

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--base-size",type=int, default=None)

    parser.add_argument(
        "--val-split", type=str, default="val_split_0"
    ) 

    # semi-supervised settings
    parser.add_argument("--split", type=str, default="1_30")
    parser.add_argument("--shuffle", type=int, default=0)
    # these are derived from the above split, shuffle and dataset. They dont need to be set
    parser.add_argument(
        "--split-file-path", type=str, default=None
    )  # "dataset/splits/melanoma/1_30/split_0/split_sample.yaml")
    parser.add_argument(
        "--test-file-path", type=str, default=None
    )  # "dataset/splits/melanoma/test_sample.yaml")
    parser.add_argument("--pseudo-mask-path", type=str, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--reliable-id-path", type=str, default=None)
    parser.add_argument(
        "--plus",
        dest="plus",
        default=True,
        help="whether to use ST++",
    )
    parser.add_argument(
        "--use-wandb", default=False, help="whether to use WandB for logging"
    )
    parser.add_argument(
        "--use-tta", default=True, help="whether to use Test Time Augmentation"
    )

    args = parser.parse_args()

    # add model specific args
    parser = STPlusPlusModule.add_model_specific_args(parser)

    # add all the availabele trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.epochs is None:
        args.epochs = {"melanoma": 80}[args.dataset]
    if args.crop_size is None:
        args.crop_size = {
            "melanoma": 256,
            "breastCancer": 256,
        }[args.dataset]
    if args.base_size is None:
        args.base_size = {
            "melanoma": 256,
            "breastCancer": 500,
        }[args.dataset]     
    if args.n_class is None:
        args.n_class ={"melanoma": 2, "breastCancer": 3}[args.dataset]
    if args.split_file_path is None:
        args.split_file_path = f"dataset/splits/{args.dataset}/{args.split}/split_{args.shuffle}/split.yaml"
    if args.test_file_path is None:
        args.test_file_path = f"dataset/splits/{args.dataset}/test.yaml"
    if args.pseudo_mask_path is None:
        args.pseudo_mask_path = (
            f"outdir/pseudo_masks/{args.dataset}/{args.split}/split_{args.shuffle}"
        )
    if args.save_path is None:
        args.save_path = (
            f"outdir/models/{args.dataset}/{args.split}/split_{args.shuffle}"
        )
    if args.reliable_id_path is None:
        args.reliable_id_path = (
            f"outdir/reliable_ids/{args.dataset}/{args.split}/split_{args.shuffle}"
        )

    if args.use_wandb:
        wandb.init(project="ST++", entity="gkeppler")
        wandb_logger = WandbLogger(project="ST++")
        wandb.define_metric("Pictures")
        wandb.define_metric("loss")
        wandb.define_metric("mIOU")
        wandb.config.update(args)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.pseudo_mask_path):
        os.makedirs(args.pseudo_mask_path)
    if args.plus and args.reliable_id_path is None:
        exit("Please specify reliable-id-path in ST++.")

    seed_everything(123, workers=True)

    # Declare an augmentation pipelines
    a_train_transforms_labeled = A.Compose([ 
        A.LongestMaxSize(args.base_size),
        A.RandomScale(scale_limit=[0,5,2],p=1), 
        A.RandomCrop(args.crop_size, args.crop_size),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2() 
    ])

    a_val_transforms = A.Compose([
        A.LongestMaxSize(args.base_size),
        A.Normalize(),
        ToTensorV2() 
    ])

    a_train_transforms_unlabeled = A.Compose([
        A.LongestMaxSize(args.base_size),
        A.RandomScale(scale_limit=[0,5,2],p=1), 
        A.RandomCrop(args.crop_size, args.crop_size),
        A.HorizontalFlip(p=0.5),
        A.GaussianBlur(p=0.5), 
        A.ColorJitter(p=0.8),
        A.CoarseDropout(), #cutout
        A.Normalize( #imagenet normalize
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2() 
    ])    

    dataModule = SemiDataModule(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        split_yaml_path=args.split_file_path,
        test_yaml_path=args.test_file_path,
        pseudo_mask_path=args.pseudo_mask_path,
        test_transforms = a_val_transforms,
        val_transforms = a_val_transforms,
        train_transforms = a_train_transforms_labeled,
        train_transforms_unlabeled = a_train_transforms_unlabeled,
        mode = "train"
    )

    model = STPlusPlusModule(args)

    # saves a file like: my/path/sample-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("./", f"{args.save_path}"),
        filename=f"{args.model}" + "-{epoch:02d}-{val_acc:.2f}",
        mode="max",
        save_weights_only=True,
    )

    dev_run = False  # not working when predicting with best_model checkpoint
    trainer = pl.Trainer.from_argparse_args(
        args,
        fast_dev_run=dev_run,
        max_epochs=args.epochs,
        log_every_n_steps=2,
        logger=wandb_logger if args.use_wandb else TensorBoardLogger("./tb_logs"),
        callbacks=[checkpoint_callback],
        #gpus=[0],
        accelerator="cpu",
        #profiler="pytorch"
    )
    # <====================== Supervised training with labeled images (SupOnly) ======================>

    trainer.fit(model=model, datamodule=dataModule)

    # <====================== Test supervised model on testset (SupOnly) ======================>
    #TODO
    trainer.test(datamodule=dataModule, ckpt_path = checkpoint_callback.best_model_path)    


    if not args.plus:
        # <============================= Pseudolabel all unlabeled images =============================>


        trainer.predict(
            datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path
        )

        # <======================== Re-training on labeled and unlabeled images ========================>

        model = STPlusPlusModule(args)

        # increase max epochs to double the amount
        trainer.fit_loop.max_epochs =  args.epochs*2
        dataModule.mode = "pseudo_train"
        trainer.fit(
            model=model, datamodule=dataModule
        )
    else:
        # <===================================== Select Reliable IDs =====================================>
        model.mode = "select_reliable"
        trainer.predict(
            datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path
        )
        # <================================ Pseudo label reliable images =================================>
        dataModule.split_yaml_path = os.path.join(args.reliable_id_path, "reliable_ids.yaml")
        dataModule.init_datasets()
        model.mode = "label"
        trainer.predict(
            datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path
        )
        # <================================== The 1st stage re-training ==================================>
        
        model = STPlusPlusModule(args)

        # increase max epochs to double the amount
        trainer.fit_loop.max_epochs = args.epochs*3
        dataModule.mode = "pseudo_train"
        trainer.fit(
            model=model, datamodule=dataModule
        )
        # <=============================== Pseudo label all images ================================>

        dataModule.split_yaml_path = args.split_file_path
        dataModule.init_datasets()
        model.mode = "label"
        trainer.predict(
            datamodule=dataModule, ckpt_path=checkpoint_callback.best_model_path
        )

        # <================================== The 2nd stage re-training ==================================>
        model = STPlusPlusModule(args)

        # increase max epochs to double the amount
        trainer.fit_loop.max_epochs *= 2
        trainer.fit(
            model=model, datamodule=dataModule
        )