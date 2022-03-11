import os
from argparse import ArgumentParser
from typing import Any, Tuple

import cv2
import numpy as np
import pytorch_lightning as pl
import wandb
from medpy import metric
from numpy import float64, ndarray
from torch import Tensor
from wandb.sdk.data_types import Image

EPS = 1e-10


class meanIOU:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred: ndarray, label_true: ndarray) -> ndarray:
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions: ndarray, gts: ndarray) -> None:
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self) -> Tuple[ndarray, float64]:
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist) + EPS)
        return iu, np.nanmean(iu)


class mulitmetrics:
    # from https://github.com/kevinzakka/pytorch-goodies/blob/c039691f349be9f21527bb38b907a940bfc5e8f3/metrics.py
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred: ndarray, label_true: ndarray) -> ndarray:
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions: ndarray, gts: ndarray) -> None:
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self) -> Tuple[float64, float64, float64]:
        A_inter_B = np.diag(self.hist)
        A = self.hist.sum(axis=1)
        B = self.hist.sum(axis=0)
        # jaccard_index
        iu = A_inter_B / (A + B - A_inter_B + EPS)
        meanIOU = np.nanmean(iu)

        # dice_coefficient
        dice = (2 * A_inter_B) / (A + B + EPS)
        avg_dice = np.nanmean(dice)

        # overall_pixel_accuracy
        correct = A_inter_B.sum()
        total = self.hist.sum()
        overall_acc = correct / (total + EPS)

        return overall_acc, meanIOU, avg_dice


def calculate_metric_percase(pred: ndarray, gt: ndarray) -> Tuple[float64, float64, float64, float64]:
    dc = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dc, jc, hd, asd


def wandb_image_mask(img: Tensor, mask: Tensor, pred: Tensor, nclass: int = 21) -> Image:
    class_labeles = dict((el, "something") for el in list(range(nclass)))
    class_labeles.update({0: "nothing"})
    return wandb.Image(
        cv2.resize(
            np.moveaxis(np.squeeze(img.cpu().numpy(), axis=0), 0, -1),
            dsize=(100, 100),
            interpolation=cv2.INTER_NEAREST,
        ),
        masks={
            "predictions": {
                "mask_data": cv2.resize(
                    np.squeeze(pred.cpu().numpy(), axis=0),
                    dsize=(100, 100),
                    interpolation=cv2.INTER_NEAREST,
                ),
                "class_labels": class_labeles,
            },
            "ground_truth": {
                "mask_data": cv2.resize(
                    np.squeeze(mask.numpy(), axis=0),
                    dsize=(100, 100),
                    interpolation=cv2.INTER_NEAREST,
                ),
                "class_labels": class_labeles,
            },
        },
    )


def sigmoid_rampup(current: int, rampup_length: int = 200) -> float:
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def base_parse_args(LightningModule) -> Any:  # type: ignore
    parser = ArgumentParser()
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
    parser.add_argument("--base-size", type=int, default=None)

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
    # add model specific args
    parser = LightningModule.add_model_specific_args(parent_parser=parser)
    # add all the availabele trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

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
        args.n_class = {"melanoma": 2, "breastCancer": 3}[args.dataset]
    if args.split_file_path is None:
        args.split_file_path = f"dataset/splits/{args.dataset}/{args.split}/split_{args.shuffle}/split.yaml"
    if args.test_file_path is None:
        args.test_file_path = f"dataset/splits/{args.dataset}/test.yaml"
    if args.pseudo_mask_path is None:
        args.pseudo_mask_path = f"outdir/{args.method}/pseudo_masks/{args.dataset}/{args.split}/split_{args.shuffle}"
    if args.save_path is None:
        args.save_path = f"outdir/{args.method}/models/{args.dataset}/{args.split}/split_{args.shuffle}"
    if args.reliable_id_path is None:
        args.reliable_id_path = f"outdir/{args.method}/reliable_ids/{args.dataset}/{args.split}/split_{args.shuffle}"

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.pseudo_mask_path):
        os.makedirs(args.pseudo_mask_path)

    return args
