from typing import Tuple

import cv2
import numpy as np
from medpy import metric
from numpy import ndarray
from torch import Tensor, argmax, tensor
from wandb.sdk.data_types import Image

import wandb

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

    def evaluate(self) -> Tuple[ndarray, float]:
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist) + EPS)
        return iu, np.nanmean(iu)


class mulitmetrics:
    # from https://github.com/kevinzakka/pytorch-goodies/blob/c039691f349be9f21527bb38b907a940bfc5e8f3/metrics.py
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
        self.medpy_dc_list = []
        self.medpy_jc_list = []
        self.medpy_hd_list = []
        self.medpy_asd_list = []

    def calculate_metric_percase(self, pred: ndarray, gt: ndarray) -> Tuple[float, float, float, float]:
        if 1 in pred and 1 in gt:
            dc = metric.binary.dc(pred, gt)
            jc = metric.binary.jc(pred, gt)
            hd = metric.binary.hd95(
                pred,
                gt,
            )
            asd = metric.binary.asd(pred, gt)
        else:
            dc = np.NaN
            jc = np.NaN
            hd = np.NaN
            asd = np.NaN
        return dc, jc, hd, asd

    def _fast_hist(self, label_pred: ndarray, label_true: ndarray) -> ndarray:
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions: ndarray, gts: ndarray) -> None:
        for i in range(1, self.num_classes):
            dc, jc, hd, asd = self.calculate_metric_percase((predictions == i).astype(int), (gts == i).astype(int))
            self.medpy_dc_list.append(dc)
            self.medpy_jc_list.append(jc)
            self.medpy_hd_list.append(hd)
            self.medpy_asd_list.append(asd)
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self) -> Tuple[float, float, float]:
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

        # medpy metrics
        medpy_dc = np.nanmean(self.medpy_dc_list)
        medpy_jc = np.nanmean(self.medpy_jc_list)
        medpy_hd = np.nanmean(self.medpy_hd_list)
        medpy_asd = np.nanmean(self.medpy_asd_list)

        return overall_acc, meanIOU, avg_dice, medpy_dc, medpy_jc, medpy_hd, medpy_asd


class getOneHot:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.threshold = tensor([0.5])

    def __call__(self, pred: Tensor) -> Tensor:
        if self.num_classes == 2:
            pred = (pred.sigmoid().cpu() > self.threshold).float() * 1
        else:
            pred = argmax(pred, dim=1)
        return pred


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
                    np.squeeze(mask.cpu().numpy(), axis=0),
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


def get_color_map(dataset):
    return {
        "melanoma": [
            [0, 0, 0],
            [255, 255, 255],
        ],
        "breastCancer": [
            [0, 0, 0],
            [0, 0, 255],
            [0, 255, 0],
        ],
        "pneumothorax": [
            [0, 0, 0],
            [255, 255, 255],
        ],
        "multiorgan": [[s * 10, s * 10, s * 10] for s in range(0, 14)],
    }[dataset]
