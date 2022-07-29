from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision
from medpy import metric
from numpy import ndarray
from torch import Tensor, argmax, tensor

import wandb
from SSLightning4Med.utils import ramps

EPS = 1e-10


class meanIOU:
    """a class for computing mean IoU"""

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
    """a class for computing several metrics including mIoU, mDSC, mHD95, mASD."""

    # from https://github.com/kevinzakka/pytorch-goodies/blob/c039691f349be9f21527bb38b907a940bfc5e8f3/metrics.py
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
        self.medpy_dc_list = []
        self.medpy_jc_list = []
        self.medpy_hd_list = []
        self.medpy_asd_list = []

    def calculate_metric_percase(self, pred: ndarray, gt: ndarray) -> Tuple[float, float, float, float]:
        if 1 in gt and 1 in pred:  # there is a ground  truth and some prediction
            dc = metric.binary.dc(pred, gt)
            jc = metric.binary.jc(pred, gt)
            hd = metric.binary.hd95(
                pred,
                gt,
            )
            asd = metric.binary.asd(pred, gt)
        elif 1 in gt:  # label not predicted: medpy error
            dc = 0
            jc = 0
            hd = np.NaN
            asd = np.NaN
        else:  # no ground truth for this class -> error
            dc = np.NaN
            jc = np.NaN
            hd = np.NaN
            asd = np.NaN
        return dc, jc, hd, asd

    def _fast_hist(self, label_pred: ndarray, label_true: ndarray) -> ndarray:
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask].astype(int),
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
    """Returns a one hot mask of the prediction."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.threshold = tensor([0.5])

    def __call__(self, pred: Tensor) -> Tensor:
        if self.num_classes == 2:
            pred = (pred.sigmoid().cpu() > self.threshold).float() * 1
            pred = pred.squeeze(dim=1)
        else:
            pred = argmax(pred, dim=1)
        return pred


def tensorboard_image_grid(pred: Tensor, color_map: ndarray) -> None:
    """Writes a grid of images to tensorboard.

    Args:
        pred (Tensor): The prediction tensor.
        color_map (ndarray): The color map to use.

    Returns: image_grid (Tensor): The image grid.
    """
    # log pred mask
    pred = pred.squeeze(1).numpy().astype(np.uint8)
    pred = np.array(color_map)[pred]
    pred = np.moveaxis(pred, -1, 1)
    return torchvision.utils.make_grid(torch.from_numpy(pred))
    # self.logger.experiment.add_image('generated_images', grid, trainer.global_step)


def wandb_image_mask(img: Tensor, mask: Tensor, pred: Tensor, nclass: int = 21, caption: str = None) -> None:
    """Writes images, masks, and predictions to wandb.

    Args:
        img (Tensor): The image tensor.
        mask (Tensor): The mask tensor.
        pred (Tensor): The prediction tensor.
        nclass (int, optional): Amount of classes. Defaults to 21.
        caption (str, optional): Caption for wandb. Defaults to None.

    Returns: wandb.Image
    """
    class_labeles = dict((el, "something") for el in list(range(nclass)))
    class_labeles.update({0: "nothing"})
    return wandb.Image(
        cv2.resize(
            np.moveaxis(np.squeeze(img.cpu().numpy(), axis=0), 0, -1),
            dsize=(200, 200),
            interpolation=cv2.INTER_NEAREST,
        ),
        masks={
            "predictions": {
                "mask_data": cv2.resize(
                    np.squeeze(pred.cpu().numpy(), axis=0),
                    dsize=(200, 200),
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
        caption=caption,
    )


# From CCT paper
class consistency_weight(object):
    """Returns a consistency weight for the consistency loss.

    Args:
    ramp_types: ['sigmoid_rampup', 'linear_rampup', 'cosine_rampup', 'log_rampup', 'exp_rampup']
    final_w: Final weight for the consistency loss.
    rampup_ends: The end of the rampup in epochs.
    """

    def __init__(self, final_w, rampup_ends=7, ramp_type="exp_rampup"):
        self.final_w = final_w
        self.rampup_ends = rampup_ends
        self.rampup_func = getattr(ramps, ramp_type)

    def __call__(self, epoch):
        self.current_rampup = self.rampup_func(epoch, self.rampup_ends)
        return self.final_w * self.current_rampup


def get_color_map(dataset):
    """Returns a color map for the given dataset.

    Args:
        dataset: The dataset to use.

    Returns: color_map (ndarray): The color map.
    """
    return {
        "melanoma": [
            [0, 0, 0],
            [255, 255, 255],
        ],
        "breastCancer": [
            [0, 0, 0],
            [255, 255, 255],
        ],
        "pneumothorax": [
            [0, 0, 0],
            [255, 255, 255],
        ],
        "multiorgan": [[s * 10, s * 10, s * 10] for s in range(0, 14)],
        "brats": [[s * 10, s * 10, s * 10] for s in range(0, 4)],
        "hippocampus": [[s * 10, s * 10, s * 10] for s in range(0, 3)],
        "zebrafish": [[s * 10, s * 10, s * 10] for s in range(0, 4)],
    }[dataset]
