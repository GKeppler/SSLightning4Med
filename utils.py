import numpy as np
import cv2
import wandb

EPS = 1e-10


class meanIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes**2,
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        iu = np.diag(self.hist) / (
            self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist) + EPS
        )
        return iu, np.nanmean(iu)


class mulitmetrics:
    # from https://github.com/kevinzakka/pytorch-goodies/blob/c039691f349be9f21527bb38b907a940bfc5e8f3/metrics.py
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes**2,
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
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

def wandb_image_mask(img,mask,pred,nclass=21):
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