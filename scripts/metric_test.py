import numpy as np
from medpy import metric
EPS = 1e-10

'''
calucates metrics based on confusion matrix
'''
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
        # array with iou of different classes. e.g 0 for class 0, 0.5 for class 1 -> mIoU = 0.25
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

def calculate_metric_percase(pred, gt):
    dc = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dc, jc, hd, asd

ownmetric = mulitmetrics(num_classes=2)

gt = np.array([1,1,1])
p = np.array([1,0,0])

ownmetric.add_batch(predictions = p, gts = gt)

print(ownmetric.evaluate())
print(calculate_metric_percase(p,gt))
