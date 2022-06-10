import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Module
from torch.nn import functional as F


class CELoss(Module):
    def __init__(self, n_class):
        super(CELoss, self).__init__()
        self.n_class = n_class
        if n_class == 2:
            self.criterion = BCEWithLogitsLoss()
        else:
            self.criterion = CrossEntropyLoss()

    def forward(self, pred, mask):
        if self.n_class == 2:
            # BCEloss
            mask = mask.float()
            pred = pred.squeeze(dim=1)
        else:
            # CEloss
            mask = mask.long()
        return self.criterion(pred, mask)


# https://stackoverflow.com/questions/63735255/how-do-i-compute-bootstrapped-cross-entropy-loss-in-pytorch
# big increase for CCT
class BootstrappedCE(Module):
    def __init__(self, start_warm=20000, end_warm=70000, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            return F.cross_entropy(input, target), 1.0

        raw_loss = F.cross_entropy(input, target, reduction="none").view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1 - self.top_p) * ((self.end_warm - it) / (self.end_warm - self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p


class DiceLoss(Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def forward(self, logits, true, eps=1e-7):
        """Computes the Sørensen–Dice loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the dice loss so we
        return the negated dice loss.
        Args:
            true: a tensor of shape [B, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            dice_loss: the Sørensen–Dice loss.
        """
        if self.n_classes == 2:
            true_1_hot = torch.eye(self.n_classes + 1)[true]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(self.n_classes)[true]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.unsqueeze(1).ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2.0 * intersection / (cardinality + eps)).mean()
        return 1 - dice_loss
