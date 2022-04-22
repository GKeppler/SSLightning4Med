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


class DiceLoss2(Module):
    def __init__(self, n_classes):
        super(DiceLoss2, self).__init__()
        self.n_classes = n_classes if n_classes > 2 else 1
        self.softmax = True if n_classes > 2 else False

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None):
        target = target.unsqueeze(1)
        if self.softmax:
            inputs = torch.softmax(inputs, dim=1)
        else:
            inputs = inputs.sigmoid()

        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), "predict & target shape do not match"
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
