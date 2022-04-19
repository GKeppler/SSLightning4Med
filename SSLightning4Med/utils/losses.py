import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Module


class CE_loss(Module):
    def __init__(self, n_class):
        super(CE_loss, self).__init__()
        self.n_class = n_class
        if n_class == 2:
            self.criterion = BCEWithLogitsLoss()
        else:
            self.criterion = CrossEntropyLoss()

    def forward(self, pred, mask):
        if self.n_class == 2:
            # BCEloss
            mask = mask.float()  # .squeeze()
            pred = pred.squeeze(dim=1)
        else:
            # CEloss
            mask = mask.long()
        return self.criterion(pred, mask)


class DiceLoss(Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
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
