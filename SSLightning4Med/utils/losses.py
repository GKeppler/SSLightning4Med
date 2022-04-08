from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss


class CE_loss(nn.Module):
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
