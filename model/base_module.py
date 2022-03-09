
from model.unet import UNet, SmallUNet, UNet_CCT

import torch
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from utils import meanIOU, mulitmetrics, wandb_image_mask

model_zoo = {
    "unet": UNet,
    "smallUnet": SmallUNet,
    "unet_cct": UNet_CCT
}

class BaseModule(pl.LightningModule):
    def __init__(self, args):
        super(BaseModule, self).__init__()
        self.model = model_zoo[args.model](in_chns=3, class_num=args.n_class)
        self.metric = meanIOU(num_classes=args.n_class)
        self.predict_metric = meanIOU(num_classes=args.n_class)
        self.test_metrics = mulitmetrics(num_classes=args.n_class)
        self.criterion = CrossEntropyLoss()
        self.previous_best = 0.0
        self.args = args

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BaseModule")
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--model", type=str, choices=list(model_zoo.keys()), default="smallUnet") 
        parser.add_argument("--n-class",type=int, default=None)   
        return parent_parser


    def forward(self, x, tta=False):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        img, mask, id = batch
        pred = self(img)
        pred = torch.argmax(pred, dim=1).cpu()
        self.test_metrics.add_batch(pred.numpy(), mask.cpu().numpy())
        return {"Test Pictures": wandb_image_mask(img, mask, pred, self.args.n_class)}
        
    def test_epoch_end(self, outputs) -> None:
        overall_acc, mIOU, mDICE = self.test_metrics.evaluate()
        self.log("accuracy", overall_acc)
        self.log("mIOU", mIOU)
        self.log("mDICE", mDICE)

    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=1e-4  # self.lr,
        )
        # scheduler = torch.optim.ReduceLROnPlateau(optimizer, mode='min')
        return [optimizer]  # , [scheduler]
