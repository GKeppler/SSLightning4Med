# the goal is to implement the the cct approa

'''
this is the implementation of the CCT SSL approach
- custom unet with aux decoders 1-3
'''
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import SGD
from model.unet import UNet_CCT
from torch.nn import CrossEntropyLoss
from data.data_module import SemiDataModule
import numpy as np
from utils import mulitmetrics

def sigmoid_rampup(current, rampup_length = 200):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

class CCTModule(pl.LightningModule):
    def __init__(self, args):
        super(CCTModule, self).__init__()
        self.ce_loss = CrossEntropyLoss()
        self.model = UNet_CCT(3,args.n_class)
        self.consistency = 0.1
        self.test_metrics = mulitmetrics(num_classes=args.n_class)
        
    def training_step(self, batch):
        batch_unusupervised = batch["unlabeled"]
        batch_supervised = batch["labeled"]

        #calculate all outputs based on the different decoders
        image_batch, label_batch, _ = batch_supervised

        outputs, outputs_aux1, outputs_aux2, outputs_aux3 = self.model(
            image_batch)
        outputs_soft = torch.softmax(outputs, dim=1)
        outputs_aux1_soft = torch.softmax(outputs_aux1, dim=1)
        outputs_aux2_soft = torch.softmax(outputs_aux2, dim=1)
        outputs_aux3_soft = torch.softmax(outputs_aux3, dim=1)
        #calc losses for labeled batch
        loss_ce = self.ce_loss(outputs,
                            label_batch)
        loss_ce_aux1 = self.ce_loss(outputs_aux1,
                                label_batch)
        loss_ce_aux2 = self.ce_loss(outputs_aux2,
                                label_batch)
        loss_ce_aux3 = self.ce_loss(outputs_aux3,
                                label_batch)

        supervised_loss = (loss_ce + loss_ce_aux1 + loss_ce_aux2 + loss_ce_aux3) / 4

        #calucalate unsupervised loss
        image_batch, _, _ = batch_unusupervised
        outputs, outputs_aux1, outputs_aux2, outputs_aux3 = self.model(
            image_batch)
        # warmup unsup loss to avoid inital noise
        consistency_weight = self.consistency * sigmoid_rampup(self.current_epoch)

        consistency_loss_aux1 = torch.mean(
            (outputs_soft - outputs_aux1_soft) ** 2)
        consistency_loss_aux2 = torch.mean(
            (outputs_soft - outputs_aux2_soft) ** 2)
        consistency_loss_aux3 = torch.mean(
            (outputs_soft - outputs_aux3_soft) ** 2)

        consistency_loss = (consistency_loss_aux1 + consistency_loss_aux2 + consistency_loss_aux3) / 3
        loss = supervised_loss + consistency_loss * consistency_weight 
        return loss
        
    def test_step(self, batch, batch_idx):
        img, mask, id = batch
        pred = self(img)
        pred = torch.argmax(pred, dim=1).cpu()
        self.test_metrics.add_batch(pred.numpy(), mask.cpu().numpy())

    def test_epoch_end(self, outputs) -> None:
        overall_acc, mIOU, mDICE = self.test_metric.evaluate()
        self.log("accuracy", overall_acc)
        self.log("mIOU", mIOU)
        self.log("mDICE", mDICE)
    
    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4 
        )
        return [optimizer]



if __name__ == "__main__":
    seed_everything(123, workers=True)
    trainer = pl.Trainer(
        logger= TensorBoardLogger("./tb_logs"),
        gpus=[0],
    )
    Datamodule = SemiDataModule(
            root_dir="/home/gustav/datasets/ISIC_Demo_2017_small",
            split_yaml_path = "dataset/splits/melanoma/laptop_test/train_laptop.yaml",
            batch_size = 4,
            mode="semi_train"
            )
    
    cct = CCTModule(args)
    trainer.fit(model = cct, datamodule = Datamodule)