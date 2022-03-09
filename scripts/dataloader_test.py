import pytorch_lightning as pl
from dataset.data_module import SemiDataModule
from model.semseg.deeplabv3plus import DeepLabV3Plus

dataset = "melanoma"
data_root = r"/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/ISIC_Demo_2017"
batch_size = 2
crop_size = 100

dataModule = SemiDataModule(
    root_dir=data_root,
    batch_size=batch_size,
    train_yaml_path="dataset/splits/melanoma/1_8/split_0/split.yaml",
    test_yaml_path="dataset/splits/melanoma/test.yaml",
)

test = dataModule.train_dataset.__getitem__(2)
test2 = dataModule.test_dataset.__getitem__(2)
test4 = dataModule.val_dataset.__getitem__(2)

Trainer = pl.Trainer(fast_dev_run=True, accelerator="cpu")
Trainer.fit(model=DeepLabV3Plus(backbone="resnet50", nclass=2), datamodule=dataModule)
