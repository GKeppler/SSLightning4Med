# SSLightning4Med


This is the official  [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) implementation of my thesis about semi supervised learning in medical image segmentation.

## Installation

### Create Conda Environment

The instructions assumes that you have Miniconda installed on your machine.

```
conda env create -f environment-gpu.yaml
```

[comment]: <> (### Alternative: pip install
TODO
`pip install -U sslightning4med`)   

## Data Preparation

### Download datasets
Sign-up for Kaggle and Synapse and put credentials into the .env file in the main folder
run
```
make.sh
```
to download and preprocess the datasets

### or

[Ultrasound detection of breast nodules](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) | [SIIM 2017 ISIC Melanoma Segmentation](https://challenge.isic-archive.com/data/) | [SIIM-ACR Pneumothorax Segmentation](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation) etc.

Download dataset and split from exmaple into following folder structure.
```
├── [Your Dataset Path]
    ├── train
        ├── images
        └── labels
    └── test
        ├── images
        └── labels
```

### Datasplits provided for the resulting structure of the make.sh run

```
├── splits
    ├── [Dataset]
        ├── 1
            ├── split_0
                └── split.yaml
            ├── split_1
            ├── split_2
            ├── split_3
            └── split_4
        ├── 1_4
        ├── 1_8
        └── 1_30
    └── [Dataset]
```
example split.yaml file:

```
val_split_0:
  labeled:
  - train/images/example_image1.png train/labels/example_mask1.png
  - train/images/example_image2.png train/labels/example_mask2.png
  unlabled:
  - train/images/example_image3.png train/labels/example_mask3.png
  - train/images/example_image4.png train/labels/example_mask4.png
  val:
  - train/images/example_image5.png train/labels/example_mask5.png
  - train/images/example_image6.png train/labels/example_mask6.png
```

## train
example
```
SSLightning4Med/train.py --dataset=melanoma --epochs=200 --loss=Dice --method=CCT --net=Unet --shuffle=0 --split=1_8 --use-wandb=True --wandb-project=SSLightning4Med
```
## test
example
```
SSLightning4Med/test.py --dataset=melanoma --method=CCT --net=Unet --shuffle=0 --split=1_8 --use-wandb=True --wandb-project=SSLightning4Med
```

Project Organization
------------
    ├── environment-gpu.yaml <- install env. for GPU
    ├── make.sh            <- download datasets
    ├── pyproject.toml     <- commit settings
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so SSLightning4Med can be imported
    ├── setup.cfg          <- commit setting
    ├── SSLightning4Med                <- Source code for use in this project.
    │       │   │
    │   ├── data           <- Scripts to download data and generate the split files
    │   │   ├── splits
    │   │   │       ├── [Dataset]
    │   │   │           ├── 1
    │   │   │               ├── split_0
    │   │   │                   └── split.yaml
    │   │   └── dataset_scripts.py <- download and preprocess datasets
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── dataset_statistics.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── base_module.py  <- base module the methods inherit 
    │   │   ├── dataset.py      <- torch dataset
    │   │   ├── data_module.py  <- torch data module
    │   │   ├── data_module_St.py  <- torch data module for St++           
    │   │   └── train_$module$.py <- the methods lightning train moduls  
    │   ├── nets         <- the ANNs               
    │   │   └──  $net$.py  <- U-Net, DeepLabV3+, etc.
    │   ├── utils         <- Utility scripts
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── wandb_download_and_plot.py <- download results from wandb and plot them
    │   ├── test.py   <- Scripts to test the trained models   
    │   └── train.py  <- Scripts to train the models  
    │
    └── .pre-commit-config.yaml            <- pre commit file for a clean repository


--------

## Augmentations => [Albumentations](https://albumentations.ai/)

## Acknowledgement
Code is partly from [ST++](https://github.com/LiheYoung/ST-PlusPlus), [CCT](https://github.com/yassouali/CCT), and [SSL4MIS](https://github.com/HiLab-git/SSL4MIS/).
Thanks a lot for their great works!
