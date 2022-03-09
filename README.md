# SSLightning4Med

This is the official [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) implementation of my thesis.

## Installation

TODO
`pip install -U sslightning4med`

## Data Preparation

### Download datasets
[Ultrasound detection of breast nodules](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) | [SIIM 2017 ISIC Melanoma Segmentation](https://challenge.isic-archive.com/data/) | 
[Automated Cardiac Diagnosis Challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/databasesTesting.html) | [SIIM-ACR Pneumothorax Segmentation](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation)

Download dataset and split fro exmaple into following folder structure.
```
├── [Your Dataset Path]
    ├── train
        ├── images
        └── labels
    └── test
        ├── images
        └── labels
```

### Run split scripts in /scripts
The provided scripts generate files with definded splits

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

## Augmentations => [Albumentations](https://albumentations.ai/)

## Acknowledgement
Code is partly from [SSL4MIS](https://github.com/HiLab-git/SSL4MIS/) and  [ST++](https://github.com/LiheYoung/ST-PlusPlus).
Thanks a lot for their great works!
