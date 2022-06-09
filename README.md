# SSLightning4Med


This is the official  [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) implementation of my thesis about semi supervised learning in medical image segmentation.

## Installation

### Create Conda Environment

The instructions assumes that you have Miniconda installed on your machine.

'''
conda env create -f environment-gpu.yaml
'''

### Alternative: pip install
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



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so SSLightning4Med can be imported
    ├── SSLightning4Med                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes SSLightning4Med a Python module
    │   │
    │   ├── data           <- Scripts to download data and generate the split files
    │   │   ├── splits
    │   │   │       ├── [Dataset]
    │   │   │           ├── 1
    │   │   │               ├── split_0
    │   │   │                   └── split.yaml
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── .pre-commit-config.yaml            <- pre commit file for a clean repository


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Augmentations => [Albumentations](https://albumentations.ai/)

## Acknowledgement
Code is partly from [SSL4MIS](https://github.com/HiLab-git/SSL4MIS/) and  [ST++](https://github.com/LiheYoung/ST-PlusPlus).
Thanks a lot for their great works!
