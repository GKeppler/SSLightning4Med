#!/bin/bash

git clone https://github.com/GKeppler/SSLightning4Med.git

cd SSLightning4Med
conda env create -f environment-gpu.yaml

conda activate sslightning4med

export DATASET_PATH="/home/data/raw/"

python SSLightning4Med/data/zebrafish_data_processing.py $DATASET_PATH/zebrafish/
python SSLightning4Med/data/melanoma_preprocessing.py $DATASET_PATH/melanoma/
python SSLightning4Med/data/pneumothorax_preprocessing.py $DATASET_PATH/pneumothorax/
python SSLightning4Med/data/multi_organ_data_processing.py $DATASET_PATH/multiorgan/
python SSLightning4Med/data/breast_cancer_preprocessing.py $DATASET_PATH/breastCancer/
python SSLightning4Med/data/splits/hippocampus_splits.py $DATASET_PATH/hippocampus/
