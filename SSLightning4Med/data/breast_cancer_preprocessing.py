import glob
import os
import random
import shutil
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import List

import click
import dotenv
import kaggle
import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from utils import resize

# this script copies the images/masks from the folders benign, malignant and normal to an overall
# images and mask folder.


# stratified train test split


def train_test_split(base_path):
    src = os.path.join(base_path, "Dataset_BUSI_with_GT/")
    allFileNames = sorted(glob.glob(os.path.join(src, "**", "*).png"), recursive=True))
    # generate list of classes 0,1,2 for normal, malign, beging based on filenames
    classes = [(0 if "normal" in name else 1 if "malignant" in name else 2) for name in allFileNames]

    allFileNames = np.array(allFileNames)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    sss.get_n_splits(allFileNames, classes)
    for train_index, test_index in sss.split(allFileNames, classes):
        print("TRAIN:", train_index, "TEST:", test_index)
        train_FileNames, test_FileNames = allFileNames[train_index], allFileNames[test_index]

    train_FileNames = train_FileNames.tolist()
    test_FileNames = test_FileNames.tolist()

    print("Total images: ", len(allFileNames))
    print("Training: ", len(train_FileNames))
    print("Testing: ", len(test_FileNames))

    if not os.path.exists(os.path.join(base_path, "train", "images")):
        os.makedirs(os.path.join(base_path, "train", "images"))
        os.makedirs(os.path.join(base_path, "train", "labels"))
        os.makedirs(os.path.join(base_path, "test", "images"))
        os.makedirs(os.path.join(base_path, "test", "labels"))

    # Copy-pasting images
    for name in train_FileNames:
        new_name = (
            name.replace("Dataset_BUSI_with_GT", "train/images")
            .replace("benign/", "")
            .replace("malignant/", "")
            .replace("normal/", "")
            .replace(" ", "")
        )
        shutil.copy(name, new_name)
        name = name.replace(".png", "_mask.png")
        new_name = (
            name.replace("Dataset_BUSI_with_GT", "train/labels")
            .replace("benign/", "")
            .replace("malignant/", "")
            .replace("normal/", "")
            .replace(" ", "")
        )
        shutil.copy(name, new_name)

    for name in test_FileNames:
        new_name = (
            name.replace("Dataset_BUSI_with_GT", "test/images")
            .replace("benign/", "")
            .replace("malignant/", "")
            .replace("normal/", "")
            .replace(" ", "")
        )
        shutil.copy(name, new_name)
        name = name.replace(".png", "_mask.png")
        new_name = (
            name.replace("Dataset_BUSI_with_GT", "test/labels")
            .replace("benign/", "")
            .replace("malignant/", "")
            .replace("normal/", "")
            .replace(" ", "")
        )
        shutil.copy(name, new_name)

    # delete old folder with files
    shutil.rmtree(os.path.join(base_path, "Dataset_BUSI_with_GT"))


# stratified split
def stratified_split(base_path):
    # set basic params and load file list
    cross_val_splits = 5
    num_shuffels = 5
    splits = ["1", "1/4", "1/8", "1/30"]
    images_folder = "images"
    labels_folder = "labels"
    training_filelist: List[str] = []
    test_filelist: List[str] = []

    dataset = r"breastCancer"
    training_filelist = [
        "train/images/%s train/labels/%s_mask.png" % (f, f[:-4])
        for f in listdir(join(base_path, "train", images_folder))
        if isfile(join(base_path, "train", images_folder, f))
    ]
    # sanity check if file in image folder are same as in
    differences = set(
        [
            "train/images/%s.png train/labels/%s_mask.png" % (f[:-9], f[:-9])
            for f in listdir(join(base_path, "train", labels_folder))
            if isfile(join(base_path, "train", labels_folder, f))
        ]
    ).symmetric_difference(set(training_filelist))
    if len(differences) != 0:
        raise Exception(
            f"files in folders '{images_folder}' and '{labels_folder}' do not match because of: {differences}"
        )

    test_filelist = [
        "test/images/%s test/labels/%s_mask.png" % (f, f[:-4])
        for f in listdir(join(base_path, "test", images_folder))
        if isfile(join(base_path, "test", images_folder, f))
    ]

    list_len = len(training_filelist)
    print(training_filelist[:2])

    # shuffle labeled/unlabeled
    for shuffle in range(num_shuffels):
        yaml_dict = {}
        for split in splits:
            random.shuffle(training_filelist)
            # calc splitpoint
            labeled_splitpoint = int(list_len * float(eval(split)))
            print(f"splitpoint for {split} in dataset with list_len {list_len} are {labeled_splitpoint}")
            unlabeled = training_filelist[labeled_splitpoint:]
            labeled = training_filelist[:labeled_splitpoint]
            skf = StratifiedKFold(n_splits=cross_val_splits)
            y = [(0 if name[0] == "n" else 1 if name[0] == "m" else 2) for name in labeled]
            count = 0
            for train_index, val_index in skf.split(labeled, y):
                unlabeled_copy = unlabeled.copy()  # or elese it cant be reused
                train = [labeled[i] for i in train_index]
                val = [labeled[i] for i in val_index]
                yaml_dict["val_split_" + str(count)] = dict(unlabeled=unlabeled_copy, labeled=train, val=val)
                count += 1

            # save to yaml
            # e.g 1/4 -> 1_4 for folder name
            zw = list(split)
            if len(zw) > 1:
                zw[1] = "_"
            split = "".join(zw)

            yaml_path = rf"./splits/{dataset}/{split}/split_{shuffle}"
            Path(yaml_path).mkdir(parents=True, exist_ok=True)
            with open(yaml_path + "/split.yaml", "w+") as outfile:
                yaml.dump(yaml_dict, outfile, default_flow_style=False)
    # test yaml file
    yaml_dict = {}
    yaml_path = rf"./splits/{dataset}/"
    Path(yaml_path).mkdir(parents=True, exist_ok=True)

    with open(yaml_path + "/test.yaml", "w+") as outfile:
        yaml.dump(test_filelist, outfile, default_flow_style=False)


@click.command()
@click.argument(
    # "/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/MultiOrgan"
    "base_path",
    type=click.Path(),
    default="/home/gustav/datasets/breastCancer",
)  # "/home/gustav/datasets/multiorgan/"
def main(base_path: str):
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    dotenv_path = os.path.join(project_dir, ".env")
    # KAGGLE_USERNAME & KAGGLE_KEY must be set in .env file!!!
    dotenv.load_dotenv(dotenv_path)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files("aryashah2k/breast-ultrasound-images-dataset", path=base_path, unzip=True)
    train_test_split(base_path)
    resize(base_path, "breastCancer", "breastCancer256", base_size=256)
    # stratified_split(base_path)


if __name__ == "__main__":
    main()
