"""
https://www.liebertpub.com/doi/10.1089/zeb.2019.1754
https://osf.io/c3ut5/download


"""
import glob
import io
import os
import random
import zipfile
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import List

import click
import cv2
import numpy as np
import requests
import yaml
from sklearn.model_selection import KFold


def download_zip(url, output_path):
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(output_path)


def combine_masks(mask_folders: List[str], save_path: str) -> None:
    """This function combines multiple masks into one mask by taking the maximum value of all masks."""  #
    # check if save path exits
    Path(save_path).mkdir(parents=True, exist_ok=True)
    # load masks
    mask_files = sorted(glob.glob(os.path.join(mask_folders[0], "**", "*.png"), recursive=True))
    for mask_file in mask_files:
        mask_paths = [
            mask_file,
            mask_file.replace(mask_folders[0], mask_folders[1]),
            mask_file.replace(mask_folders[0], mask_folders[2]),
        ]
        masks = []
        for mask_path in mask_paths:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # replace value in mask, becasuse train in different encoded than test
            # atrium is 20 already
            mask[mask == 6] = 0  # 6 is unkonwn
            mask[mask == 255] = 30  # 255 is the value for heart in the test data
            mask[mask == 33] = 30  # 33 is the value for heart in the train data
            mask[mask == 19] = 10  # 19 is the value for bulbus in test and train
            masks.append(mask)
        # combine masks
        mask = np.max(masks, axis=0)
        # save mask
        cv2.imwrite(os.path.join(save_path, os.path.basename(mask_file)), mask)


def split(base_path: str):
    """This function splits the data into training and testing data

    Args:
        base_path (str): Path to the raw images:
    """

    # set basic params and load file list
    cross_val_splits = 5
    num_shuffels = 5
    splits = ["1", "1/4", "1/8", "1/30"]
    training_filelist: List[str] = []
    test_filelist: List[str] = []
    dataset = r"zebrafish"
    training_filelist = [
        f
        for f in listdir(join(base_path, "train_images", "ventral_samples"))
        if isfile(join(base_path, "train_images", "ventral_samples", f))
    ]
    training_filelist = [
        "train_images/ventral_samples/%s train_images/ventral_mask_combined/%s_mask.png" % (f, f[:-4])
        for f in training_filelist
    ]

    # the same for test filelist
    test_filelist = [
        f
        for f in listdir(join(base_path, "test_images", "ventral_samples_R0004"))
        if isfile(join(base_path, "test_images", "ventral_samples_R0004", f))
    ]
    test_filelist = [
        "test_images/ventral_samples_R0004/%s test_images/ventral_mask_combined_R0004/%s_mask.png" % (f, f[:-4])
        for f in test_filelist
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
            kf = KFold(n_splits=cross_val_splits)
            count = 0
            for train_index, val_index in kf.split(labeled):
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
    "base_path",
    type=click.Path(),
    default="/home/kit/stud/uwdus/Masterthesis/data/zebrafish",
)
def main(base_path: str):
    download_zip("https://osf.io/c3ut5/download", base_path)
    # change color code as test and train are different
    combine_masks(
        [
            os.path.join(
                base_path, "train_images/ventral_mask_atrium"
            ),  # atrium is 20 in grey scale in test and train
            os.path.join(
                base_path, "train_images/ventral_mask_bulbus"
            ),  # bulbus is 19 in grey scale in test and train
            os.path.join(base_path, "train_images/ventral_mask_heart"),
        ],  # heart is 33 in grey scale in train and 255 in test
        os.path.join(base_path, "train_images/ventral_mask_combined"),
    )
    combine_masks(
        [
            os.path.join(base_path, "test_images/ventral_mask_atrium_R0004"),  # atrium is 20 in grey scale
            os.path.join(base_path, "test_images/ventral_mask_bulbus_R0004"),  # bulbus is 19 in grey scale
            os.path.join(base_path, "test_images/ventral_mask_heart_R0004"),
        ],  # heart is 255 in grey scale
        os.path.join(base_path, "test_images/ventral_mask_combined_R0004"),
    )

    # split(base_path)


if __name__ == "__main__":
    main()
