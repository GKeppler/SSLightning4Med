"""
https://www.liebertpub.com/doi/10.1089/zeb.2019.1754
https://osf.io/c3ut5/download


"""
import glob
import io
import os
import zipfile
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import List

import click
import cv2
import numpy as np
import requests
from utils import resize


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


@click.command()
@click.argument(
    "base_path",
    type=click.Path(),
    default="/home/gustav/datasets/zebrafish",
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

    resize(base_path, "zebrafish", "zebrafish256", 256)

    # generate filelist for training and test
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
    # split("zebrafish", training_filelist, test_filelist)


if __name__ == "__main__":
    main()
