import glob
import logging
import os
import shutil
import sys
from os import listdir
from os.path import isfile, join
from pathlib import Path

import click
import cv2
import kaggle
import numpy as np
import pandas as pd
import pydicom
from joblib import Parallel, delayed
from PIL import Image
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm


def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    if rle == " -1" or rle == "-1":
        return mask.reshape(width, height)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position : current_position + lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)


def get_mask(encode, width, height):
    if encode == [] or encode == [" -1"]:
        return rle2mask(" -1", width, height)
    mask = rle2mask(encode[0], width, height)
    for e in encode[1:]:
        mask += rle2mask(e, width, height)
    return mask.T


def save_train_file(f, encode_df, out_path, img_size):
    img = pydicom.read_file(f).pixel_array
    name = f.split("/")[-1][:-4]
    encode = list(encode_df.loc[encode_df["ImageId"] == name, " EncodedPixels"].values)
    encode = get_mask(encode, img.shape[1], img.shape[0])
    # if mask is empty dont save
    if encode.sum() == 0:
        return
    encode = cv2.resize(encode, (img_size, img_size), Image.NEAREST)
    # encode = resize(encode,(img_size,img_size))
    img = cv2.resize(img, (img_size, img_size))

    cv2.imwrite("{}/images/{}.png".format(out_path, name), img * 255)
    cv2.imwrite("{}/labels/{}.png".format(out_path, name), encode)


def save_train(train_images_names, encode_df, out_path="../dataset128", img_size=128, n_train=-1, n_threads=1):
    if os.path.isdir(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path + "/images", exist_ok=True)
    os.makedirs(out_path + "/labels", exist_ok=True)
    if n_train < 0:
        n_train = len(train_images_names)
    try:
        Parallel(n_jobs=n_threads, backend="threading")(
            delayed(save_train_file)(f, encode_df, out_path, img_size) for f in tqdm(train_images_names[:n_train])
        )
    except pydicom.errors.InvalidDicomError:
        print("InvalidDicomError")


def train_test_split(base_path):
    src = os.path.join(base_path, "train", "images")
    allFileNames = sorted(glob.glob(os.path.join(src, "*.png"), recursive=True))
    # generate list of classes 0,1,2 for normal, malign, beging based on filenames

    allFileNames = np.array(allFileNames)

    sss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    sss.get_n_splits(allFileNames)
    for train_index, test_index in sss.split(allFileNames):
        print("TRAIN:", train_index, "TEST:", test_index)
        train_FileNames, test_FileNames = allFileNames[train_index], allFileNames[test_index]

    train_FileNames = train_FileNames.tolist()
    test_FileNames = test_FileNames.tolist()

    print("Total images: ", len(allFileNames))
    print("Training: ", len(train_FileNames))
    print("Testing: ", len(test_FileNames))

    if not os.path.exists(os.path.join(base_path, "test", "images")):
        os.makedirs(os.path.join(base_path, "test", "images"))
        os.makedirs(os.path.join(base_path, "test", "labels"))

    for name in test_FileNames:
        shutil.move(name, name.replace("train", "test"))
        shutil.move(name.replace("images", "labels"), name.replace("train", "test").replace("images", "labels"))


@click.command()
@click.argument(
    "base_path",
    type=click.Path(),
    default="/home/kit/stud/uwdus/Masterthesis/data/pneumothorax",
)
def main(base_path: str):
    Path(base_path).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(os.path.join(base_path, "download.log")), logging.StreamHandler(sys.stdout)],
    )
    logging.info("Starting download")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "jesperdramsch/siim-acr-pneumothorax-segmentation-data", path=base_path, unzip=True
    )
    logging.info("downloading done. Rezing images")
    train_fns = sorted(glob.glob("{}/*/*/*.dcm".format(os.path.join(base_path, "dicom-images-train"))))
    rle = pd.read_csv(os.path.join(base_path, "train-rle.csv"))
    save_train(train_fns, rle, os.path.join(base_path, "train"), 256)
    train_test_split(base_path)

    logging.info("resizing done. Splitting")
    training_filelist = [
        "train/images/%s train/labels/%s" % (f, f)
        for f in listdir(join(base_path, "train", "images"))
        if isfile(join(base_path, "train", "images", f))
    ]
    # sanity check if file in image folder are same as in
    differences = set(
        [
            "train/images/%s train/labels/%s" % (f, f)
            for f in listdir(join(base_path, "train", "labels"))
            if isfile(join(base_path, "train", "labels", f))
        ]
    ).symmetric_difference(set(training_filelist))
    if len(differences) != 0:
        raise Exception(f"files in folders images and labels do not match because of: {differences}")

    test_filelist = [
        "test/images/%s test/labels/%s" % (f, f)
        for f in listdir(join(base_path, "test", "images"))
        if isfile(join(base_path, "test", "images", f))
    ]

    # split("pneumothorax", training_filelist, test_filelist)
    logging.info("splitting done. Finished")


if __name__ == "__main__":
    main()
