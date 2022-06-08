import glob
import logging
import os
import random
import sys
import zipfile
from pathlib import Path
from typing import List

import click
import tqdm
import yaml
from PIL import Image
from sklearn.model_selection import KFold


def download_zip(url, filename):
    import functools
    import pathlib
    import shutil

    import requests
    from tqdm.auto import tqdm

    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get("Content-Length", 0))

    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    desc = "(Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)
    archive = zipfile.ZipFile(filename)
    for file in archive.namelist():
        archive.extract(file, os.path.dirname(filename))
    # move all files to another folder
    for file in os.listdir(filename.split(".")[0]):
        shutil.move(os.path.join(filename.split(".")[0], file), os.path.dirname(filename))
    # delte empty folder
    os.rmdir(filename.split(".")[0])


# this resizes all images in a folder to center crop


def resize(base_path: str, old_name, new_name, base_size=256):
    def resize_crop(img: Image, base_size: int) -> Image:
        w, h = img.size
        if h > w:
            crop_size = w
        else:
            crop_size = h
        left = (w - crop_size) / 2
        top = (h - crop_size) / 2
        right = (w + crop_size) / 2
        bottom = (h + crop_size) / 2
        # make it sqaure
        img = img.crop((left, top, right, bottom))

        # resize to base_size
        img = img.resize((base_size, base_size), Image.NEAREST)
        return img

    for path, subdirs, files in os.walk(base_path):
        # tqdm for loop
        for name in tqdm.tqdm(files):
            img_path = os.path.join(path, name)
            if (
                img_path.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp", ".gif"))
                and "superpixel" not in img_path
            ):
                im = Image.open(img_path)
                imResize = resize_crop(im, base_size)
                img_path_new = img_path.replace(old_name, new_name)
                if not os.path.exists(os.path.dirname(img_path_new)):
                    os.makedirs(os.path.dirname(img_path_new))
                imResize.save(img_path_new)


def split(base_path: str):
    """This function splits the data into training and testing data

    Args:
        base_path (str): Path to the raw images:
    """
    # set basic params and load file list
    dataset = r"melanoma"
    cross_val_splits = 5
    num_shuffels = 5
    splits = ["1", "1/4", "1/8", "1/30"]
    training_filelist: List[str] = []
    test_filelist: List[str] = []

    training_filelist = sorted(glob.glob(os.path.join(base_path, "train", "images", "*.jpg"), recursive=True))
    training_filelist = [
        "train/images/%s.jpg train/labels/%s_segmentation.png" % (f[-16:-4], f[-16:-4]) for f in training_filelist
    ]

    # all iamges are in this case in the train folder
    test_filelist = sorted(glob.glob(os.path.join(base_path, "test", "images", "*.jpg"), recursive=True))
    test_filelist = [
        "test/images/%s.jpg test/labels/%s_segmentation.png" % (f[-16:-4], f[-16:-4]) for f in test_filelist
    ]

    list_len = len(training_filelist)
    print(training_filelist[:2], list_len)

    # %%
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

            yaml_path = rf"data/splits/{dataset}/{split}/split_{shuffle}"
            Path(yaml_path).mkdir(parents=True, exist_ok=True)
            with open(yaml_path + "/split.yaml", "w+") as outfile:
                yaml.dump(yaml_dict, outfile, default_flow_style=False)

    # test yaml file
    yaml_dict = {}
    yaml_path = rf"data/splits/{dataset}/"
    Path(yaml_path).mkdir(parents=True, exist_ok=True)

    with open(yaml_path + "/test.yaml", "w+") as outfile:
        yaml.dump(test_filelist, outfile, default_flow_style=False)


@click.command()
@click.argument(
    "base_path",
    type=click.Path(),
    default="/home/kit/stud/uwdus/Masterthesis/data/melanoma",  # /home/kit/stud/uwdus/Masterthesis/data/melanoma",
)
def main(base_path: str):
    Path(base_path).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(os.path.join(base_path, "download.log")), logging.StreamHandler(sys.stdout)],
    )
    logging.info("Starting download")
    download_zip(
        "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip",
        os.path.join(base_path, "raw", "train", "images", "ISIC-2017_Training_Data.zip"),
    )
    download_zip(
        "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part1_GroundTruth.zip",
        os.path.join(base_path, "raw", "train", "labels", "ISIC-2017_Training_Part1_GroundTruth.zip"),
    )
    download_zip(
        "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Data.zip",
        os.path.join(base_path, "raw", "test", "images", "ISIC-2017_Validation_Data.zip"),
    )
    download_zip(
        "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Part1_GroundTruth.zip",
        os.path.join(base_path, "raw", "test", "labels", "ISIC-2017_Validation_Part1_GroundTruth.zip"),
    )
    logging.info("downloading done. Rezing images")
    resize(base_path, "raw", "")
    logging.info("resizing done. Splitting")
    # split(base_path)
    logging.info("splitting done. Finished")


if __name__ == "__main__":
    main()
