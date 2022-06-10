import glob
import logging
import os
import sys
import zipfile
from pathlib import Path

import click
from utils import resize


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
        os.path.join(base_path, "train", "images", "ISIC-2017_Training_Data.zip"),
    )
    download_zip(
        "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part1_GroundTruth.zip",
        os.path.join(base_path, "train", "labels", "ISIC-2017_Training_Part1_GroundTruth.zip"),
    )
    download_zip(
        "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Data.zip",
        os.path.join(base_path, "test", "images", "ISIC-2017_Validation_Data.zip"),
    )
    download_zip(
        "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Part1_GroundTruth.zip",
        os.path.join(base_path, "test", "labels", "ISIC-2017_Validation_Part1_GroundTruth.zip"),
    )
    logging.info("downloading done. Rezing images")
    resize(base_path, "melanoma", "melanoma256")
    logging.info("resizing done. Splitting")
    training_filelist = sorted(glob.glob(os.path.join(base_path, "train", "images", "*.jpg"), recursive=True))
    training_filelist = [
        "train/images/%s.jpg train/labels/%s_segmentation.png" % (f[-16:-4], f[-16:-4]) for f in training_filelist
    ]

    # all iamges are in this case in the train folder
    test_filelist = sorted(glob.glob(os.path.join(base_path, "test", "images", "*.jpg"), recursive=True))
    test_filelist = [
        "test/images/%s.jpg test/labels/%s_segmentation.png" % (f[-16:-4], f[-16:-4]) for f in test_filelist
    ]
    # split("melanoma", training_filelist, test_filelist)
    logging.info("splitting done. Finished")


if __name__ == "__main__":
    main()
