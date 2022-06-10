"""
This script downloads http://medicaldecathlon.com/
The data set consists of 195 MRI images acquired from 90 healthy adults and 105
adults with a non-affective psychotic disorder. T1-weighted MPRAGE was used as the imaging
sequence. The corresponding target ROIs were
- the anterior
- posterior of the hippocampus,
defined as the hippocampus proper and parts of the subiculum. "

"""
import glob
import os
import random
import tarfile
import time
from os import listdir
from os.path import isfile, join

import click
import cv2
import gdown
import numpy as np
import SimpleITK as sitk


def download_data_from_gdrive(raw_path: str) -> None:
    """This function downloads the dataset to the provided base folder"""
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
    t = time.time()
    link = "https://drive.google.com/file/d/1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C/view?usp=sharing"
    output = "Hippocampus.tar"
    gdown.download(link, os.path.join(raw_path, output), quiet=False, fuzzy=True)
    print("Time elapsed in acquiring data is", time.time() - t)


def slice_images(raw_path: str, save_path_slices: str) -> None:
    """This function slices the images into smaller images

    Args:
        raw_path (str): Path to the raw images
        save_path_slices(str): Path to save the sliced images
    """
    image_files = sorted(glob.glob(os.path.join(raw_path, "**", "imagesTr", "*.nii.gz"), recursive=True))
    if not os.path.exists(save_path_slices):
        os.makedirs(save_path_slices)
        os.makedirs(join(save_path_slices, "images"))
        os.makedirs(join(save_path_slices, "labels"))
    slice_num = 0
    for case in image_files:
        print(f"laoding case {case}")

        try:
            img_itk = sitk.ReadImage(case)
        except ():
            print(f"failed for {case}")
            break
        image = sitk.GetArrayFromImage(img_itk)
        msk_path = case.replace("imagesTr", "labelsTr")
        if os.path.exists(msk_path):
            print(msk_path)
            msk_itk = sitk.ReadImage(msk_path)
            mask = sitk.GetArrayFromImage(msk_itk)
            # normalisiert
            image = (image - image.min()) / (image.max() - image.min())
            print(image.shape)
            image = image.astype(np.float32)
            item = case.split("/")[-1].split(".")[0]
            if image.shape != mask.shape:
                print("Error")
            print(item)
            for slice_ind in range(image.shape[0]):
                print(image[slice_ind].shape)
                # skip slices with only zeros in them
                all_zeros = not mask[slice_ind].any()
                if all_zeros is False:
                    cv2.imwrite(
                        os.path.join(save_path_slices, "images", "{}_slice_{}.png").format(item, slice_ind),
                        image[slice_ind] * 255,
                    )
                    cv2.imwrite(
                        os.path.join(save_path_slices, "labels", "{}_slice_{}_mask.png").format(item, slice_ind),
                        mask[slice_ind] * 10,
                    )
                    slice_num += 1
    print("Converted all volumes to 2D slices")
    print("Total {} slices".format(slice_num))


def unpack(tar_path: str):
    tar = tarfile.open(tar_path, "r:")
    tar.extractall(path=os.path.dirname(tar_path))
    tar.close()


@click.command()
@click.argument(
    "base_path",
    type=click.Path(),
    default="/home/gustav/datasets/hippocampus",  # "/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/hippocampus"
)
def main(base_path: str):
    raw_path = os.path.join(base_path, "raw")
    download_data_from_gdrive(raw_path)
    unpack(os.path.join(raw_path, "Hippocampus.tar"))
    slices_path = os.path.join(raw_path, os.pardir, "slices")
    slice_images(raw_path, slices_path)
    # split data into train and test
    filelist = [f for f in listdir(join(base_path, "images")) if isfile(join(base_path, "images", f))]
    # devide into train and test. Only last patient is test
    # 10% test data: first  19 subjects of 195 total
    test_subjects = [f[12:15] for f in filelist]
    test_subjects = list(set(test_subjects))  # remove duplicates
    # shuffle list
    random.shuffle(test_subjects)
    test_subjects = test_subjects[: len(test_subjects) // 10]
    training_filelist = [s for s in filelist if s[12:15] not in test_subjects]
    training_filelist = ["slices/images/%s slices/labels/%s_mask.png" % (f, f[:-4]) for f in training_filelist]
    test_filelist = [s for s in filelist if s[12:15] in test_subjects]
    test_filelist = ["slices/images/%s slices/labels/%s_mask.png" % (f, f[:-4]) for f in test_filelist]
    # split("hippocampus", training_filelist, test_filelist)


if __name__ == "__main__":
    main()
