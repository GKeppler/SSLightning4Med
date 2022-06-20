"""
This script downloads t
Dataset from Syanaps
https://www.synapse.org/#!Synapse:syn3193805/files/

the cervix images were acquired from CT scanners at the Erasmus Medical Center (EMC) Cancer Institute in Rotterdam.
All datasets have been manually labeled by trained raters and reviewed for label accuracy by a
radiologist or radiation oncologist.
Division of scans into training and testing cohorts was performed pseudo
randomly such that data from all scanners was included in both the training testing cohorts.

example implementation:
https://github.com/282857341/nnFormer/blob/main/nnformer/inference_synapse.py

"000": "background",
"010": "spleen",
"020": "right kidney",
"030": "left kidney",
"040": "gallbladder",
"050": "esophagus",
"060": "liver",
"070": "stomach",
"080": "aorta",
"090": "inferior vena cava",
"100": "portal vein and splenic vein",
"110": "pancreas",
"120": "right adrenal gland",
"130": "left adrenal gland"
"""
import glob
import os
import time
import zipfile
from os import listdir
from os.path import isfile, join

import click
import cv2
import dotenv
import numpy as np
import SimpleITK as sitk
import synapseclient
import synapseutils


def download_data_from_synaps(user_name: str, password: str, raw_path: str) -> None:
    """This function downloads the dataset to the provided base folder

    Args:
        user_name (str): Synapse login name
        password (str): Synapse passwort
    """
    syn = synapseclient.Synapse()
    syn.login(user_name, password)
    t = time.time()
    # ids from https://www.synapse.org/#!Synapse:syn3193805/files/
    synapseutils.syncFromSynapse(syn, "syn3553734", raw_path)  # Abdomen.zip
    print("Time elapsed in acquiring data from Synapse.org is", time.time() - t)


def slice_images(raw_path: str, save_path_slices: str) -> None:
    """This function slices the images into smaller images

    Args:
        raw_path (str): Path to the raw images
        save_path_slices(str): Path to save the sliced images
    """
    image_files = sorted(glob.glob(os.path.join(raw_path, "**", "img*.nii.gz"), recursive=True))
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
        msk_path = case.replace("img", "label")
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


def unzip(zip_path: str):
    """Unzip the raw Abodomen.zip file but only the Raw Training Data"""
    archive = zipfile.ZipFile(zip_path)
    for file in archive.namelist():
        if file.startswith("Abdomen/RawData/Training"):
            archive.extract(file, os.path.dirname(zip_path))


@click.command()
@click.argument(
    "base_path", type=click.Path(), default="/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/MultiOrgan"
)  # "/home/gustav/datasets/multiorgan/"
def main(base_path: str):
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    dotenv_path = os.path.join(project_dir, ".env")
    dotenv.load_dotenv(dotenv_path)
    user_name = os.environ.get("SYNAPSE_USER")
    password = os.environ.get("SYNAPSE_PASSWORD")
    raw_path = os.path.join(base_path, "raw")
    download_data_from_synaps(user_name, password, raw_path)
    unzip(os.path.join(raw_path, "Abdomen.zip"))
    slices_path = os.path.join(raw_path, os.pardir, "slices")
    slice_images(raw_path, slices_path)

    # split into training and testing
    filelist = [
        f for f in listdir(join(base_path, "slices", "images")) if isfile(join(base_path, "slices", "images", f))
    ]

    # devide into train and test. Only last patient is test
    test_subjects = ["img" + str(s).zfill(4) for s in list(range(37, 41))]  # 10% test data
    training_filelist = [s for s in filelist if s[:7] not in test_subjects]
    training_filelist = ["slices/images/%s slices/labels/%s_mask.png" % (f, f[:-4]) for f in training_filelist]
    test_filelist = [s for s in filelist if s[:7] in test_subjects]
    test_filelist = ["slices/images/%s slices/labels/%s_mask.png" % (f, f[:-4]) for f in test_filelist]
    # split("multiorgan", training_filelist, test_filelist)


if __name__ == "__main__":
    main()
