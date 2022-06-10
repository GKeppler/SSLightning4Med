"""
This script downloads the Brats Dataset from Syanaps

"""
import glob
import os
import random
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
    synapseutils.syncFromSynapse(
        syn, "syn25956772", raw_path
    )  # RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021.zip.
    print("Time elapsed in acquiring data from Synapse.org is", time.time() - t)


def slice_images(raw_path: str, save_path_slices: str) -> None:
    """This function slices the images into smaller images

    Args:
        raw_path (str): Path to the raw images
        save_path_slices(str): Path to save the sliced images
    """
    image_files = sorted(glob.glob(os.path.join(raw_path, "**", "*flair.nii.gz"), recursive=True))
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
        msk_path = case.replace("flair", "seg")
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
        if "_flair.nii.gz" in file or "_seg.nii.gz" in file:
            archive.extract(file, os.path.dirname(zip_path))


@click.command()
@click.argument(
    "base_path",
    type=click.Path(),
    default="/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/brats",  # "/home/gustav/datasets/brats"
)
def main(base_path: str):
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    dotenv_path = os.path.join(project_dir, ".env")
    dotenv.load_dotenv(dotenv_path)
    user_name = os.environ.get("SYNAPSE_USER")
    password = os.environ.get("SYNAPSE_PASSWORD")
    raw_path = os.path.join(base_path, "raw")
    download_data_from_synaps(user_name, password, raw_path)
    unzip(os.path.join(raw_path, "RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021.zip"))
    slices_path = os.path.join(raw_path, os.pardir, "slices")
    slice_images(raw_path, slices_path)

    # split the data into train and test
    filelist = [f for f in listdir(join(base_path, "images")) if isfile(join(base_path, "images", f))]
    # devide into train and test. 10% test data
    test_subjects = [f[10:15] for f in filelist]
    test_subjects = list(set(test_subjects))  # remove duplicates
    # shuffle list
    random.shuffle(test_subjects)
    test_subjects = test_subjects[: len(test_subjects) // 10]

    training_filelist = [s for s in filelist if s[10:15] not in test_subjects]
    training_filelist = ["slices/images/%s slices/labels/%s_mask.png" % (f, f[:-4]) for f in training_filelist]
    test_filelist = [s for s in filelist if s[10:15] in test_subjects]
    test_filelist = ["slices/images/%s slices/labels/%s_mask.png" % (f, f[:-4]) for f in test_filelist]
    # drop duplucates from list
    training_filelist = list(set(training_filelist))

    # split("brats", training_filelist, test_filelist)


if __name__ == "__main__":
    main()
