import os
import random
from pathlib import Path
from typing import List

import tqdm
import yaml
from PIL import Image
from sklearn.model_selection import KFold


def resize(base_path: str, old_name, new_name, base_size=256):
    """
    This function resizes all images in a a folder structure and copies them to a new folder.
    e.g all images in folder "train/images" are resized to 256x256 and copied to "train256/images"

    Args:
        base_path (str): Path to the raw images:
        old_name (str): Name of the old folder
        new_name (str): Name of the new folder
        base_size (int): Size of the resized images, but squared(default: 256)
    """

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


def split(
    dataset,
    training_filelist: List[str],
    test_filelist: List[str],
    cross_val_splits=5,
    num_shuffels=5,
    splits=["1", "1/4", "1/8", "1/30"],
) -> None:
    """This function splits the data into training and testing data

    Args:
        dataset (str): name of the dataset
        training_filelist (List[str]): list of training files
        test_filelist (List[str]): list of test files
        cross_val_splits (int): number of cross validation splits (default: 5)
        num_shuffels (int): number of shuffles (default: 5)
        splits (List[str]): list of splits (default: ["1", "1/4", "1/8", "1/30"])
    """

    list_len = len(training_filelist)
    print(training_filelist[:2])
    # get number of spaces in the file names
    assert training_filelist[0].count(" ") == 1

    # shuffle labeled/unlabeled
    for shuffle in range(num_shuffels):
        yaml_dict = {}
        for split in splits:
            random.shuffle(training_filelist)
            # calc splitpoint
            labeled_splitpoint = int(list_len * float(eval(split)))
            print(f"shuffle {shuffle}: splitpoint for {split} in dataset with len {list_len} are {labeled_splitpoint}")
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
