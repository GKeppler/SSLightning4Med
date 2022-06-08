import os
import random
import shutil
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import List

import numpy as np
import yaml
from PIL import Image
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

# this script copies the images/masks from the folders benign, malignant and normal to an overall
# images and mask folder. Also the mask images are preprocessed an assigned a color depensing on the label

path = "/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/BreastCancer"
old_names = ["benign", "malignant", "normal"]
dirs = os.listdir(path)

##copy maks to labels folder and convert white backgroudn to corresponsing color
cmap = np.zeros((256, 3), dtype="uint8")
for path, subdirs, files in os.walk(path):
    gen = (name for name in files if name[-8:] == "mask.png")
    for name in gen:
        img_path = os.path.join(path, name)
        folder_path_new = os.path.dirname(img_path)
        filename = os.path.basename(img_path)
        for i, old_name in enumerate(old_names):
            if old_name in folder_path_new:
                if i == 0:
                    cmap[255] = np.array([0, 0, 255])  # is equal to label 0
                elif i == 1:
                    cmap[255] = np.array([0, 255, 0])  # is label 1
            folder_path_new = folder_path_new.replace(old_name, "labels")
        img_path_new = os.path.join(folder_path_new, filename)
        if not os.path.exists(os.path.dirname(img_path_new)):
            os.makedirs(os.path.dirname(img_path_new))

        im = Image.open(img_path)
        im = im.convert("P")
        im.putpalette(cmap)
        im.save(img_path_new.replace(" ", ""))

path = "/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/BreastCancer"
# copy images to images folder
for path, subdirs, files in os.walk(path):
    gen = (name for name in files if name[-5:] == ").png")
    for name in gen:
        img_path = os.path.join(path, name)
        folder_path_new = os.path.dirname(img_path)
        filename = os.path.basename(img_path)
        for i, old_name in enumerate(old_names):
            folder_path_new = folder_path_new.replace(old_name, "images")
        img_path_new = os.path.join(folder_path_new, filename)
        if not os.path.exists(os.path.dirname(img_path_new)):
            os.makedirs(os.path.dirname(img_path_new))
        shutil.copy(img_path, img_path_new.replace(" ", ""))


# import os

# import cv2
# import numpy as np

# # this script changes the color of all masks, as the first try wasn't good

# path = "/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/BreastCancer"
# old_names = ["benign", "malignant", "normal"]
# dirs = os.listdir(path)

# ##copy maks to labels folder and convert white backgroudn to corresponsing color
# for path, subdirs, files in os.walk(path):
#     gen = (name for name in files if name[-8:] == "mask.png")
#     for name in gen:
#         img_path = os.path.join(path, name)
#         folder_path_new = os.path.dirname(img_path)
#         filename = os.path.basename(img_path)
#         if "train" in img_path or "test" in img_path:
#             folder_path_new = folder_path_new.replace("labels", "labels_neu")
#             img_path_new = os.path.join(folder_path_new, filename)
#             if not os.path.exists(os.path.dirname(img_path_new)):
#                 os.makedirs(os.path.dirname(img_path_new))
#             old_im = cv2.imread(img_path)
#             old_im = cv2.cvtColor(old_im, cv2.COLOR_BGR2RGB)
#             w, h = old_im.shape[:2]
#             image = np.zeros((w, h, 3), dtype="uint8")
#             image[np.where((old_im == [128, 64, 128]).all(axis=2))] = [255, 0, 0]
#             image[np.where((old_im == [244, 35, 232]).all(axis=2))] = [0, 255, 0]
#             cv2.imwrite(
#                 img_path_new,
#                 image,
#             )


# stratified train test split

base_path = r"/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/BreastCancer"

if not os.path.exists("train/images"):
    os.makedirs("train/images")
    os.makedirs("train/labels")
    os.makedirs("test/images")
    os.makedirs("test/labels")

src = "images/"
allFileNames = np.array(os.listdir(src))
# generate list of classes 0,1,2 for normal, malign, beging based on filenames
classes = [(0 if name[0] == "n" else 1 if name[0] == "m" else 2) for name in allFileNames]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
sss.get_n_splits(allFileNames, classes)
for train_index, test_index in sss.split(allFileNames, classes):
    print("TRAIN:", train_index, "TEST:", test_index)
    train_FileNames, test_FileNames = allFileNames[train_index], allFileNames[test_index]


# not straified
# np.random.shuffle(allFileNames)
# train_FileNames, test_FileNames = np.split(np.array(allFileNames), [int(len(allFileNames)*0.8)])

train_FileNames = [src + name for name in train_FileNames.tolist()]
test_FileNames = [src + name for name in test_FileNames.tolist()]

print("Total images: ", len(allFileNames))
print("Training: ", len(train_FileNames))
print("Testing: ", len(test_FileNames))

# Copy-pasting images
for name in train_FileNames:
    shutil.copy(name, "train/" + name)
    name = name.replace("images", "labels").replace(".png", "_mask.png")
    shutil.copy(name, "train/" + name)

for name in test_FileNames:
    shutil.copy(name, "test/" + name)
    name = name.replace("images", "labels").replace(".png", "_mask.png")
    shutil.copy(name, "test/" + name)


# stratified split


# set basic params and load file list
cross_val_splits = 5
num_shuffels = 5
splits = ["1", "1/4", "1/8", "1/30"]
images_folder = "images"
labels_folder = "labels"
training_filelist: List[str] = []
val_filelist: List[str] = []
test_filelist: List[str] = []

dataset = r"breastCancer"
base_path = r"/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/BreastCancer"
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
    raise Exception(f"files in folders '{images_folder}' and '{labels_folder}' do not match because of: {differences}")

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
