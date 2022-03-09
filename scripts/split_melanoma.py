import pandas as pd
import random
import yaml
from pathlib import Path
from sklearn.model_selection import KFold

# set basic params and load file list
dataset = r"melanoma"
cross_val_splits = 5
num_shuffels = 1
splits = ["1/40"]
csv_train_path = r"ISIC-2017_Training_Part3_GroundTruth(1).csv"
csv_test_path = r"ISIC-2017_Validation_Part3_GroundTruth.csv"
images_folder = "images"
labels_folder = "labels"
training_filelist = []
val_filelist = []
test_filelist = []

training_filelist = pd.read_csv(csv_train_path)["image_id"].to_list()
training_filelist = [
    "train/images/%s.jpg train/labels/%s_segmentation.png" % (f, f)
    for f in training_filelist
]

# all iamges are in this case in the train folder
test_filelist = pd.read_csv(csv_test_path)["image_id"].to_list()
test_filelist = [
    "train/images/%s.jpg train/labels/%s_segmentation.png" % (f, f)
    for f in test_filelist
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
        print(
            f"splitpoint for {split} in dataset with list_len {list_len} are {labeled_splitpoint}"
        )
        unlabeled = training_filelist[labeled_splitpoint:]
        labeled = training_filelist[:labeled_splitpoint]
        kf = KFold(n_splits=cross_val_splits)
        count = 0
        for train_index, val_index in kf.split(labeled):
            unlabeled_copy = unlabeled.copy()  # or elese it cant be reused
            train = [labeled[i] for i in train_index]
            val = [labeled[i] for i in val_index]
            yaml_dict["val_split_" + str(count)] = dict(
                unlabeled=unlabeled_copy, labeled=train, val=val
            )
            count += 1

        # save to yaml
        # e.g 1/4 -> 1_4 for folder name
        zw = list(split)
        if len(zw) > 1:
            zw[1] = "_"
        split = "".join(zw)

        yaml_path = rf"dataset/splits/{dataset}/{split}/split_{shuffle}"
        Path(yaml_path).mkdir(parents=True, exist_ok=True)
        with open(yaml_path + "/split.yaml", "w+") as outfile:
            yaml.dump(yaml_dict, outfile, default_flow_style=False)

# test yaml file
yaml_dict = {}
yaml_path = rf"dataset/splits/{dataset}/"
Path(yaml_path).mkdir(parents=True, exist_ok=True)

with open(yaml_path + "/test_valset.yaml", "w+") as outfile:
    yaml.dump(test_filelist, outfile, default_flow_style=False)
