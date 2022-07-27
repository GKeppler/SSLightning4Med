#%%import numpy as np
import os

import albumentations as A
import cv2
import matplotlib
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt

from SSLightning4Med.models.dataset import BaseDataset
from SSLightning4Med.utils.utils import get_color_map

#%%

# Define DataModule with Augmentations
transform = A.Compose([A.SmallestMaxSize(200), A.CenterCrop(200, 200)])

dataset_dict = {
    "melanoma": "ISIC Melanoma 2017",
    "breastCancer": "Breast Ultrasound",
    "pneumothorax": "Pneumothorax",
    "hippocampus": "Hippocampus",
    "zebrafish": "HeartSeg",
    "multiorgan": "Synapse multi-organ",
}
method_dict = {
    "St++": "ST++",
    "Supervised": "Supervised",
    "CCT": "CCT",
    "FixMatch": "FixMatch",
    "MeanTeacher": "MeanTeacher",
}
split_dict = {
    "1_4": "1/4",
    "1_8": "1/8",
    "1_30": "1/30",
}


def preprocess_mask(mask, color_map):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    height, width = mask.shape[:2]
    segmentation_mask = np.zeros((height, width, len(color_map)), dtype=np.float32)
    for label_index, label in enumerate(color_map):
        segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)
    return segmentation_mask


def pascal_color_map_bw():
    cmap = [
        [0, 0, 0],
        [255, 255, 255],
    ]
    return cmap


def pascal_color_map():
    cmap = [
        [0, 0, 0],
        # [255,255,255],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
    ]
    return cmap


# a circel as legend
organ_dict = {
    0: "Background",
    1: "Spleen",
    2: "Right kidney",
    3: "Left kidney",
    4: "Gallbladder",
    5: "Esophagus",
    6: "Liver",
    7: "Stomach",
    8: "Aorta",
    9: "Inferior vena cava",
    10: "Portal vein",
    11: "Pancreas",
    12: "Right adrenal gland",
    13: "Left adrenal gland",
}
zebra_dict = {
    0: "Background",
    1: "Bulbus",
    2: "Atrium",
    3: "Ventricle",
}
hippocampus_dict = {
    0: "Background",
    1: "Anterior",
    2: "Posterior",
}
melanoma_dict = {
    0: "Background",
    1: "Melanoma",
}
breast_dict = {
    0: "Background",
    1: "Breast Nodule",
}
pneumothorax_dict = {
    0: "Background",
    1: "Pneumothorax",
}

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
#%%
df = pd.DataFrame()

fig, ax = plt.subplots(3, 7, figsize=(21, 9))

# make border between images small
fig.subplots_adjust(hspace=0.1, wspace=0.1)


for dataset_name in dataset_dict.keys():
    colormap_dict = {
        "melanoma": pascal_color_map_bw(),
        "breastCancer": pascal_color_map_bw(),
        "pneumothorax": pascal_color_map_bw(),
        "hippocampus": pascal_color_map(),
        "zebrafish": pascal_color_map(),
        "multiorgan": pascal_color_map(),
    }
    color_map2 = colormap_dict[dataset_name]
    data_root = {
        "melanoma": "/home/gustav/datasets/melanoma256",  # /lsdf/kit/iai/projects/iai-aida/Daten_Keppler/melanoma256",
        "breastCancer": "/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/breastCancer256",
        "pneumothorax": "/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/pneumothorax",
        "multiorgan": "/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/multiorgan256",
        "brats": "/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/brats",
        "hippocampus": "/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/hippocampus32",
        "zebrafish": "/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/zebrafish256",
    }[dataset_name]
    i = 0
    for split in split_dict.keys():
        # get image, mask
        color_map = get_color_map(dataset_name)
        with open(f"../data/splits/{dataset_name}/test.yaml") as file:
            test_list = yaml.load(file, Loader=yaml.FullLoader)
        dataset = BaseDataset(
            root_dir=data_root,
            id_list=test_list,
            transform=transform,
            color_map=color_map,
        )
        # get first image
        image, org_mask, id = dataset[0]
        org_mask = np.array(color_map2)[org_mask]
        image = image.copy()
        # save image, mask, id in pandas series object
        df_row = pd.Series(
            {
                "dataset": dataset_name,
                "split": split,
                "image": image,
                "mask": org_mask,
                "id": id,
            }
        )
        for method in method_dict.keys():
            fname = os.path.basename(id.split(" ")[1])
            test_mask = cv2.imread(
                os.path.join(f"{data_root}/test_masks/{method}/{split}/split_0", fname), cv2.IMREAD_UNCHANGED
            )
            mask = preprocess_mask(test_mask, color_map)
            test_mask = transform(image=image, mask=test_mask)["mask"]
            test_mask = np.argmax(test_mask, axis=2)
            test_mask = np.array(color_map2)[test_mask]
            # append to df row
            df_row[method + "_mask"] = test_mask
        ax[i, 0].imshow(df_row["image"])
        ax[i, 1].imshow(df_row["mask"])
        ax[i, 2].imshow(df_row["Supervised_mask"])
        ax[i, 3].imshow(df_row["St++_mask"])
        ax[i, 4].imshow(df_row["CCT_mask"])
        ax[i, 5].imshow(df_row["MeanTeacher"])
        ax[i, 6].imshow(df_row["FixMatch_mask"])
        # add titels to each row of the figure
        ax[i, 0].set_title("Image", fontsize=12)
        ax[i, 1].set_title("Original mask", fontsize=12)
        ax[i, 3].set_title("ST++", fontsize=12)
        ax[i, 4].set_title("CCT", fontsize=12)
        ax[i, 5].set_title("MeanTeacher", fontsize=12)
        ax[i, 6].set_title("FixMatch", fontsize=12)
        ax[i, 0].set_ylabel(split_dict[split], fontsize=12)

        for j in range(7):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
        i += 1

    legend_elements = []
    legend_dict = {
        "zebrafish": zebra_dict,
        "multiorgan": organ_dict,
        "hippocampus": hippocampus_dict,
        "melanoma": melanoma_dict,
        "breastCancer": breast_dict,
        "pneumothorax": pneumothorax_dict,
    }[dataset_name]

    for i in range(0, len(legend_dict)):

        legend_elements.append(
            mpatches.Patch(facecolor=tuple(np.array(color_map2)[i] / 255.0), label=legend_dict[i], edgecolor="black")
        )

    # legend without overlap to axes
    plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 2), ncol=1, title="Classes")

    plt.savefig(f"{dataset_name}_comparison.png", pad_inches=0, bbox_inches="tight", transparent=True, dpi=300)

    # %%
