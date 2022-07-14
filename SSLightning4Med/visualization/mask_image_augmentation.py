#%%import numpy as np
import random

import albumentations as A
import cv2
import matplotlib
import matplotlib.patches as mpatches
import numpy as np
import yaml
from matplotlib import pyplot as plt

from SSLightning4Med.models.dataset import BaseDataset
from SSLightning4Med.utils.utils import get_color_map


#%%
def cutout(
    img,
    mask,
    p=0.5,
    size_min=0.02,
    size_max=0.4,
    ratio_1=0.3,
    ratio_2=1 / 0.3,
    value_min=0,
    value_max=255,
    pixel_level=False,
):
    if random.random() < p:

        img_h, img_w, img_c = img.shape

        while True:
            size = np.random.uniform(size_min, size_max) * img_h * img_w
            ratio = np.random.uniform(ratio_1, ratio_2)
            erase_w = int(np.sqrt(size / ratio))
            erase_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, img_w)
            y = np.random.randint(0, img_h)

            if x + erase_w <= img_w and y + erase_h <= img_h:
                break

        if pixel_level:
            value = np.random.uniform(value_min, value_max, (erase_h, erase_w, img_c))
        else:
            value = np.random.uniform(value_min, value_max)

        img[y : y + erase_h, x : x + erase_w] = value
        mask[y : y + erase_h, x : x + erase_w] = 255
    return img, mask


#%%
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


#%%

# Define DataModule with Augmentations
transform = A.Compose(
    [
        # A.RandomCrop(224, 224),
        A.SmallestMaxSize(50),
        A.CenterCrop(50, 50)
        # A.HorizontalFlip(p=0.5),
        # A.GaussianBlur(p=0.5),
        # A.ColorJitter(p=0.8),
        # A.CoarseDropout(max_height=32, max_width=32, mask_fill_value=255),  # cutout
    ]
)
dataset_name = "hippocampus"
color_map = get_color_map(dataset_name)
color_map2 = pascal_color_map()
with open(f"/home/gustav/git/SSLightning4Med/SSLightning4Med/data/splits/{dataset_name}/1/split_0/split.yaml") as file:
    split_dict = yaml.load(file, Loader=yaml.FullLoader)
train_id_dict = split_dict["val_split_0"]
dataset = BaseDataset(
    root_dir="/home/gustav/datasets/hippocampus/",
    id_list=train_id_dict["labeled"],
    transform=transform,
    color_map=color_map,
)
# plot figures
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
# plt.tight_layout()
f, ax = plt.subplots(2, 4, figsize=(8, 4))
# remove spacing between subplots
plt.subplots_adjust(wspace=0.1, hspace=0)
for i in range(0, 4):
    # visulaize mask and image
    image_aug, mask_aug, _ = dataset[i * 39 + 55]
    try:
        image_aug = cv2.cvtColor(image_aug, cv2.COLOR_GRAY2RGB)
    except Exception as e:
        pass
    # mask_aug = label2rgb(mask_aug, bg_label=0)
    # image_aug, mask_aug = cutout(image_aug, mask_aug, p=0.5)
    print(np.unique(mask_aug))
    mask_aug = np.array(color_map2)[mask_aug]

    ax[0, i].imshow(image_aug)
    # ax[0,i].set_title("Image")
    ax[0, i].axis("off")

    ax[1, i].imshow(mask_aug, interpolation="nearest")
    # ax[1,i].set_title("Mask")
    # remove axis
    ax[1, i].axis("off")
# set dpi of plot
# add legend based on color map
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

# f.legend(handles=legend_elements, ncol=1,
#            loc="center right",   # Position of legend
#            borderaxespad=0.1,
# )
plt.savefig(f"{dataset_name}.png", pad_inches=0, bbox_inches="tight", transparent=True, dpi=300)
# %%
