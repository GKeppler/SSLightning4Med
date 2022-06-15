#%%import numpy as np
import random

import albumentations as A
import cv2
import matplotlib
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

# Define DataModule with Augmentations
transform = A.Compose(
    [
        # A.RandomCrop(224, 224),
        # A.HorizontalFlip(p=0.5),
        # A.GaussianBlur(p=0.5),
        # A.ColorJitter(p=0.8),
        # A.CoarseDropout(max_height=32, max_width=32, mask_fill_value=255),  # cutout
    ]
)
dataset_name = "melanoma"
color_map = get_color_map("melanoma")
with open("/home/gustav/git/SSLightning4Med/SSLightning4Med/data/splits/melanoma/1/split_0/split.yaml") as file:
    split_dict = yaml.load(file, Loader=yaml.FullLoader)
train_id_dict = split_dict["val_split_0"]
dataset = BaseDataset(
    root_dir="/home/gustav/datasets/melanoma256/",
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
    image_aug, mask_aug, _ = dataset[i]
    image_aug = cv2.cvtColor(image_aug, cv2.COLOR_GRAY2RGB)
    # mask_aug = label2rgb(mask_aug, bg_label=0)
    image_aug, mask_aug = cutout(image_aug, mask_aug, p=0.5)
    # mask_aug = np.array(color_map)[mask_aug]

    ax[0, i].imshow(image_aug)
    # ax[0,i].set_title("Image")
    ax[0, i].axis("off")

    ax[1, i].imshow(mask_aug, interpolation="nearest")
    # ax[1,i].set_title("Mask")
    # remove axis
    ax[1, i].axis("off")
# set dpi of plot

# plt.savefig(f'{dataset_name}.png', pad_inches=0, bbox_inches='tight', transparent=True, dpi=300)

# %%
