#%%import numpy as np
import albumentations as A
import cv2
import yaml
from matplotlib import pyplot as plt
from skimage.color import label2rgb

from SSLightning4Med.models.dataset import BaseDataset
from SSLightning4Med.utils.utils import get_color_map

#%%

# Define DataModule with Augmentations
transform = A.Compose(
    [
        # A.RandomCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.GaussianBlur(p=0.5),
        A.ColorJitter(p=0.8),
        A.CoarseDropout(max_height=32, max_width=32, mask_fill_value=255),  # cutout
    ]
)
color_map = get_color_map("multiorgan")
with open("/home/gustav/git/SSLightning4Med/SSLightning4Med/data/splits/multiorgan/laptop_test/split.yaml") as file:
    split_dict = yaml.load(file, Loader=yaml.FullLoader)
train_id_dict = split_dict["val_split_0"]
dataset = BaseDataset(
    root_dir="/home/gustav/datasets/multiorgan/",
    id_list=train_id_dict["labeled"],
    transform=transform,
    color_map=color_map,
)

for i in range(0, 10):
    # visulaize mask and image
    image_aug, mask_aug, _ = dataset[i]
    image_aug = cv2.cvtColor(image_aug, cv2.COLOR_GRAY2RGB)
    f, ax = plt.subplots(1, 2, figsize=(16, 16))
    mask_aug = label2rgb(mask_aug, bg_label=0)
    # mask_aug = np.array(color_map)[mask_aug]

    ax[0].imshow(image_aug)
    ax[0].set_title("Augmented image")

    ax[1].imshow(mask_aug, interpolation="nearest")
    ax[1].set_title("Augmented mask")


# %%
