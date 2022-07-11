#%%import numpy as np

import os

import albumentations as A
import cv2
import numpy as np
import yaml
from torch.utils.data import DataLoader

from SSLightning4Med.models.dataset import BaseDataset
from SSLightning4Med.utils.utils import get_color_map

# from https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/17
if __name__ == "__main__":
    root_dir = "/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/hippocampus"
    path_new = root_dir.replace("hippocampus", "hippocampus32")
    dataset_name = "hippocampus"
    color_map = get_color_map(dataset_name)
    a_train_transforms = A.Compose([A.SmallestMaxSize(32), A.CenterCrop(32, 32)])
    # standard dataloader -> uses labeled

    with open("../data/splits/hippocampus/1/split_0/split.yaml") as file:
        split_dict = yaml.load(file, Loader=yaml.FullLoader)
    val_split_0 = split_dict["val_split_0"]

    with open("../data/splits/hippocampus/test.yaml") as file:
        test_list = yaml.load(file, Loader=yaml.FullLoader)

    full_dataset = BaseDataset(
        root_dir=root_dir,
        id_list=val_split_0["labeled"] + val_split_0["unlabeled"] + val_split_0["val"] + test_list,
        transform=a_train_transforms,
        color_map=get_color_map("hippocampus"),
    )
    loader = DataLoader(full_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    for img, mask, id in loader:
        img_path = os.path.join(path_new, id[0].split(" ")[0])
        mask_path = os.path.join(path_new, id[0].split(" ")[1])
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        mask = mask.squeeze(0).cpu().numpy().astype(np.uint8)
        mask = np.array(color_map)[mask]
        cv2.imwrite(
            mask_path,
            cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_BGR2RGB),
        )
        cv2.imwrite(
            img_path,
            img.squeeze(0).cpu().numpy().astype(np.uint8),
        )
