import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
'''
This is the basic dataset. It hols a list of image_paths and corresponding masks,
loads them and applies given transformations.
'''


class BaseDataset:
    def __init__(
        self,
        root_dir: str,
        id_list=None,
        pseudo_mask_path=None,
        transform=None
    ):
        self.root_dir = root_dir
        self.id_list = id_list
        self.pseudo_mask_path = pseudo_mask_path
        if transform is None:
            self.transform = A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.id_list)



    def __getitem__(self, idx):
        id = self.id_list[idx]
        img_path = os.path.join(self.root_dir, id.split(" ")[0])
        # Read an image with OpenCV
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError
        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.pseudo_mask_path is not None:
            fname = os.path.basename(id.split(" ")[1])
            mask = cv2.imread(os.path.join(self.pseudo_mask_path, fname), cv2.IMREAD_UNCHANGED)
        else:
            mask = cv2.imread(os.path.join(self.root_dir, id.split(" ")[1]), cv2.IMREAD_UNCHANGED)

        mask = preprocess_mask(mask)
        transformed = self.transform(image=image,mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        return image, mask, id

def preprocess_mask(mask):
    mask = mask.astype(np.float32)
    mask[mask == 255.0] = 1.0
    #mask[(mask == 1.0) | (mask == 3.0)] = 1.0
    return mask