import math
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from albumentations.core.composition import Compose
from numpy import ndarray
from torch import Tensor


class BaseDatasetEW:
    """
    This is the basic dataset. It hols a list of image_paths and corresponding masks,
    loads them and applies given transformations.
    """

    def __init__(
        self,
        root_dir: str,
        id_list: List[str],
        color_map: ndarray,
        transform: Compose,
        pseudo_mask_path: Optional[str] = None,
        id_list_unlabeled: Optional[List[str]] = None,
        transform_unlabeled: Optional[Compose] = None,
    ) -> None:
        self.root_dir = root_dir
        self.id_list_labeled = id_list
        self.id_list_unlabeled = id_list_unlabeled
        self.id_list = (
            self.id_list_labeled * math.ceil(len(self.id_list_unlabeled) / len(self.id_list_labeled))
            + self.id_list_unlabeled
            if self.id_list_unlabeled is not None
            else self.id_list_labeled
        )
        self.pseudo_mask_path = pseudo_mask_path
        self.transform = transform
        self.color_map = color_map
        self.id_list_unlabeled = id_list_unlabeled
        self.transform_unlabeled = transform_unlabeled

    @staticmethod
    def _preprocess_mask(mask: ndarray, color_map: ndarray) -> ndarray:
        # mask = mask.astype(np.float32)
        # multiclass problem
        # This function converts a mask from the Pascal VOC format to the format required by AutoAlbument.
        #
        # Pascal VOC uses an RGB image to encode the segmentation mask for that image. RGB values of a pixel
        # encode the pixel's class.
        #
        # AutoAlbument requires a segmentation mask to be a NumPy array with the shape [height, width, num_classes].
        # Each channel in this mask should encode values for a single class. Pixel in a mask channel should have
        # a value of 1.0 if the pixel of the image belongs to this class and 0.0 otherwise.
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, len(color_map)), dtype=np.float32)
        for label_index, label in enumerate(color_map):
            segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)
        return segmentation_mask

    def __len__(self) -> int:
        assert len(self.id_list) != 0, "Oh no! A Dataset is emtpy"
        return len(self.id_list)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, str]:
        id = self.id_list[idx]
        img_path = os.path.join(self.root_dir, id.split(" ")[0])
        # Read an image with OpenCV
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"No image loaded. Check Dataset. id: {id}")
        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.

        # check if image is grayscale, if not convert to RGB
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.pseudo_mask_path is not None and id in self.id_list_unlabeled:
            fname = os.path.basename(id.split(" ")[1])
            mask = cv2.imread(os.path.join(self.pseudo_mask_path, fname), cv2.IMREAD_UNCHANGED)
        else:
            mask = cv2.imread(os.path.join(self.root_dir, id.split(" ")[1]), cv2.IMREAD_UNCHANGED)

        mask = self._preprocess_mask(mask, self.color_map)
        if self.id_list_unlabeled is not None and id in self.id_list_unlabeled:
            transformed = self.transform_unlabeled(image=image, mask=mask)
        else:
            transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]
        mask = np.argmax(mask, axis=2)
        return image, mask, id
