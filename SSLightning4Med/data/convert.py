import os

import cv2
import numpy as np

# this script changes the color of all masks, as the first try wasn't good

path = "/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/BreastCancer"
old_names = ["benign", "malignant", "normal"]
dirs = os.listdir(path)

##copy maks to labels folder and convert white backgroudn to corresponsing color
for path, subdirs, files in os.walk(path):
    gen = (name for name in files if name[-8:] == "mask.png")
    for name in gen:
        img_path = os.path.join(path, name)
        folder_path_new = os.path.dirname(img_path)
        filename = os.path.basename(img_path)
        if "train" in img_path or "test" in img_path:
            folder_path_new = folder_path_new.replace("labels", "labels_neu")
            img_path_new = os.path.join(folder_path_new, filename)
            if not os.path.exists(os.path.dirname(img_path_new)):
                os.makedirs(os.path.dirname(img_path_new))
            old_im = cv2.imread(img_path)
            old_im = cv2.cvtColor(old_im, cv2.COLOR_BGR2RGB)
            w, h = old_im.shape[:2]
            image = np.zeros((w, h, 3), dtype="uint8")
            image[np.where((old_im == [128, 64, 128]).all(axis=2))] = [255, 0, 0]
            image[np.where((old_im == [244, 35, 232]).all(axis=2))] = [0, 255, 0]
            cv2.imwrite(
                img_path_new,
                image,
            )
