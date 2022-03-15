import os
import shutil

import numpy as np
from PIL import Image

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
                    cmap[255] = np.array([128, 64, 128])  # is equal to label 0
                elif i == 1:
                    cmap[255] = np.array([244, 35, 232])  # is label 1
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
