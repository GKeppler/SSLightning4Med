import os
import shutil

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

if not os.path.exists("train/images"):
    os.makedirs("train/images")
    os.makedirs("train/labels")
    os.makedirs("test/images")
    os.makedirs("test/labels")

src = "images/"
allFileNames = np.array(os.listdir(src))
# generate list of classes 0,1,2 for normal, malign, beging based on filenames
classes = [(0 if name[0] == "n" else 1 if name[0] == "m" else 2) for name in allFileNames]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
sss.get_n_splits(allFileNames, classes)
for train_index, test_index in sss.split(allFileNames, classes):
    print("TRAIN:", train_index, "TEST:", test_index)
    train_FileNames, test_FileNames = allFileNames[train_index], allFileNames[test_index]


# not straified
# np.random.shuffle(allFileNames)
# train_FileNames, test_FileNames = np.split(np.array(allFileNames), [int(len(allFileNames)*0.8)])

train_FileNames = [src + name for name in train_FileNames.tolist()]
test_FileNames = [src + name for name in test_FileNames.tolist()]

print("Total images: ", len(allFileNames))
print("Training: ", len(train_FileNames))
print("Testing: ", len(test_FileNames))

# Copy-pasting images
for name in train_FileNames:
    shutil.copy(name, "train/" + name)
    name = name.replace("images", "labels").replace(".png", "_mask.png")
    shutil.copy(name, "train/" + name)

for name in test_FileNames:
    shutil.copy(name, "test/" + name)
    name = name.replace("images", "labels").replace(".png", "_mask.png")
    shutil.copy(name, "test/" + name)
