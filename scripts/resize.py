# this notebook resizes all images in a folder to center crop
import os

from PIL import Image

path = "/home/gustav/datasets/ISIC_Demo_2017/"
old_name = "ISIC_Demo_2017"
new_name = "ISIC_Demo_2017_small"
dirs = os.listdir(path)


def resize_crop(img, base_size):
    w, h = img.size
    if h > w:
        crop_size = w
    else:
        crop_size = h
    left = (w - crop_size) / 2
    top = (h - crop_size) / 2
    right = (w + crop_size) / 2
    bottom = (h + crop_size) / 2
    # make it sqaure
    img = img.crop((left, top, right, bottom))

    # resize to base_size
    img = img.resize((base_size, base_size), Image.NEAREST)
    return img


for path, subdirs, files in os.walk(path):
    for name in files:
        img_path = os.path.join(path, name)
        im = Image.open(img_path)
        imResize = resize_crop(im, 512)
        img_path_new = img_path.replace(old_name, new_name)
        if not os.path.exists(os.path.dirname(img_path_new)):
            os.makedirs(os.path.dirname(img_path_new))
        imResize.save(img_path_new)
