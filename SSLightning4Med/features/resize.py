# this notebook resizes all images in a folder to center crop
import os

from PIL import Image

path = "/lsdf/kit/iai/projects/iai-aida/Daten_Keppler/BreastCancer"
old_name = "BreastCancer"
new_name = "BreastCancer_cropped"
dirs = os.listdir(path)


def resize_crop(img: Image, base_size: int) -> Image:
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
        if img_path.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")):
            im = Image.open(img_path)
            imResize = resize_crop(im, 512)
            img_path_new = img_path.replace(old_name, new_name)
            if not os.path.exists(os.path.dirname(img_path_new)):
                os.makedirs(os.path.dirname(img_path_new))
            imResize.save(img_path_new)
