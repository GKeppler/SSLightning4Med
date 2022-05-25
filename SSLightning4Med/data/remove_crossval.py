"""
This script removes the cross val entries in the split fiels from brats to reduce the size

"""
import glob
import os
from pathlib import Path

import yaml
from yaml import CLoader as Loader

raw_path = "/home/gustav/git/SSLightning4Med/SSLightning4Med/data/splits"

split_files = sorted(glob.glob(os.path.join(raw_path, "**", "*.yaml"), recursive=True))


for split_file in split_files:
    if "brats" not in split_file:
        continue
    with open(split_file, "r") as file:
        split_dict = yaml.load(file, Loader=Loader)
    id_dict = {}
    if "test" not in split_file:
        id_dict["val_split_0"] = split_dict["val_split_0"]
    else:
        id_dict = split_dict

    new_split_file = split_file.replace("/splits", "/splits_new")

    Path(os.path.dirname(new_split_file)).mkdir(parents=True, exist_ok=True)
    with open(new_split_file, "w+") as outfile:
        yaml.dump(id_dict, outfile, default_flow_style=False)
