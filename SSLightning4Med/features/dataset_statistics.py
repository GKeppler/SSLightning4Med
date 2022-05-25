"""This script calculated the dataset statistics(mean, standard deviation) for image normalization.
ImageNet: mean = [0.485, 0.456, 0.406] and                      std = [0.229, 0.224, 0.225]

melanoma        mean:  tensor([0.7116, 0.5834, 0.5337]) std:  tensor([0.1471, 0.1646, 0.1795])
pneumothorax    mean:  tensor([0.5380])                 std:  tensor([0.2641])
breastCancer    mean:  tensor([0.3298])                 std:  tensor([0.2218])
multiorgan      mean:  tensor([0.1935])                 std:  tensor([0.1889])
brats           mean:  tensor([0.0775])                 std:  tensor([0.1539])
hippocampus     mean:  tensor([0.2758])                 std:  tensor([0.1628])
"""
import albumentations as A
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from SSLightning4Med.models.dataset import BaseDataset
from SSLightning4Med.models.train_CCT import CCTModule
from SSLightning4Med.train import base_parse_args
from SSLightning4Med.utils.utils import get_color_map

# from https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/17
if __name__ == "__main__":
    args = base_parse_args(CCTModule)

    if args.n_channel == 3:
        a_train_transforms = A.Compose(
            [
                # scale from 0 .. 255 to 0 .. 1
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
                ToTensorV2(),
            ]
        )
    elif args.n_channel == 1:
        a_train_transforms = A.Compose(
            [
                # scale from 0 .. 255 to 0 .. 1
                A.Normalize(mean=(0), std=(1)),
                ToTensorV2(),
            ]
        )
    # standard dataloader -> uses labeled

    with open(args.split_file_path, "r") as file:
        split_dict = yaml.load(file, Loader=yaml.FullLoader)
    val_split_0 = split_dict["val_split_0"]

    full_dataset = BaseDataset(
        root_dir=args.data_root,
        id_list=val_split_0["labeled"] + val_split_0["unlabeled"] + val_split_0["val"],
        transform=a_train_transforms,
        color_map=get_color_map(args.dataset),
    )
    loader = DataLoader(full_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _, _ in loader:
        # Mean over batch, height and width, but not over the channels
        data = data.float()
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    print(args.dataset, " mean: ", mean, "std: ", std)
