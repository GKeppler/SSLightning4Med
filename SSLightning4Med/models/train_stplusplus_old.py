import math
import os
import random
from copy import deepcopy

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import wandb
from SSLightning4Med.models.base_module import BaseModule
from SSLightning4Med.models.transform import (  # to_polar,; to_cart,
    blur,
    crop,
    cutout,
    hflip,
    normalize,
    resize,
    resize_crop,
)
from SSLightning4Med.nets.unet import UNet, small_UNet
from SSLightning4Med.train import base_parse_args
from SSLightning4Med.utils.utils import mulitmetrics

MODE = None
global step_train
global step_val
step_train = 0
step_val = 0


def main(args):
    if args.use_wandb:
        wandb.init(project=args.wandb_project, entity="gkeppler")
        wandb.config.update(args)

    criterion = CrossEntropyLoss(ignore_index=255)
    valset = SemiDataset(args.dataset, args.data_root, "val", args.crop_size, args.split_file_path)
    valloader = DataLoader(
        valset,
        batch_size=4 if args.dataset == "cityscapes" else 1,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=False,
    )

    # <====================== Supervised training with labeled images (SupOnly) ======================>
    print(
        "\n================> Total stage 1/%i: "
        "Supervised training on labeled images (SupOnly)" % (6 if args.plus else 3)
    )

    global MODE
    MODE = "train"

    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size, args.split_file_path)
    trainset.ids = 2 * trainset.ids if len(trainset.ids) < 200 else trainset.ids
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.n_workers,
        drop_last=True,
    )  # ,sampler=torch.utils.data.SubsetRandomSampler(subset_indices))

    model, optimizer = init_basic_elems(args)
    print("\nParams: %.1fM" % count_params(model))

    best_model, checkpoints = train(model, trainloader, valloader, criterion, optimizer, args)

    # <====================== Test supervised model on testset (SupOnly) ======================>
    print("\n\n\n================> Test supervised model on testset (SupOnly)")
    testset = SemiDataset(args.dataset, args.data_root, "test", args.crop_size, args.test_file_path)
    testloader = DataLoader(testset, 1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    test(best_model, testloader, args)

    """
        ST framework without selective re-training
    """
    if not args.plus:
        # <============================= Pseudo label all unlabeled images =============================>
        print("\n\n\n================> Total stage 2/3: Pseudo labeling all unlabeled images")

        dataset = SemiDataset(args.dataset, args.data_root, "label", args.crop_size, args.split_file_path)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
            drop_last=False,
        )

        label(best_model, dataloader, args)

        # <======================== Re-training on labeled and unlabeled images ========================>
        print("\n\n\n================> Total stage 3/3: Re-training on labeled and unlabeled images")

        MODE = "semi_train"

        trainset = SemiDataset(
            args.dataset,
            args.data_root,
            MODE,
            args.crop_size,
            args.split_file_path,
            args.pseudo_mask_path,
        )
        trainloader = DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.n_workers,
            drop_last=True,
        )

        model, optimizer = init_basic_elems(args)

        best_model = train(model, trainloader, valloader, criterion, optimizer, args)

        # <====================== Test supervised model on testset (SupOnly) ======================>
        print("\n\n\n================> Test supervised model on testset (Re-trained)")

        test(best_model, testloader, args)

        return

    """
        ST++ framework with selective re-training
    """
    # <===================================== Select Reliable IDs =====================================>
    print("\n\n\n================> Total stage 2/6: Select reliable images for the 1st stage re-training")

    dataset = SemiDataset(args.dataset, args.data_root, "label", args.crop_size, args.split_file_path)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=False,
    )

    select_reliable(checkpoints, dataloader, args)

    # <================================ Pseudo label reliable images =================================>
    print("\n\n\n================> Total stage 3/6: Pseudo labeling reliable images")

    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, "reliable_ids.yaml")
    dataset = SemiDataset(
        args.dataset,
        args.data_root,
        "label",
        args.crop_size,
        cur_unlabeled_id_path,
        None,
        True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=False,
    )

    label(best_model, dataloader, args)

    # <================================== The 1st stage re-training ==================================>
    print(
        "\n\n\n================> Total stage 4/6: The 1st stage re-training on labeled and reliable unlabeled images"
    )

    MODE = "semi_train"

    trainset = SemiDataset(
        args.dataset,
        args.data_root,
        MODE,
        args.crop_size,
        cur_unlabeled_id_path,
        args.pseudo_mask_path,
        True,
    )
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.n_workers,
        drop_last=True,
    )

    model, optimizer = init_basic_elems(args)

    best_model = train(model, trainloader, valloader, criterion, optimizer, args)

    # <=============================== Pseudo label unreliable images ================================>
    print("\n\n\n================> Total stage 5/6: Pseudo labeling unreliable images")

    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, "reliable_ids.yaml")
    dataset = SemiDataset(
        args.dataset,
        args.data_root,
        "label",
        args.crop_size,
        cur_unlabeled_id_path,
        None,
        False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=False,
    )

    label(best_model, dataloader, args)

    # <================================== The 2nd stage re-training ==================================>
    print("\n\n\n================> Total stage 6/6: The 2nd stage re-training on labeled and all unlabeled images")

    trainset = SemiDataset(
        args.dataset,
        args.data_root,
        MODE,
        args.crop_size,
        args.split_file_path,
        args.pseudo_mask_path,
    )
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.n_workers,
        drop_last=True,
    )

    model, optimizer = init_basic_elems(args)

    best_model = train(model, trainloader, valloader, criterion, optimizer, args)

    # <====================== Test supervised model on testset (Re-trained) ======================>
    print("\n\n\n================> Test supervised model on testset (Re-trained)")

    test(best_model, testloader, args)

    wandb.finish()


net_zoo = {
    "DeepLabV3Plus": (None, None),
    "Unet": (UNet, None),
    "smallUnet": (small_UNet, None),
    "SegFormer": (None, None),
}


def init_basic_elems(args):
    model = net_zoo[args.net][0]
    model = model(in_chns=args.n_channel, n_class=args.n_class)  # if args.n_class > 2 else 1)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model = model.cuda()
    # model = DataParallel(model).cuda()

    return model, optimizer


def train(model, trainloader, valloader, criterion, optimizer, args):
    global step_train
    global step_val

    previous_best = 0.0

    global MODE

    if MODE == "train":
        checkpoints = []

    for epoch in range(args.epochs):
        print(
            "\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f"
            % (epoch, optimizer.param_groups[0]["lr"], previous_best)
        )

        model.train()
        total_loss = 0.0
        tbar = tqdm(trainloader)

        for i, (img, mask) in enumerate(tbar):
            if args.dataset == "melanoma" or args.dataset == "breastCancer":
                mask = mask.clip(max=1)

            img, mask = img.cuda(), mask.cuda()

            pred = model(img)
            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # wandb log with custom step
            if args.use_wandb:
                wandb.log({"loss": loss, "step_train": step_train, "epoch": epoch})
            step_train += 1
            tbar.set_description("Loss: %.3f" % (total_loss / (i + 1)))

        metric = meanIOU(
            num_classes=21
            if args.dataset == "pascal"
            else 2
            if args.dataset == "melanoma"
            else 2
            if args.dataset == "breastCancer"
            else 19
        )

        model.eval()
        tbar = tqdm(valloader)
        # set i for sample images
        i = 0
        wandb_iamges = []
        torch.cuda.empty_cache()
        with torch.no_grad():
            for img, mask, _ in tbar:
                if args.dataset == "melanoma" or args.dataset == "breastCancer":
                    mask = mask.clip(max=1)
                i = i + 1
                img = img.cuda()
                pred = model(img)
                pred = torch.argmax(pred, dim=1)

                metric.add_batch(pred.cpu().numpy(), mask.numpy())
                mIOU = metric.evaluate()[-1]
                if args.use_wandb:
                    wandb.log({"mIOU": mIOU, "step_val": step_val})
                    if i <= 10:
                        # wandb.log({"img": [wandb.Image(img, caption="img")]})
                        # wandb.log({"mask": [wandb.Image(pred.cpu().numpy(), caption="mask")]})
                        class_labeles = dict((el, "something") for el in list(range(21)))
                        class_labeles.update({255: "boarder"})
                        class_labeles.update({0: "nothing"})
                        wandb_iamge = wandb.Image(
                            cv2.resize(
                                np.moveaxis(np.squeeze(img.cpu().numpy(), axis=0), 0, -1),
                                dsize=(100, 100),
                                interpolation=cv2.INTER_NEAREST,
                            ),
                            masks={
                                "predictions": {
                                    "mask_data": cv2.resize(
                                        np.squeeze(pred.cpu().numpy(), axis=0),
                                        dsize=(100, 100),
                                        interpolation=cv2.INTER_NEAREST,
                                    ),
                                    "class_labels": class_labeles,
                                },
                                "ground_truth": {
                                    "mask_data": cv2.resize(
                                        np.squeeze(mask.numpy(), axis=0),
                                        dsize=(100, 100),
                                        interpolation=cv2.INTER_NEAREST,
                                    ),
                                    "class_labels": class_labeles,
                                },
                            },
                        )
                        wandb_iamges.append(wandb_iamge)
                step_val += 1

                tbar.set_description("mIOU: %.2f" % (mIOU * 100.0))
        if args.use_wandb:
            wandb.log({"Pictures": wandb_iamges, "step_epoch": epoch})
            wandb.log({"final mIOU": mIOU})
        mIOU *= 100.0
        if mIOU > previous_best:
            if previous_best != 0:
                os.remove(
                    os.path.join(
                        args.save_path,
                        f"{args.net}-{previous_best}.pth",
                    )
                )
            previous_best = mIOU
            torch.save(
                model.state_dict(),
                os.path.join(
                    args.save_path,
                    f"{args.net}-{mIOU}.pth",  # "%s_%.2f.pth" % (args.method, mIOU)
                ),
            )

            best_model = deepcopy(model)

        if MODE == "train" and ((epoch + 1) in [args.epochs // 3, args.epochs * 2 // 3, args.epochs]):
            checkpoints.append(deepcopy(model))

    if MODE == "train":
        return best_model, checkpoints

    return best_model


def select_reliable(models, dataloader, args):
    if not os.path.exists(args.reliable_id_path):
        os.makedirs(args.reliable_id_path)

    for i in range(len(models)):
        models[i].eval()
    tbar = tqdm(dataloader)

    id_to_reliability = []

    with torch.no_grad():
        for img, mask, id in tbar:
            if args.dataset == "melanoma" or args.dataset == "breastCancer":
                mask = mask.clip(max=1)
            img = img.cuda()

            preds = []
            for model in models:
                preds.append(torch.argmax(model(img), dim=1).cpu().numpy())

            mIOU = []
            for i in range(len(preds) - 1):
                metric = meanIOU(
                    num_classes=21
                    if args.dataset == "pascal"
                    else 2
                    if args.dataset == "melanoma"
                    else 2
                    if args.dataset == "breastCancer"
                    else 19
                )
                metric.add_batch(preds[i], preds[-1])
                mIOU.append(metric.evaluate()[-1])

            reliability = sum(mIOU) / len(mIOU)
            id_to_reliability.append((id[0], reliability))

    labeled_ids = []
    with open(args.split_file_path, "r") as file:
        split_dict = yaml.load(file, Loader=yaml.FullLoader)
        labeled_ids = split_dict[args.val_split]["labeled"]

    yaml_dict = dict()
    yaml_dict[args.val_split] = dict(
        labeled=labeled_ids,
        reliable=[i[0] for i in id_to_reliability[: len(id_to_reliability) // 2]],
        unreliable=[i[0] for i in id_to_reliability[len(id_to_reliability) // 2 :]],
    )
    # save to yaml
    with open(os.path.join(args.reliable_id_path, "reliable_ids.yaml"), "w+") as outfile:
        yaml.dump(yaml_dict, outfile, default_flow_style=False)


def label(model, dataloader, args):
    model.eval()
    tbar = tqdm(dataloader)

    metric = meanIOU(
        num_classes=21
        if args.dataset == "pascal"
        else 2
        if args.dataset == "melanoma"
        else 2
        if args.dataset == "breastCancer"
        else 19
    )
    cmap = color_map(args.dataset)

    with torch.no_grad():
        for img, mask, id in tbar:
            if args.dataset == "melanoma" or args.dataset == "breastCancer":
                mask = mask.clip(max=1)  # clips max value to 1: 255 to 1
            img = img.cuda()
            pred = model(img)
            pred = torch.argmax(pred, dim=1).cpu()

            metric.add_batch(pred.numpy(), mask.numpy())
            mIOU = metric.evaluate()[-1]

            pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode="P")
            pred.putpalette(cmap)

            pred.save("%s/%s" % (args.pseudo_mask_path, os.path.basename(id[0].split(" ")[1])))

            tbar.set_description("mIOU: %.2f" % (mIOU * 100.0))


def test(model, dataloader, args):
    metric = mulitmetrics(
        num_classes=21
        if args.dataset == "pascal"
        else 2
        if args.dataset == "melanoma"
        else 2
        if args.dataset == "breastCancer"
        else 19
    )
    model.eval()
    tbar = tqdm(dataloader)
    # set i for sample images
    i = 0
    wandb_iamges = []
    torch.cuda.empty_cache()
    with torch.no_grad():
        for img, mask, _ in tbar:
            if args.dataset == "melanoma" or args.dataset == "breastCancer":
                mask = mask.clip(max=1)  # clips max value to 1: 255 to 1
            i = i + 1
            img = img.cuda()
            pred = model(img)
            pred = torch.argmax(pred, dim=1)

            metric.add_batch(pred.cpu().numpy(), mask.numpy())
            overall_acc, mIOU, mDICE, medpy_dc, medpy_jc, medpy_hd, medpy_asd = metric.evaluate()
            tbar.set_description(
                "test mIOU: %.2f, mDICE: %.2f,overall_acc: %.2f" % (mIOU * 100.0, mDICE * 100.0, overall_acc * 100.0)
            )
            if args.use_wandb:
                if i <= 10:
                    # wandb.log({"img": [wandb.Image(img, caption="img")]})
                    # wandb.log({"mask": [wandb.Image(pred.cpu().numpy(), caption="mask")]})
                    class_labeles = dict((el, "something") for el in list(range(21)))
                    class_labeles.update({255: "boarder"})
                    class_labeles.update({0: "nothing"})
                    wandb_iamge = wandb.Image(
                        cv2.resize(
                            np.moveaxis(np.squeeze(img.cpu().numpy(), axis=0), 0, -1),
                            dsize=(100, 100),
                            interpolation=cv2.INTER_NEAREST,
                        ),
                        masks={
                            "predictions": {
                                "mask_data": cv2.resize(
                                    np.squeeze(pred.cpu().numpy(), axis=0),
                                    dsize=(100, 100),
                                    interpolation=cv2.INTER_NEAREST,
                                ),
                                "class_labels": class_labeles,
                            },
                            "ground_truth": {
                                "mask_data": cv2.resize(
                                    np.squeeze(mask.numpy(), axis=0),
                                    dsize=(100, 100),
                                    interpolation=cv2.INTER_NEAREST,
                                ),
                                "class_labels": class_labeles,
                            },
                        },
                    )
                    wandb_iamges.append(wandb_iamge)
        if args.use_wandb:
            wandb.log({"Test Pictures": wandb_iamges})
            wandb.log(
                {"test mIOU": mIOU, "test mDICE": mDICE, "test overall_acc": overall_acc, "test medpy_dc": medpy_dc}
            )


class SemiDataset(Dataset):
    def __init__(
        self,
        name,
        root,
        mode,
        size,
        split_file_path=None,
        pseudo_mask_path=None,
        reliable=None,
        val_split="val_split_0",
    ):
        """
        :param name: dataset name, pascal, melanoma or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        :param split_file_path: path of yaml file for splits.
        :param pseudo_mask_path: path of generated pseudo masks, needed in semi_train mode.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        self.pseudo_mask_path = pseudo_mask_path

        if mode == "semi_train":
            with open(split_file_path, "r") as file:
                split_dict = yaml.load(file, Loader=yaml.FullLoader)[val_split]
                self.labeled_ids = split_dict["labeled"]
                if reliable is None:
                    self.unlabeled_ids = split_dict["unlabeled"]
                elif reliable is True:
                    self.unlabeled_ids = split_dict["reliable"]
                elif reliable is False:
                    self.unlabeled_ids = split_dict["unreliable"]
                # multiply label to match the cound of unlabeled
                self.ids = (
                    self.labeled_ids * math.ceil(len(self.unlabeled_ids) / len(self.labeled_ids)) + self.unlabeled_ids
                )
        elif mode == "test":
            with open(split_file_path, "r") as file:
                self.ids = yaml.load(file, Loader=yaml.FullLoader)
        else:
            with open(split_file_path) as file:
                split_dict = yaml.load(file, Loader=yaml.FullLoader)[val_split]
                if mode == "val":
                    self.ids = split_dict["val"]
                elif mode == "label":
                    if reliable is None:
                        self.ids = split_dict["unlabeled"]
                    elif reliable is True:
                        self.ids = split_dict["reliable"]
                    elif reliable is False:
                        self.ids = split_dict["unreliable"]
                elif mode == "train":
                    self.ids = split_dict["labeled"]

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(" ")[0]))

        if self.mode == "val" or self.mode == "label" or self.mode == "test":
            mask = Image.open(os.path.join(self.root, id.split(" ")[1]))
            # unet needs much memory on
            if self.name == "melanoma":
                img, mask = resize_crop(img, mask, self.size)
            img, mask = normalize(img, mask)
            # print(img.cpu().numpy().shape)
            return img, mask, id

        if self.mode == "train" or (self.mode == "semi_train" and id in self.labeled_ids):
            mask = Image.open(os.path.join(self.root, id.split(" ")[1]))
        else:
            # mode == 'semi_train' and the id corresponds to unlabeled image
            fname = os.path.basename(id.split(" ")[1])
            mask = Image.open(os.path.join(self.pseudo_mask_path, fname))

        # basic augmentation on all training images
        base_size = (
            400
            if self.name == "pascal"
            else 256
            if self.name == "melanoma"
            else 500
            if self.name == "breastCancer"
            else 2048
        )
        img, mask = resize(img, mask, base_size, (0.5, 2.0))
        img, mask = crop(img, mask, self.size)
        img, mask = hflip(img, mask, p=0.5)

        # strong augmentation on unlabeled images
        if self.mode == "semi_train" and id in self.unlabeled_ids:
            if random.random() < 0.8:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.2)(img)
            img = blur(img, p=0.5)
            img, mask = cutout(img, mask, p=0.5)

        img, mask = normalize(img, mask)

        return img, mask

    def __len__(self):
        return len(self.ids)


EPS = 1e-10


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


class meanIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist) + EPS)
        return iu, np.nanmean(iu)


def color_map(dataset="pascal"):
    cmap = np.zeros((256, 3), dtype="uint8")

    if dataset == "pascal" or dataset == "coco":

        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == "cityscapes":
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255, 0, 0])
        cmap[13] = np.array([0, 0, 142])
        cmap[14] = np.array([0, 0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0, 0, 230])
        cmap[18] = np.array([119, 11, 32])

    elif dataset == "melanoma":
        cmap[1] = np.array([255, 255, 255])

    elif dataset == "breastCancer":
        cmap[1] = np.array([255, 255, 255])

    return cmap


if __name__ == "__main__":
    args = base_parse_args(BaseModule)
    main(args)
