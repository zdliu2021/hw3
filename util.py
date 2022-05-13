import pickle
from matplotlib import image
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import os
import torchvision.transforms as transforms



train_transformer = transforms.Compose([
    transforms.ToPILImage(),
    torchvision.transforms.CenterCrop(256),
    transforms.RandomApply(torch.nn.ModuleList([
        transforms.RandomRotation(100), transforms.Pad(10, fill=0, padding_mode='constant')]), p=0.8),
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


test_transformer = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def train_dataloader(path):
    img = Image.open(path)
    img = np.array(img)
    img = train_transformer(img)
    return img.numpy()


def test_dataloader(path):
    img = Image.open(path)
    img = np.array(img)
    img = test_transformer(img)
    return img.numpy()


def collate_fn(batch):
    x = []
    for key, value, l in batch:
        x.append(key.tolist())

    y = []
    for key, value, l in batch:
        y.extend(value)

    len = []
    for key, value, l in batch:
        len.append(l)

    X = torch.FloatTensor(x)
    Y = torch.LongTensor(y)
    Y_len = torch.LongTensor(len)

    return X, Y, Y_len


def read_path(tag):
    if tag == "train":
        dir = "dataset/train"
        paths = os.listdir(dir)
        res = []
        for path in paths:
            res.append(dir+"/"+path)

        res = sorted(res, key=lambda x: int(x.split("/")[-1].split(".")[0]))
        return res
    else:
        dir = "dataset/test"
        paths = os.listdir(dir)
        res = []
        for path in paths:
            res.append(dir+"/"+path)
        res = sorted(res, key=lambda x: int(x.split("/")[-1].split(".")[0]))
        return res


def read_label(tag):
    if tag == "train":
        return pickle.load(open("train.pkl", "rb"))["label"]
    else:
        return pickle.load(open("test.pkl", "rb"))["label"]


