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
from util import *



class dataset(Dataset):
    def __init__(self, file, label, loader):
        #定义好 image 的路径
        self.images = file
        self.target = label
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img, target,len(target)

    def __len__(self):
        return len(self.images)

train_data = dataset(read_path("train"),read_label("train"),train_dataloader)
test_data = dataset(read_path("test"),read_label("test"),test_dataloader)


train_loader = DataLoader(train_data, batch_size=128,
                         shuffle=True, collate_fn=collate_fn,num_workers=8)
val_loader = DataLoader(test_data, batch_size=128,
                        shuffle=True, collate_fn=collate_fn,num_workers=8)


# data = iter(train_loader)
# print(data.next()[0].size())
