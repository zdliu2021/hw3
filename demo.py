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
from model import CRNN,Pretrained_CRNN

model = Pretrained_CRNN()

x = torch.rand((64,3,256,256))
y = model(x)
print(y.size())
