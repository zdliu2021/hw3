import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import models


class base_resnet(nn.Module):
    def __init__(self):
        super(base_resnet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        #self.model.load_state_dict(torch.load('./model/resnet50-19c8e357.pth'))
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        #x = self.model(x)
        return x


x = torch.rand((64, 3, 256, 256))
model = base_resnet()
y = model(x)
print(y.size())
