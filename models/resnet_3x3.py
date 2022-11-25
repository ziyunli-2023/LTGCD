from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import  transforms
import pickle
import os.path
import datetime
import numpy as np


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1    = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2   = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3   = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4   = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear   = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        #print(x.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        #print(out.shape)
        out = self.layer1(out)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = self.layer3(out)
        #print(out.shape)
        out = self.layer4(out)
        #print(out.shape)
        out = F.avg_pool2d(out, 4)
        #print(out.shape)
        large = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.linear(large)
        #print(out.shape)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        self.is_padding = 0
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.AvgPool2d(2)
            if in_planes != self.expansion*planes:
                self.is_padding = 1

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.is_padding:
            shortcut = self.shortcut(x)
            out += torch.cat([shortcut,torch.zeros(shortcut.shape).type(torch.cuda.FloatTensor)],1)
        else:
            out += self.shortcut(x)

        out = F.relu(out)
        return out

# def _resnet(
#     arch: str,
#     block: Type[Union[BasicBlock, Bottleneck]],
#     layers: List[int],
#     pretrained: bool,
#     progress: bool,
#     **kwargs: Any
# ) -> ResNet:
#     model = ResNet(block, layers, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls[arch],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model
#
# def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
#     r"""ResNet-18 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
#                    **kwargs)

if __name__ == '__main__':

    from torch.nn.parameter import Parameter
    device = torch.device('cuda')
    num_classes = 100
    model = ResNet(BasicBlock, [2,2,2,2],num_classes) ## 实例
    model.center= Parameter(torch.Tensor(num_classes, num_classes))
    torch.nn.init.xavier_normal_(model.center.data)
    model = model.to(device)
    print(model)
    y = model(Variable(torch.randn(256,3,32,32).to(device)))
    print(y.size())

