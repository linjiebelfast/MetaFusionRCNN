import torch.nn as nn
import torch
import logging
from mmcv.runner import load_checkpoint
from ..builder import BACKBONES
from .Source import Source
from .Fusion import Fusion
from .conv2d_mtl import Conv2dMtl
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    # 实现子module:Residual Block
    def __init__(self, in_ch, out_ch, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

@BACKBONES.register_module
class XJJRes(nn.Module):
    # 实现主module:ResNet34
    def __init__(self, init_weights=True, mtl=False, frozen_stages=-1, num_cls=2):
        super(XJJRes, self).__init__()
        if mtl:
            self.Conv2d = Conv2dMtl
        else:
            self.Conv2d = nn.Conv2d
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 重复的layer，分别为3,4,6,3个residual block
        self.layer1 = self.make_layer(64, 256, block_num=3)
        # layer1层输入输出一样，make_layer里应该不用对shortcut进行处理，但是为了统一操作
        self.layer2 = self.make_layer(256, 512, block_num=4, stride=2)
        # 第一个stride=2，剩下3个stride=1
        self.layer3 = self.make_layer(512, 1024, block_num=6, stride=2)
        self.layer4 = self.make_layer(1024, 2048, block_num=3, stride=2)

        self.fc = nn.Linear(2048, num_cls)

        self.frozen_stages = frozen_stages
        self._freeze_stages()

        if init_weights:
            self.init_weights()

    def make_layer(self, in_ch, out_ch, block_num, stride=1):
        # 当维度增加时，对shortcut进行option B的处理
        shortcut = nn.Sequential(
            # 首个Residual需要进行option B处理
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
            # 1x1卷积用于增加维度，stride=2用于减半size，为简化不考虑偏差
            nn.BatchNorm2d(out_ch)
        )
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(out_ch, out_ch))
            # 后面几个ResidualBlock,shortcut直接相加
        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.pre.eval()
            for param in self.pre.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(XJJRes, self).train(mode)
        self._freeze_stages()
        if mode:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def forward(self, x):
        # print('original input size is : ', x.shape)
        # units = self.feature_extractor(x)
        # FS = self.adaptive_fusion(units)
        outs = []
        x = self.pre(x)  # 64
        x = self.layer1(x)  # 256
        outs.append(x)
        x = self.layer2(x)  # 512
        outs.append(x)
        x = self.layer3(x)  # 1024
        outs.append(x)
        x = self.layer4(x)  # 2048
        outs.append(x)

        return tuple(outs)

    def init_weights(self, pretrained=None):
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
        elif isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        else:
            raise TypeError('pretrained must be a str or None!')
