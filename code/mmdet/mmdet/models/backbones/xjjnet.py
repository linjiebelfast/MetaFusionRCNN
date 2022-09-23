import torch.nn as nn
import torch
import logging
from mmcv.runner import load_checkpoint
from ..builder import BACKBONES
from .Source import Source
from .Fusion import Fusion
from .conv2d_mtl import Conv2dMtl

@BACKBONES.register_module
class XJJNet(nn.Module):

    def __init__(self, init_weights=True, mtl=True, frozen_stages=-1, num_cls=2):
        super(XJJNet, self).__init__()
        if mtl:
            self.Conv2d = Conv2dMtl
        else:
            self.Conv2d = nn.Conv2d
        self.conv1 = nn.Sequential(
            self.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )  # 128*80*80
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128*40*40
        self.conv2 = nn.Sequential(
            self.Conv2d(128 + 0, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )  # 256*40*40
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256*20*20
        self.conv3 = nn.Sequential(
            self.Conv2d(256 + 0, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )  # 512*20*20
        self.conv4 = nn.Sequential(
            self.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )  # 512*20*20
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 512*10*10
        self.conv5 = nn.Sequential(
            self.Conv2d(512 + 0, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )  # 1024*10*10
        self.conv6 = nn.Sequential(
            self.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )  # 1024*10*10
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 1024*5*5
        self.conv7 = nn.Sequential(
            self.Conv2d(1024 + 0, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )  # 2048*5*5
        self.conv8 = nn.Sequential(
            self.Conv2d(2048, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )  # 2048*5*5

        self.avgpool = nn.AvgPool2d(5, stride=1)

        self.adaptive_fusion = Fusion(mtl)
        self.feature_extractor = Source()

        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, num_cls)
        )
        self.frozen_stages = frozen_stages
        self._freeze_stages()

        if init_weights:
            self.init_weights()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            pass  # dont know what to freeze

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(XJJNet, self).train(mode)
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

        x = self.conv1(x)
        # outs.append(x)  # 128
        x = self.pool1(x)

        # x = torch.cat([x, FS[0]], dim=1)
        x = self.conv2(x)
        outs.append(x)  # 256
        x = self.pool2(x)

        # x = torch.cat([x, FS[1]], dim=1)
        x = self.conv3(x)
        x = self.conv4(x)
        outs.append(x)  # 512
        x = self.pool3(x)

        # x = torch.cat([x, FS[2]], dim=1)
        x = self.conv5(x)
        x = self.conv6(x)
        outs.append(x)  # 1024
        x = self.pool4(x)

        # x = torch.cat([x, FS[3]], dim=1)
        x = self.conv7(x)
        x = self.conv8(x)
        outs.append(x)  # 2048

        # x = self.avgpool(x)
        # y = x.view(x.size(0), -1)
        #
        # y = self.fc(y)
        # print("======================================================")
        # print(outs)
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
