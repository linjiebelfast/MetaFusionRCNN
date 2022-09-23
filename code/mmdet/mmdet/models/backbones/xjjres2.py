import torch
import torch.nn as nn
from ..builder import BACKBONES
from mmcv.runner import load_checkpoint
import logging
from .conv2d_mtl import Conv2dMtl

class ConvBlock(nn.Module):
    def __init__(self, in_ch, k_size, filters, stride, mtl=False):
        super(ConvBlock, self).__init__()
        if mtl:
            self.Conv2d = Conv2dMtl
        else:
            self.Conv2d = nn.Conv2d
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            self.Conv2d(in_ch, F1, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            self.Conv2d(F1, F2, kernel_size=k_size, stride=1, padding=True, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            self.Conv2d(F2, F3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.shortcut_1 = self.Conv2d(in_ch, F3, kernel_size=1, stride=stride, padding=0, bias=False)
        self.batch_1 =nn.BatchNorm2d(F3)
        self.relu_1 = nn.ReLU(True)

    def forward(self, x):
        # print('x shape is:', x.shape)
        x_shortcut = self.shortcut_1(x)
        x_shortcut = self.batch_1(x_shortcut)
        # print('after shortcut, x shape is:', x_shortcut.shape)
        x = self.stage(x)
        # print('after stage, x shape is:', x.shape)
        x = x + x_shortcut
        x = self.relu_1(x)
        return x

class IndentityBlock(nn.Module):
    def __init__(self, in_ch, k_size, filters, mtl=False):
        super(IndentityBlock, self).__init__()
        if mtl:
            self.Conv2d = Conv2dMtl
        else:
            self.Conv2d = nn.Conv2d
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            self.Conv2d(in_ch, F1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            self.Conv2d(F1, F2, kernel_size=k_size, stride=1, padding=True, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            self.Conv2d(F2, F3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.relu_1 = nn.ReLU(True)

    def forward(self, x):
        x_shortcut = x
        x = self.stage(x)
        x = x + x_shortcut
        x = self.relu_1(x)
        return x


@BACKBONES.register_module
class XJJRes2(nn.Module):
    def __init__(self, init_weights=True, mtl=False, frozen_stages=-1, num_cls=2):
        super(XJJRes2, self).__init__()
        if mtl:
            self.Conv2d = Conv2dMtl
        else:
            self.Conv2d = nn.Conv2d

        self.pre_conv = self.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.pre_norm = nn.BatchNorm2d(64)
        self.pre_relu = nn.ReLU(True)
        self.pre_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage0 = nn.Sequential(
            ConvBlock(in_ch=64, k_size=3, filters=[64, 64, 256], stride=1, mtl=mtl),
            IndentityBlock(in_ch=256, k_size=3, filters=[64, 64, 256], mtl=mtl),
            IndentityBlock(in_ch=256, k_size=3, filters=[64, 64, 256], mtl=mtl),
        )

        self.stage1 = nn.Sequential(
            ConvBlock(in_ch=256, k_size=3, filters=[128, 128, 512], stride=2, mtl=mtl),
            IndentityBlock(in_ch=512, k_size=3, filters=[128, 128, 512], mtl=mtl),
            IndentityBlock(in_ch=512, k_size=3, filters=[128, 128, 512], mtl=mtl),
            IndentityBlock(in_ch=512, k_size=3, filters=[128, 128, 512], mtl=mtl),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(in_ch=512, k_size=3, filters=[256, 256, 1024], stride=2, mtl=mtl),
            IndentityBlock(in_ch=1024, k_size=3, filters=[256, 256, 1024], mtl=mtl),
            IndentityBlock(in_ch=1024, k_size=3, filters=[256, 256, 1024], mtl=mtl),
            IndentityBlock(in_ch=1024, k_size=3, filters=[256, 256, 1024], mtl=mtl),
            IndentityBlock(in_ch=1024, k_size=3, filters=[256, 256, 1024], mtl=mtl),
            IndentityBlock(in_ch=1024, k_size=3, filters=[256, 256, 1024], mtl=mtl),
        )
        self.stage3 = nn.Sequential(
            ConvBlock(in_ch=1024, k_size=3, filters=[512, 512, 2048], stride=2, mtl=mtl),
            IndentityBlock(in_ch=2048, k_size=3, filters=[512, 512, 2048], mtl=mtl),
            IndentityBlock(in_ch=2048, k_size=3, filters=[512, 512, 2048], mtl=mtl),
        )
        # self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        # self.fc = nn.Linear(8192, num_cls)

        self.frozen_stages = frozen_stages
        self._freeze_stages()

        if init_weights:
            self.init_weights()

    def forward(self, x):

        x = self.pre_conv(x)
        x = self.pre_norm(x)
        x = self.pre_relu(x)
        x = self.pre_pool(x)

        outs = []
        out = self.stage0(x)
        outs.append(out)
        out = self.stage1(out)
        outs.append(out)
        out = self.stage2(out)
        outs.append(out)
        out = self.stage3(out)
        outs.append(out)

        # out = self.pool(out)
        # out = out.view(out.size(0), 8192)
        # out = self.fc(out)
        return tuple(outs)

    def init_weights(self, pretrained=None):
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, self.Conv2d):
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

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.pre_norm.eval()
            m_list = []
            for m in [self.pre_conv, self.pre_norm]:
                for param in m.parameters():
                    param.requires_grad = False
            if self.frozen_stages == 0:
                pass

            elif self.frozen_stages == 1:
                m_list.append(self.stage0)
            elif self.frozen_stages == 2:
                m_list.append(self.stage0)
                m_list.append(self.stage1)
            elif self.frozen_stages == 3:
                m_list.append(self.stage0)
                m_list.append(self.stage1)
                m_list.append(self.stage2)
            elif self.frozen_stages == 4:
                m_list.append(self.stage0)
                m_list.append(self.stage1)
                m_list.append(self.stage2)
                m_list.append(self.stage3)
            else:
                raise AssertionError('frozen_stage only support 0 to 4!')

            for m in m_list:
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(XJJRes2, self).train(mode)
        self._freeze_stages()
        if mode and True:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
