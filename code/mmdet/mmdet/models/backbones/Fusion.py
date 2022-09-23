import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv2d_mtl import Conv2dMtl


class Fusion(nn.Module):
    def __init__(self, mtl=False, init_weights=True):
        super(Fusion, self).__init__()
        if mtl:
            self.Conv2d = Conv2dMtl
        else:
            self.Conv2d = nn.Conv2d
        self.conversion_4_64 = nn.Sequential(
            self.Conv2d(in_channels=2 * 64,out_channels=5,kernel_size=1,stride=1),
            nn.BatchNorm2d(5)
        )
        self.conversion_4_128 = nn.Sequential(
            self.Conv2d(in_channels=2 * 128, out_channels=5, kernel_size=1, stride=1),
            nn.BatchNorm2d(5)
        )
        self.conversion_7_256 = nn.Sequential(
            self.Conv2d(in_channels=3 * 256, out_channels=5, kernel_size=1, stride=1),
            nn.BatchNorm2d(5)
        )
        self.conversion_7_512_10 = nn.Sequential(
            self.Conv2d(in_channels=3 * 512, out_channels=5, kernel_size=1, stride=1),
            nn.BatchNorm2d(5)
        )
        self.conversion_7_512_5 = nn.Sequential(
            self.Conv2d(in_channels=3 * 512, out_channels=5, kernel_size=1, stride=1),
            nn.BatchNorm2d(5)
        )
        self.integration = nn.Sequential(
            self.Conv2d(5,5,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(5),
            nn.ReLU(inplace=True)
        )

        self.integration_25 = nn.Sequential(
            self.Conv2d(25,5,kernel_size=1,stride=1),
            nn.BatchNorm2d(5)
        )
        self.integration_20 = nn.Sequential(
            self.Conv2d(20, 5, kernel_size=1, stride=1),
            nn.BatchNorm2d(5)
        )
        self.integration_15 = nn.Sequential(
            self.Conv2d(15, 5, kernel_size=1, stride=1),
            nn.BatchNorm2d(5)
        )
        self.integration_10 = nn.Sequential(
            self.Conv2d(10, 5, kernel_size=1, stride=1),
            nn.BatchNorm2d(5)
        )
        self.integration_5 = nn.Sequential(
            self.Conv2d(5, 5, kernel_size=1, stride=1),
            nn.BatchNorm2d(5)
        )
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
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

    def forward(self, unit):
        # print(len(unit))
        for feature in unit:
            print('after Source the size is : ', feature.shape)
            if feature.shape[1] == 2*64:
                f1 = self.conversion_4_64(feature)
                f1 = self.integration(f1)
            elif feature.shape[1] == 2*128:
                f2 = self.conversion_4_128(feature)
                f2 = self.integration(f2)
            elif feature.shape[1] == 3*256:
                f3 = self.conversion_7_256(feature)
                f3 = self.integration(f3)
            elif feature.shape[1] == 3*512 and feature.shape[-1] == 96 or feature.shape[-1] == 72:
                f4 = self.conversion_7_512_10(feature)
                f4 = self.integration(f4)
            elif feature.shape[1] == 3*512 and feature.shape[-1] == 48 or feature.shape[-1] == 36:
                f5 = self.conversion_7_512_5(feature)
                f5 = self.integration(f5)
            else:
                print('feature size is:', feature.shape)
                print('feature size has problem!')
        f54 = F.interpolate(f5, scale_factor=2, mode='nearest')
        f53 = F.interpolate(f54, scale_factor=2, mode='nearest')
        f52 = F.interpolate(f53, scale_factor=2, mode='nearest')
        f51 = F.interpolate(f52, scale_factor=2, mode='nearest')

        f43 = F.interpolate(f4, scale_factor=2, mode='nearest')
        f42 = F.interpolate(f43, scale_factor=2, mode='nearest')
        f41 = F.interpolate(f42, scale_factor=2, mode='nearest')

        f32 = F.interpolate(f3, scale_factor=2, mode='nearest')
        f31 = F.interpolate(f32, scale_factor=2, mode='nearest')

        f21 = F.interpolate(f2, scale_factor=2, mode='nearest')

        # F1 = torch.cat([f1,f21,f31,f41,f51], dim=1)  # 25
        F2 = torch.cat([f2,f32,f42,f52],dim=1)  # 20
        F3 = torch.cat([f3,f43,f53],dim=1)  # 15
        F4 = torch.cat([f4,f54],dim=1)  # 10
        F5 = f5

        # F1 = self.integration_25(F1)
        F2 = self.integration_20(F2)
        F3 = self.integration_15(F3)
        F4 = self.integration_10(F4)
        F5 = self.integration_5(F5)
        # print(F1.shape)

        # F1 = F.relu(torch.add(f1, F1))
        F2 = F.relu(torch.add(f2, F2))
        F3 = F.relu(torch.add(f3, F3))
        F4 = F.relu(torch.add(f4, F4))
        F5 = F.relu(torch.add(f5, F5))

        # featureSet = [F1,F2,F3,F4,F5]
        featureSet = [F2, F3, F4, F5]

        return featureSet







