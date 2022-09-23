import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url

class Source(nn.Module):

    def __init__(self, init=True):
        super(Source, self).__init__()
        self.source_1 = VGG16()
        self.source_2 = VGG19()
        if init:
            for p in self.parameters():
                p.requires_grad = False
            self.weights_init()

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, VGG16):
                path_1 = './vgg16_bn-6c64b313.pth'
                pretrained_dict_1 = torch.load(path_1)
                keys = []
                for k, v in pretrained_dict_1.items():
                    keys.append(k)
                i = 0
                model_dict_1 = m.state_dict()
                for k, v in model_dict_1.items():
                    if v.size() == pretrained_dict_1[keys[i]].size():
                        model_dict_1[k] = pretrained_dict_1[keys[i]]
                        i = i + 1
                m.load_state_dict(model_dict_1)
            if isinstance(m, VGG19):
                path_2 = './vgg19_bn-c79401a0.pth'
                pretrained_dict_2 = torch.load(path_2)
                keys = []
                for k, v in pretrained_dict_2.items():
                    keys.append(k)
                i = 0
                model_dict_2 = m.state_dict()
                for k, v in model_dict_2.items():
                    if v.size() == pretrained_dict_2[keys[i]].size():
                        model_dict_2[k] = pretrained_dict_2[keys[i]]
                        i = i + 1
                m.load_state_dict(model_dict_2)

                # for k, v in self.source_2.state_dict().items():
                #     print(k)
                #     print(v)

    def forward(self, x):
        # print(x.shape)
        units1 = self.source_1(x)
        # print(units1[4].shape)
        # units2 = self.source_2(x)
        # print(units2[3].shape)
        units = []
        # if len(units1) == len(units2):
        #     for i in range(0, len(units1)):
        #         units.append(torch.cat([units1[i], units2[i]], dim=0))
        # else:
        #     units = units1
        # print(units1[3].shape)
        return units1


class VGG19(nn.Module):

    def __init__(self, num_classes=1000, init_weights=False):
        super(VGG19, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # 64*224*224
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # 64*224*224
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 64*112*112

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )  # 128*112*112
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )  # 128*112*112
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 128*56*56

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )  # 256*56*56
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )  # 256*56*56
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )  # 256*56*56
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )  # 256*56*56
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 256*28*28

        self.conv9 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )  # 512*28*28
        self.conv10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )  # 512*28*28
        self.conv11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )  # 512*28*28
        self.conv12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )  # 512*28*28
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 512*14*14

        self.conv13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )  # 512*14*14
        self.conv14 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )  # 512*14*14
        self.conv15 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )  # 512*14*14
        self.conv16 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )  # 512*14*14
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 512*7*7

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        unit1 = torch.cat([f1, f2], dim=1)  # 2x64*80*80
        pool1 = self.pool1(f2)

        f3 = self.conv3(pool1)
        f4 = self.conv4(f3)
        unit2 = torch.cat([f3, f4], dim=1)  # 2x128*40*40
        pool2 = self.pool2(f4)

        f5 = self.conv5(pool2)
        f6 = self.conv6(f5)
        f7 = self.conv7(f6)
        f8 = self.conv8(f7)
        unit3 = torch.cat([f5, f6, f7, f8], dim=1)  # 4x256*20*20
        pool3 = self.pool3(f8)

        f9 = self.conv9(pool3)
        f10 = self.conv10(f9)
        f11 = self.conv11(f10)
        f12 = self.conv12(f11)
        unit4 = torch.cat([f9, f10, f11, f12], dim=1)  # 4x512*10*10
        pool4 = self.pool4(f12)

        f13 = self.conv13(pool4)
        f14 = self.conv14(f13)
        f15 = self.conv15(f14)
        f16 = self.conv16(f15)
        unit5 = torch.cat([f13, f14, f15, f16], dim=1)  # 4x512*5*5

        return [unit1, unit2, unit3, unit4, unit5]
        # return []

class VGG16(nn.Module):

    def __init__(self, num_classes=1000, init_weights=False):
        super(VGG16, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # 64*224*224
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # 64*224*224
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 64*112*112

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )  # 128*112*112
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )  # 128*112*112
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 128*56*56

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )  # 256*56*56
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )  # 256*56*56
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )  # 256*56*56
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 256*28*28

        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )  # 512*28*28
        self.conv9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )  # 512*28*28
        self.conv10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )  # 512*28*28
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 512*14*14

        self.conv11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )  # 512*14*14
        self.conv12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )  # 512*14*14
        self.conv13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )  # 512*14*14
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 512*7*7

    def forward(self, x):

        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        # print(f1.shape)
        # print(f2.shape)
        unit1 = torch.cat([f1, f2], dim=1)  # 2x64*576*768
        # print(unit1.shape)
        pool1 = self.pool1(f2)

        f3 = self.conv3(pool1)
        f4 = self.conv4(f3)
        unit2 = torch.cat([f3, f4], dim=1)  # 2x128*288*384
        pool2 = self.pool2(f4)

        f5 = self.conv5(pool2)
        f6 = self.conv6(f5)
        f7 = self.conv7(f6)
        unit3 = torch.cat([f5, f6, f7], dim=1)  # 3x256*144*192
        pool3 = self.pool3(f7)

        f8 = self.conv8(pool3)
        f9 = self.conv9(f8)
        f10 = self.conv10(f9)
        unit4 = torch.cat([f8, f9, f10], dim=1)  # 3x512*72*96
        pool4 = self.pool4(f10)

        f11 = self.conv11(pool4)
        f12 = self.conv12(f11)
        f13 = self.conv13(f12)
        unit5 = torch.cat([f11, f12, f13], dim=1)  # 3x512*36*48

        return [unit2, unit3, unit4, unit5]

