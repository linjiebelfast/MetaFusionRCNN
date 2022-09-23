from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .hourglass import HourglassNet
from .hrnet import HRNet
from .regnet import RegNet
from .res2net import Res2Net
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG

from .xjjnet import XJJNet
from .xjjres import XJJRes
from .xjjres2 import XJJRes2
from .resnet_origin import OriginResNet, OriginResNetV1d


__all__ = [
    'OriginResNet', 'XJJRes2', 'XJJRes', 'XJJNet', 'RegNet', 'ResNet', 'ResNetV1d', 'OriginResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'Res2Net',
    'HourglassNet', 'DetectoRS_ResNet', 'DetectoRS_ResNeXt', 'Darknet'
]
