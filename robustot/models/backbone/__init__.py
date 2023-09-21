import torch.nn as nn

from .alexnet import alexnet, alexnetlegacy
from .config import BackboneCfg
from .mobile_v2 import mobilenetv2
from .resnet_atrous import resnet18, resnet34, resnet50


def build_backbone(config: BackboneCfg) -> nn.Module:
    return {
        "alexnetlegacy": alexnetlegacy,
        "mobilenetv2": mobilenetv2,
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "alexnet": alexnet,
    }[config.type](**config.kwargs)
