from pysot.models.backbone.alexnet import alexnetlegacy, alexnet
from pysot.models.backbone.mobile_v2 import mobilenetv2
from pysot.models.backbone.resnet_atrous import resnet18, resnet34, resnet50
from pysot.types import VALID_BACKBONE  # TODO:


def get_backbone(name: VALID_BACKBONE, **kwargs):
    print(f'[DEBUG] get_backbone: {name}; kwargs: {kwargs}')
    BACKBONES = {
        "alexnetlegacy": alexnetlegacy,
        "mobilenetv2": mobilenetv2,
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "alexnet": alexnet,
    }
    return BACKBONES[name](**kwargs)
