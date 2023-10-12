from dataclasses import dataclass, field
from typing import Any, Literal

from msot.utils.config import Config
from msot.utils.option import Option, NONE

VALID_BACKBONE = Literal[
    "alexnet",
    "alexnetlegacy",
    "mobilenetv2",
    "resnet18",
    "resnet34",
    "resnet50",
]


@dataclass
class BackboneCfg(Config):
    # TODO:
    type: VALID_BACKBONE = "resnet50"
    """
    Backbone type, current only support resnet18,34,50;alexnet;mobilenet
    """

    kwargs: dict[str, Any] = field(default_factory=dict)

    pretrained: Option[str] = NONE
    """Pretrained backbone weights"""

    train_layers: list[str] = field(
        default_factory=lambda: ["layer2", "layer3", "layer4"]
    )
    """Train layers"""

    layers_lr: float = 0.1
    """Layer LR"""

    train_epoch: int = 10
    """Switch to train layer"""
