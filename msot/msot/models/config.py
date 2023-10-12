from dataclasses import dataclass, field

from msot.utils.config import Config

from .backbone import BackboneCfg
from .neck import AdjustCfg
from .head import MaskCfg, MaskRefineCfg, RPNCfg


@dataclass
class ModelConfig(Config):
    backbone: BackboneCfg = field(default_factory=BackboneCfg)
    mask: MaskCfg = field(default_factory=MaskCfg)
    rpn: RPNCfg = field(default_factory=RPNCfg)
    refine: MaskRefineCfg = field(default_factory=MaskRefineCfg)
    adjust: AdjustCfg = field(default_factory=AdjustCfg)

    pretrained: str | None = None
    """path to pretrained model"""
