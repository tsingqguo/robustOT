from dataclasses import dataclass, field
from typing import Any, Literal

from msot.utils.config import Config

VALID_MASK = Literal["MaskCorr",]

VALID_REFINE = Literal["Refine",]

VALID_RPN = Literal[
    "UPChannelRPN",
    "DepthwiseRPN",
    "MultiRPN",
]


@dataclass
class RPNCfg(Config):
    type: VALID_RPN = "MultiRPN"

    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class MaskCfg(Config):
    mask: bool = False
    """Whether to use mask generate segmentation"""

    type: VALID_MASK = "MaskCorr"

    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class MaskRefineCfg(Config):
    refine: bool = False
    """Mask refine"""

    type: VALID_REFINE = "Refine"
    """Refine type"""
