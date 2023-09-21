from dataclasses import dataclass, field
from typing import Any, Literal

from msot.utils.config import Config

VALID_ADJUST = Literal[
    "AdjustLayer",
    "AdjustAllLayer",
]


@dataclass
class AdjustCfg(Config):
    adjust: bool = True

    kwargs: dict[str, Any] = field(default_factory=dict)

    type: VALID_ADJUST = "AdjustAllLayer"
    """Adjust layer type"""
