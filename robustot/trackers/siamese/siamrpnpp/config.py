from dataclasses import dataclass, field

from msot.libs.pysot.utils.anchor import AnchorsCfg

from .. import TrackConfig as BaseCfg


@dataclass
class TrackConfig(BaseCfg):
    anchor: AnchorsCfg = field(default_factory=AnchorsCfg)

    penalty_k: float = 0.04
    """Scale penalty"""

    window_influence: float = 0.44
    """Window influence"""

    lr: float = 0.4
    """Interpolation learning rate"""

    exemplar_size: int = 127
    """Exemplar size; default `127`"""

    instance_size: int = 255
    """Instance size; default `255`"""

    base_size: int = 8
    """Base size; default 8"""

    context_amount: float = 0.5
    """Context amount; default 0.5"""

    # lost_instance_size: int = 831
    # """Long term lost search size"""

    # confidence_low: float = 0.85
    # """Long term confidence low"""

    # confidence_high: float = 0.998
    # """Long term confidence high"""

    # mask_threshold: float = 0.3
    # """Mask threshold"""

    # mask_output_size: int = 127
    # """Mask output size"""
