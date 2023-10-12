from typing import Literal

import torch.nn as nn

from .config import AdjustCfg
from .neck import AdjustLayer, AdjustAllLayer


def build_neck(cfg: AdjustCfg) -> nn.Module:
    return {
        "AdjustLayer": AdjustLayer,
        "AdjustAllLayer": AdjustAllLayer,
    }[cfg.type](**cfg.kwargs)
