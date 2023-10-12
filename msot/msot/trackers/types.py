from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import torch


class Backends(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class TrackSize(NamedTuple):
    """size of template and search region in original image"""

    z_size: int
    """size of ~~template~~target in original image"""

    x_size: int
    """size of search region in original image"""

    scale: float
    """ratio of tracking size to original size"""


@dataclass
class ScaledCrop:
    crop: torch.Tensor

    size: TrackSize
