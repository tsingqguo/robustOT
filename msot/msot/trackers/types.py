from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple

import torch


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

    # def __copy__(self) -> ScaledCrop:
    #     ...

    # def __deepcopy__(self, memo: dict[int, object]) -> ScaledCrop:
    #     ...
