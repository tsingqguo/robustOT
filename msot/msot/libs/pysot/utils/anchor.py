import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from msot.utils.config import Config
from msot.utils.region import Corner, Center


@dataclass
class AnchorsCfg(Config):
    stride: int = 8
    """Anchor stride"""

    ratios: list[float] = field(default_factory=lambda: [0.33, 0.5, 1, 2, 3])
    """Anchor ratios"""

    scales: list[int] = field(default_factory=lambda: [8])
    """Anchor scales"""

    @property
    def anchor_num(self) -> int:
        """Anchor number"""
        return len(self.ratios) * len(self.scales)


class Anchors:
    config: AnchorsCfg
    image_center: int
    anchor_num: int

    anchors: npt.NDArray[np.float32]
    """shape in (anchor_num, 4)"""

    all_anchors: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]

    def __init__(
        self,
        config: AnchorsCfg,
        image_center: int = 0,
        size: int = 0,
    ):
        self.config = config
        self.image_center = image_center
        self.size = size

        self.anchor_num = len(self.config.scales) * len(self.config.ratios)

        self.generate_anchors()

    def generate_anchors(self) -> None:
        """
        generate anchors based on predefined configuration
        """
        self.anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)
        size = self.config.stride * self.config.stride
        count = 0
        for ratio in self.config.ratios:
            ws = int(math.sqrt(size * 1.0 / ratio))
            hs = int(ws * ratio)

            for s in self.config.scales:
                w = ws * s
                h = hs * s
                self.anchors[count][:] = [
                    -w * 0.5,
                    -h * 0.5,
                    w * 0.5,
                    h * 0.5,
                ][:]
                count += 1

    def generate_all_anchors(self, im_c: int, size: int) -> bool:
        """
        im_c: image center
        size: image size
        """
        if self.image_center == im_c and self.size == size:
            return False
        self.image_center = im_c
        self.size = size

        a0x = im_c - size // 2 * self.config.stride
        ori = np.array([a0x] * 4, dtype=np.float32)
        zero_anchors = self.anchors + ori

        x1 = zero_anchors[:, 0]
        y1 = zero_anchors[:, 1]
        x2 = zero_anchors[:, 2]
        y2 = zero_anchors[:, 3]

        x1, y1, x2, y2 = map(
            lambda x: x.reshape(self.anchor_num, 1, 1), [x1, y1, x2, y2]
        )
        cx, cy, w, h = Corner(x1, y1, x2, y2).to_center().unpack()

        disp_x = np.arange(0, size).reshape(1, 1, -1) * self.config.stride
        disp_y = np.arange(0, size).reshape(1, -1, 1) * self.config.stride

        cx = cx + disp_x
        cy = cy + disp_y

        # broadcast
        zero = np.zeros((self.anchor_num, size, size), dtype=np.float32)
        cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])
        x1, y1, x2, y2 = Center(cx, cy, w, h).to_corner().unpack()

        self.all_anchors = (
            np.stack([x1, y1, x2, y2]).astype(np.float32),
            np.stack([cx, cy, w, h]).astype(np.float32),
        )
        return True
