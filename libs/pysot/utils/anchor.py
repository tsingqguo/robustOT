import math
import numpy as np
import numpy.typing as npt
from pysot.utils.bbox import corner2center, center2corner
from typing import List, Optional, Tuple


class Anchors:
    stride: int
    ratios: List[float]
    scales: List[int]
    image_center: int
    anchor_num: int

    anchors: Optional[npt.NDArray[np.float32]]
    """shape in (anchor_num, 4)"""

    all_anchors: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]

    def __init__(
        self,
        stride: int,
        ratios: List[float],
        scales: List[int],
        image_center: int = 0,
        size: int = 0,
    ):
        self.stride = stride
        self.ratios = ratios
        self.scales = scales
        self.image_center = image_center
        self.size = size

        self.anchor_num = len(self.scales) * len(self.ratios)

        self.anchors = None

        self.generate_anchors()

    def generate_anchors(self) -> None:
        """
        generate anchors based on predefined configuration
        """
        self.anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)
        size = self.stride * self.stride
        count = 0
        for ratio in self.ratios:
            ws = int(math.sqrt(size * 1.0 / ratio))
            hs = int(ws * ratio)

            for s in self.scales:
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

        a0x = im_c - size // 2 * self.stride
        ori = np.array([a0x] * 4, dtype=np.float32)
        zero_anchors = self.anchors + ori

        x1 = zero_anchors[:, 0]
        y1 = zero_anchors[:, 1]
        x2 = zero_anchors[:, 2]
        y2 = zero_anchors[:, 3]

        x1, y1, x2, y2 = map(
            lambda x: x.reshape(self.anchor_num, 1, 1), [x1, y1, x2, y2]
        )
        cx, cy, w, h = corner2center([x1, y1, x2, y2])

        disp_x = np.arange(0, size).reshape(1, 1, -1) * self.stride
        disp_y = np.arange(0, size).reshape(1, -1, 1) * self.stride

        cx = cx + disp_x
        cy = cy + disp_y

        # broadcast
        zero = np.zeros((self.anchor_num, size, size), dtype=np.float32)
        cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])
        x1, y1, x2, y2 = center2corner([cx, cy, w, h])

        self.all_anchors = (
            np.stack([x1, y1, x2, y2]).astype(np.float32),
            np.stack([cx, cy, w, h]).astype(np.float32),
        )
        return True
