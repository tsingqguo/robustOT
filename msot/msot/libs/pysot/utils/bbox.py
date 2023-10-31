import numpy as np
import numpy.typing as npt

from msot.utils.region import Bbox, Center


def get_axis_aligned_bbox(
    region: npt.NDArray[np.float_],
) -> Center[float]:
    if region.size == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(
            region[2:4] - region[4:6]
        )
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        bbox = Bbox(*region)
        cx, cy, w, h = bbox.to_center().unpack()
    return Center(cx, cy, w, h)
