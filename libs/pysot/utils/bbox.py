import numpy as np
import numpy.typing as npt
from typing import List, NamedTuple, Tuple, Union, overload


class Corner(NamedTuple):
    x1: float
    y1: float
    x2: float
    y2: float


BBox = Corner


class Center(NamedTuple):
    x: float
    y: float
    w: float
    h: float


@overload
def corner2center(corner: Corner) -> Center:
    ...


@overload
def corner2center(corner: List[float]) -> Tuple[float, float, float, float]:
    ...


def corner2center(corner: Union[Corner, List[float]]):
    """convert (x1, y1, x2, y2) to (cx, cy, w, h)
    Args:
        conrner: Corner or np.array (4*N)
    Return:
        Center or np.array (4 * N)
    """
    if isinstance(corner, Corner):
        x1, y1, x2, y2 = corner
        return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1))
    else:
        x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1
        return x, y, w, h


@overload
def center2corner(center: Center) -> Corner:
    ...


@overload
def center2corner(center: List[float]) -> Tuple[float, float, float, float]:
    ...


def center2corner(center: Union[Center, List[float]]):
    """convert (cx, cy, w, h) to (x1, y1, x2, y2)
    Args:
        center: Center or np.array (4 * N)
    Return:
        center or np.array (4 * N)
    """
    if isinstance(center, Center):
        x, y, w, h = center
        return Corner(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5)
    else:
        x, y, w, h = center[0], center[1], center[2], center[3]
        x1 = x - w * 0.5
        y1 = y - h * 0.5
        x2 = x + w * 0.5
        y2 = y + h * 0.5
        return x1, y1, x2, y2


def IoU(
    rect1: List[npt.NDArray[np.float32]],
    rect2: Corner,
) -> npt.NDArray[np.float32]:
    """caculate interection over union
    Args:
        rect1: (x1, y1, x2, y2)
        rect2: (x1, y1, x2, y2)
    Returns:
        iou
    """
    # overlap
    x1, y1, x2, y2 = rect1[0], rect1[1], rect1[2], rect1[3]
    tx1, ty1, tx2, ty2 = rect2[0], rect2[1], rect2[2], rect2[3]

    xx1: npt.NDArray[np.float32] = np.maximum(tx1, x1)
    yy1: npt.NDArray[np.float32] = np.maximum(ty1, y1)
    xx2: npt.NDArray[np.float32] = np.minimum(tx2, x2)
    yy2: npt.NDArray[np.float32] = np.minimum(ty2, y2)

    ww: npt.NDArray[np.float32] = np.maximum(0, xx2 - xx1)
    hh: npt.NDArray[np.float32] = np.maximum(0, yy2 - yy1)

    area = (x2 - x1) * (y2 - y1)
    target_a: float = (tx2 - tx1) * (ty2 - ty1)
    inter = ww * hh # shape in (5, 25, 25)
    iou = inter / (area + target_a - inter)
    return iou


def cxy_wh_2_rect(pos, sz):
    """convert (cx, cy, w, h) to (x1, y1, w, h), 0-index"""
    return np.array([pos[0] - sz[0] / 2, pos[1] - sz[1] / 2, sz[0], sz[1]])


def rect_2_cxy_wh(rect):
    """convert (x1, y1, w, h) to (cx, cy, w, h), 0-index"""
    return np.array([rect[0] + rect[2] / 2, rect[1] + rect[3] / 2]), np.array(
        [rect[2], rect[3]]
    )


def cxy_wh_2_rect1(pos, sz):
    """convert (cx, cy, w, h) to (x1, y1, w, h), 1-index"""
    return np.array(
        [pos[0] - sz[0] / 2 + 1, pos[1] - sz[1] / 2 + 1, sz[0], sz[1]]
    )


def rect1_2_cxy_wh(rect):
    """convert (x1, y1, w, h) to (cx, cy, w, h), 1-index"""
    return np.array(
        [rect[0] + rect[2] / 2 - 1, rect[1] + rect[3] / 2 - 1]
    ), np.array([rect[2], rect[3]])


def get_axis_aligned_bbox(
    region: npt.NDArray[np.float64],
) -> tuple[float, float, float, float]:
    """convert region to (cx, cy, w, h) that represent by axis aligned box"""
    nv = region.size
    if nv == 8:
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
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x + w / 2
        cy = y + h / 2
    return cx, cy, w, h


def get_min_max_bbox(region):
    """convert region to (cx, cy, w, h) that represent by mim-max box"""
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        w = x2 - x1
        h = y2 - y1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x + w / 2
        cy = y + h / 2
    return cx, cy, w, h
