from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .point import Point

if TYPE_CHECKING:
    from .polygon import Polygon


class Box:
    _data: npt.NDArray[np.float_]

    def __init__(self) -> None:
        ...

    @property
    def _x1(self) -> np.float_:
        raise NotImplementedError

    @property
    def _x2(self) -> np.float_:
        raise NotImplementedError

    @property
    def _y1(self) -> np.float_:
        raise NotImplementedError

    @property
    def _y2(self) -> np.float_:
        raise NotImplementedError

    @property
    def _cx(self) -> np.float_:
        raise NotImplementedError

    @property
    def _cy(self) -> np.float_:
        raise NotImplementedError

    @property
    def center(self) -> Point:
        return Point(self._cx, self._cy)

    @property
    def _h(self) -> np.float_:
        raise NotImplementedError

    @property
    def _w(self) -> np.float_:
        raise NotImplementedError

    @property
    def lt(self) -> Point:
        return Point(self._x1, self._y1)

    @property
    def rt(self) -> Point:
        return Point(self._x2, self._y1)

    @property
    def lb(self) -> Point:
        return Point(self._x1, self._y2)

    @property
    def rb(self) -> Point:
        return Point(self._x2, self._y2)

    @property
    def vertices(self):
        return [
            self.lt,
            self.rt,
            self.rb,
            self.lb,
        ]

    @property
    def size(self) -> npt.NDArray[np.float_]:
        return np.array([self._w, self._h])

    @property
    def area(self) -> float:
        return float(self._w * self._h)

    def get_overlap_ratio(self, other: Box) -> float:
        # TODO: bounds impl
        x_a = max(float(self._x1), float(other._x1))
        y_a = max(float(self._y1), float(other._y1))
        x_b = min(
            float(self._x1) + float(self._w),
            float(other._x1 + float(other._w)),
        )
        y_b = min(
            float(self._y1) + float(self._h),
            float(other._y1 + float(other._h)),
        )
        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
        box_a_area = (float(self._w) + 1) * (float(self._h) + 1)
        box_b_area = (float(other._w) + 1) * (float(other._h) + 1)
        return inter_area / float(box_a_area + box_b_area - inter_area)

    def __array__(self) -> npt.NDArray[np.float_]:
        return self._data

    def numpy(self) -> npt.NDArray[np.float_]:
        return self._data

    def unpack(self):
        raise NotImplementedError

    def to_bbox(self) -> Bbox:
        return Bbox(self._x1, self._y1, self._w, self._h)

    def to_center(self) -> Center:
        return Center(self._cx, self._cy, self._w, self._h)

    def to_corner(self) -> Corner:
        return Corner(self._x1, self._y1, self._x2, self._y2)

    def to_polygon(self) -> Polygon:
        from .polygon import Polygon

        return Polygon(self.vertices)


class Corner(Box):
    def __init__(self, x1, y1, x2, y2) -> None:
        super().__init__()
        self._data = np.array([x1, y1, x2, y2], dtype=np.float_)

    def __str__(self) -> str:
        return f"Corner {{ x1: {self._x1}, y1: {self._x2}, x2: {self._y1}, y2: {self._y2} }}"

    @property
    def _x1(self) -> np.float_:
        return self._data[0]

    @property
    def _x2(self) -> np.float_:
        return self._data[2]

    @property
    def _y1(self) -> np.float_:
        return self._data[1]

    @property
    def _y2(self) -> np.float_:
        return self._data[3]

    @property
    def _cx(self) -> np.float_:
        return (self._x1 + self._x2) / 2

    @property
    def _cy(self) -> np.float_:
        return (self._y1 + self._y2) / 2

    @property
    def _h(self) -> np.float_:
        return self._y2 - self._y1

    @property
    def _w(self) -> np.float_:
        return self._x2 - self._x1

    @property
    def x1(self) -> float:
        return float(self._x1)

    @property
    def x2(self) -> float:
        return float(self._x2)

    @property
    def y1(self) -> float:
        return float(self._y1)

    @property
    def y2(self) -> float:
        return float(self._y2)

    def unpack(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


class Bbox(Box):
    def __init__(self, x1, y1, w, h) -> None:
        super().__init__()
        self._data = np.array([x1, y1, w, h], dtype=np.float_)

    def __str__(self) -> str:
        return f"Bbox {{ x1: {self._x1}, y1: {self._y1}, w: {self._w}, h: {self._h} }}"

    @property
    def _x1(self) -> np.float_:
        return self._data[0]

    @property
    def _x2(self) -> np.float_:
        return self._data[0] + self._data[2]

    @property
    def _y1(self) -> np.float_:
        return self._data[1]

    @property
    def _y2(self) -> np.float_:
        return self._data[1] + self._data[3]

    @property
    def _cx(self) -> np.float_:
        return self._data[0] + self._data[2] / 2

    @property
    def _cy(self) -> np.float_:
        return self._data[1] + self._data[3] / 2

    @property
    def _h(self) -> np.float_:
        return self._data[3]

    @property
    def _w(self) -> np.float_:
        return self._data[2]

    @property
    def x1(self) -> float:
        return float(self._x1)

    @property
    def y1(self) -> float:
        return float(self._y1)

    @property
    def w(self) -> float:
        return float(self._w)

    @property
    def h(self) -> float:
        return float(self._h)

    def unpack(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.w, self.h


class Center(Box):
    def __init__(self, cx, cy, w, h) -> None:
        super().__init__()
        self._data = np.array([cx, cy, w, h], dtype=np.float_)

    def __str__(self) -> str:
        return f"Center {{ cx: {self._cx}, cy: {self._cy}, w: {self._w}, h: {self._h} }}"

    @property
    def _x1(self) -> np.float_:
        return self._data[0] - self._data[2] / 2

    @property
    def _x2(self) -> np.float_:
        return self._data[0] + self._data[2] / 2

    @property
    def _y1(self) -> np.float_:
        return self._data[1] - self._data[3] / 2

    @property
    def _y2(self) -> np.float_:
        return self._data[1] + self._data[3] / 2

    @property
    def _cx(self) -> np.float_:
        return self._data[0]

    @property
    def _cy(self) -> np.float_:
        return self._data[1]

    @property
    def _h(self) -> np.float_:
        return self._data[3]

    @property
    def _w(self) -> np.float_:
        return self._data[2]

    @property
    def cx(self) -> float:
        return float(self._cx)

    @property
    def cy(self) -> float:
        return float(self._cy)

    @property
    def w(self) -> float:
        return float(self._w)

    @property
    def h(self) -> float:
        return float(self._h)

    def unpack(self) -> tuple[float, float, float, float]:
        return self.cx, self.cy, self.w, self.h

    def __sub__(self, other: Point) -> Center:
        pt = np.array(other)
        return Center(self._cx - pt[0], self.cy - pt[1], self.w, self.h)

    def __add__(self, other: Point) -> Center:
        pt = np.array(other)
        return Center(self.cx + pt[0], self.cy + pt[1], self.w, self.h)
