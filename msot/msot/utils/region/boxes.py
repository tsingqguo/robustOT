from __future__ import annotations
from typing import Any, Callable, Generic, Type, TypeVar

from .point import ValidType, Point

T = TypeVar("T", bound=ValidType)
_BD = Any


class Box(Generic[T]):
    _dt: Type[T] | None

    def __init__(self, dtype: Type[T] | None = None) -> None:
        self._dt = dtype

    @property
    def x1(self) -> T:
        raise NotImplementedError

    @property
    def x2(self) -> T:
        raise NotImplementedError

    @property
    def y1(self) -> T:
        raise NotImplementedError

    @property
    def y2(self) -> T:
        raise NotImplementedError

    @property
    def cx(self) -> T:
        raise NotImplementedError

    @property
    def cy(self) -> T:
        raise NotImplementedError

    @property
    def center(self) -> Point[T]:
        return Point(self.cx, self.cy)

    @property
    def h(self) -> T:
        raise NotImplementedError

    @property
    def w(self) -> T:
        raise NotImplementedError

    @property
    def lt(self) -> Point[T]:
        return Point(self.x1, self.y1)

    @property
    def rt(self) -> Point[T]:
        return Point(self.x2, self.y1)

    @property
    def lb(self) -> Point[T]:
        return Point(self.x1, self.y2)

    @property
    def rb(self) -> Point[T]:
        return Point(self.x2, self.y2)

    @property
    def vertices(self):
        return [
            self.lt,
            self.rt,
            self.rb,
            self.lb,
        ]

    @property
    def area(self) -> float:
        return float(self.w * self.h)

    def get_overlap_ratio(self, other: Box) -> float:
        # TODO: bounds impl
        x_a = max(float(self.x1), float(other.x1))
        y_a = max(float(self.y1), float(other.y1))
        x_b = min(
            float(self.x1) + float(self.w),
            float(other.x1 + float(other.w)),
        )
        y_b = min(
            float(self.y1) + float(self.h),
            float(other.y1 + float(other.h)),
        )
        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
        box_a_area = (float(self.w) + 1) * (float(self.h) + 1)
        box_b_area = (float(other.w) + 1) * (float(other.h) + 1)
        return inter_area / float(box_a_area + box_b_area - inter_area)

    def unpack(self):
        raise NotImplementedError

    def to_bbox(self) -> Bbox[T]:
        return Bbox[T](self.x1, self.y1, self.w, self.h, self._dt)  # type: ignore

    def to_center(self) -> Center[T]:
        return Center[T](self.cx, self.cy, self.w, self.h, self._dt)  # type: ignore

    def to_corner(self) -> Corner[T]:
        return Corner[T](self.x1, self.y1, self.x2, self.y2, self._dt)  # type: ignore

    @staticmethod
    def _cvt(t: Callable[[Any], T] | None, val) -> T:
        if t is None:
            return val
        return t(val)


class Corner(Box[T]):
    _x1: _BD
    _y1: _BD
    _x2: _BD
    _y2: _BD

    def __init__(
        self,
        x1: _BD,
        y1: _BD,
        x2: _BD,
        y2: _BD,
        dtype: Type[T] = float,
    ) -> None:
        super().__init__(dtype)
        self._x1 = x1
        self._y1 = y1
        self._x2 = x2
        self._y2 = y2

    def __str__(self) -> str:
        return f"Corner {{ x1: {self.x1}, y1: {self.x2}, x2: {self.y1}, y2: {self.y2} }}"

    @property
    def x1(self) -> T:
        return self._cvt(self._dt, self._x1)

    @property
    def x2(self) -> T:
        return self._cvt(self._dt, self._x2)

    @property
    def y1(self) -> T:
        return self._cvt(self._dt, self._y1)

    @property
    def y2(self) -> T:
        return self._cvt(self._dt, self._y2)

    @property
    def cx(self) -> T:
        return self._cvt(self._dt, (self._x1 + self._x2) / 2)

    @property
    def cy(self) -> T:
        return self._cvt(self._dt, (self._y1 + self._y2) / 2)

    @property
    def h(self) -> T:
        return self._cvt(self._dt, self._y2 - self._y1)

    @property
    def w(self) -> T:
        return self._cvt(self._dt, self._x2 - self._x1)

    def unpack(self) -> tuple[T, T, T, T]:
        return self.x1, self.y1, self.x2, self.y2


class Bbox(Box[T]):
    _x1: _BD
    _y1: _BD
    _w: _BD
    _h: _BD

    def __init__(
        self,
        x1: _BD,
        y1: _BD,
        w: _BD,
        h: _BD,
        dtype: Type[T] = float,
    ) -> None:
        super().__init__(dtype)
        self._x1 = x1
        self._y1 = y1
        self._w = w
        self._h = h

    def __str__(self) -> str:
        return f"Bbox {{ x1: {self.x1}, y1: {self.y1}, w: {self.w}, h: {self.h} }}"

    @property
    def x1(self) -> T:
        return self._cvt(self._dt, self._x1)

    @property
    def x2(self) -> T:
        return self._cvt(self._dt, self._x1 + self._w)

    @property
    def y1(self) -> T:
        return self._cvt(self._dt, self._y1)

    @property
    def y2(self) -> T:
        return self._cvt(self._dt, self._y1 + self._h)

    @property
    def cx(self) -> T:
        return self._cvt(self._dt, self._x1 + self._w / 2)

    @property
    def cy(self) -> T:
        return self._cvt(self._dt, self._y1 + self._h / 2)

    @property
    def h(self) -> T:
        return self._cvt(self._dt, self._h)

    @property
    def w(self) -> T:
        return self._cvt(self._dt, self._w)

    def unpack(self) -> tuple[T, T, T, T]:
        return self.x1, self.y1, self.w, self.h


class Center(Box[T]):
    _cx: _BD
    _cy: _BD
    _w: _BD
    _h: _BD

    def __init__(
        self,
        cx: _BD,
        cy: _BD,
        w: _BD,
        h: _BD,
        dtype: Type[T] = float,
    ) -> None:
        super().__init__(dtype)
        self._cx = cx
        self._cy = cy
        self._w = w
        self._h = h

    def __str__(self) -> str:
        return f"Center {{ cx: {self.cx}, cy: {self.cy}, w: {self.w}, h: {self.h} }}"

    @property
    def x1(self) -> T:
        return self._cvt(self._dt, self._cx - self._w / 2)

    @property
    def x2(self) -> T:
        return self._cvt(self._dt, self._cx + self._w / 2)

    @property
    def y1(self) -> T:
        return self._cvt(self._dt, self._cy - self._h / 2)

    @property
    def y2(self) -> T:
        return self._cvt(self._dt, self._cy + self._h / 2)

    @property
    def cx(self) -> T:
        return self._cvt(self._dt, self._cx)

    @property
    def cy(self) -> T:
        return self._cvt(self._dt, self._cy)

    @property
    def h(self) -> T:
        return self._cvt(self._dt, self._h)

    @property
    def w(self) -> T:
        return self._cvt(self._dt, self._w)

    def unpack(self) -> tuple[T, T, T, T]:
        return self.cx, self.cy, self.w, self.h

    def __sub__(self, other: Point) -> Center[T]:
        return Center[T](
            self.cx - self._cvt(self._dt, (other.x)),
            self.cy - self._cvt(self._dt, (other.y)),
            self.w,
            self.h,
            self._dt,  # type: ignore
        )

    def __add__(self, other: Point) -> Center[T]:
        return Center[T](
            self.cx + self._cvt(self._dt, (other.x)),
            self.cy + self._cvt(self._dt, (other.y)),
            self.w,
            self.h,
            self._dt,  # type: ignore
        )
