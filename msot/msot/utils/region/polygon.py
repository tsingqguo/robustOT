from __future__ import annotations
import math
from dataclasses import dataclass
from functools import cached_property
from typing import Generic, TypeVar

import cv2
import numpy as np

from .point import Point, ValidType

T = TypeVar("T", bound=ValidType)


@dataclass
class Bounds(Generic[T]):
    top: T
    bottom: T
    left: T
    right: T

    def get_intersection(self, other: Bounds[T] | None) -> Bounds[T]:
        if other is None:
            return Bounds(self.top, self.bottom, self.left, self.right)
        else:
            return Bounds(
                max(self.top, other.top),
                min(self.bottom, other.bottom),
                max(self.left, other.left),
                min(self.right, other.right),
            )

    def get_overlap_ratio(self, other: Bounds[T]) -> float:
        o = self.get_intersection(other)
        oa = o.area
        return max(0, oa / (self.area + other.area - oa))

    def round(self) -> Bounds[int]:
        return Bounds(
            math.floor(self.top),
            math.ceil(self.bottom),
            math.floor(self.left),
            math.ceil(self.right),
        )

    @cached_property
    def area(self) -> float:
        return (self.right - self.left) * (self.bottom - self.top)


class Polygon(Generic[T]):
    _vertices: list[Point[T]]

    def __init__(self, vertices: list[Point[T]]) -> None:
        self._vertices = vertices

    def __str__(self) -> str:
        return f"Polygon {{ {self.vertices} }}"

    @property
    def vertices(self):
        return self._vertices

    @property
    def count(self) -> int:
        return len(self.vertices)

    @property
    def area(self) -> float:
        raise NotImplementedError

    def get_bounds(self) -> Bounds[T]:
        x: list[T] = []
        y: list[T] = []

        for v in self.vertices:
            x.append(v.x)
            y.append(v.y)

        return Bounds(top=min(y), bottom=max(y), left=min(x), right=max(x))

    def get_overlap_ratio(
        self, other: Polygon, bounds: Bounds | None = None
    ) -> float:
        b1 = self.get_bounds().round().get_intersection(bounds)
        b2 = other.get_bounds().round().get_intersection(bounds)

        x = min(b1.left, b2.left)
        y = min(b1.top, b2.top)
        w = int(max(b1.right, b2.right) - x) + 1
        h = int(max(b1.bottom, b2.bottom) - y + 1)

        # if (
        #     b1.area / b2.area < 1e-10
        #     or b2.area / b1.area < 1e-10
        #     or w < 1
        #     or h < 1
        # ):
        #     return 0

        # if b1.get_overlap_ratio(b2) == 0:
        #     return 0

        v1_offset = self.offset_vertices(-x, -y)
        v2_offset = other.offset_vertices(-x, -y)

        mask1 = self.rasterize(v1_offset, w, h).flatten()
        mask2 = self.rasterize(v2_offset, w, h).flatten()

        mask_intersect = np.sum(np.logical_and(mask1, mask2))
        return mask_intersect / (
            np.sum(mask1) + np.sum(mask2) - mask_intersect
        )

    def offset_vertices(self, x, y) -> list[Point[int]]:
        vertices = []
        for v in self.vertices:
            vertices.append(Point(int(v.x + x), int(v.y + y)))

        return vertices

    @staticmethod
    def rasterize(vertices: list[Point[T]], width: int, height: int):
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(
            mask,
            [np.array(list(map(lambda p: p.__tuple__(), vertices)))],
            [1],
        )
        return mask
