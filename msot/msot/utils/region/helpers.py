from typing import Sequence, Type, overload

from .boxes import Box, Bbox, Center, Corner
from .point import Point
from .polygon import Bounds, Polygon


@overload
def eval_from_list(
    points: Sequence[int | float], box_type: Type[Bbox] = Bbox
) -> Polygon | Bbox:
    ...


@overload
def eval_from_list(
    points: Sequence[int | float], box_type: Type[Center] = Center
) -> Polygon | Center:
    ...


@overload
def eval_from_list(
    points: Sequence[int | float], box_type: Type[Corner] = Corner
) -> Polygon | Corner:
    ...


def eval_from_list(
    points: Sequence[int | float], box_type: Type[Box] = Bbox
) -> Polygon | Box:
    if len(points) < 4:
        raise ValueError("Invalid points")
    elif len(points) == 4:
        return box_type(*points)
    else:
        return eval_polygon_from_list(points)
    # TODO: mask support


def eval_polygon_from_list(points: Sequence[int | float]) -> Polygon:
    vertices = []
    for i in range(0, len(points), 2):
        vertices.append(Point(points[i], points[i + 1]))
    return Polygon(vertices)


def calculate_overlap_ratio(
    p1: Polygon | Box,
    p2: Polygon | Box,
    bounds: Bounds | None = None,
) -> float:
    if not isinstance(p1, Polygon):
        p1 = Polygon(p1.vertices)
    if not isinstance(p2, Polygon):
        p2 = Polygon(p2.vertices)

    return p1.get_overlap_ratio(p2, bounds)
