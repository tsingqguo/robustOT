from typing import Sequence, Type, TypeVar, overload

from .boxes import Box, Bbox, Center, Corner
from .point import ValidType
from .polygon import Bounds, Polygon, Vertex

T = TypeVar("T", bound=ValidType)


@overload
def eval_from_list(
    points: Sequence[T], box_type: Type[Bbox] = Bbox
) -> Polygon[T] | Bbox[T]:
    ...


@overload
def eval_from_list(
    points: Sequence[T], box_type: Type[Center] = Center
) -> Polygon[T] | Center[T]:
    ...


@overload
def eval_from_list(
    points: Sequence[T], box_type: Type[Corner] = Corner
) -> Polygon[T] | Corner[T]:
    ...


def eval_from_list(
    points: Sequence[T], box_type: Type[Box] = Bbox
) -> Polygon[T] | Box[T]:
    if len(points) < 4:
        raise ValueError("Invalid points")
    elif len(points) == 4:
        return box_type(*points)
    else:
        return eval_polygon_from_list(points)
    # TODO: mask support


def eval_polygon_from_list(points: Sequence[T] | list[T]) -> Polygon[T]:
    vertices = []
    for i in range(0, len(points), 2):
        vertices.append(Vertex(points[i], points[i + 1]))
    return Polygon(vertices)


def calculate_overlap_ratio(
    p1: Polygon[T],
    p2: Polygon[T],
    bounds: Bounds[T] | None = None,
) -> float:
    if p1.__class__ is not Polygon:
        p1 = Polygon(p1.vertices)
    if p2.__class__ is not Polygon:
        p2 = Polygon(p2.vertices)

    return p1.get_overlap_ratio(p2, bounds)
