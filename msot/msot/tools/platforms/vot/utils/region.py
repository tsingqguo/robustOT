from typing import TypeAlias

import trax

from msot.utils.region.boxes import Bbox
from msot.utils.region.polygon import Polygon

Region: TypeAlias = Bbox  # TODO:


class Rectangle:
    @staticmethod
    def trax2msot(rect: trax.Rectangle) -> Bbox[float]:
        return Bbox[float](*rect.bounds())

    @staticmethod
    def msot2trax(rect: Bbox) -> trax.Rectangle:
        return trax.Rectangle.create(*rect.unpack())


def msot2trax(region: Region | None) -> trax.Region:
    if region is None:
        return trax.Rectangle.create(0, 0, 0, 0)
    elif isinstance(region, Polygon):
        raise NotImplementedError
    elif isinstance(region, Bbox):
        return Rectangle.msot2trax(region)
    # elif isinstance(region, Mask):
    #     raise NotImplementedError
    else:
        raise NotImplementedError


def trax2msot(region: trax.Region) -> Region:
    if isinstance(region, trax.Polygon):
        raise NotImplementedError
    elif isinstance(region, trax.Mask):
        raise NotImplementedError
    elif isinstance(region, trax.Rectangle):
        return Rectangle.trax2msot(region)
    else:
        raise NotImplementedError
