import trax
from msot.utils.region.boxes import Bbox


class Rectangle:
    @staticmethod
    def trax2msot(rect: trax.Rectangle) -> Bbox[float]:
        return Bbox[float](*rect.bounds())

    @staticmethod
    def msot2trax(rect: Bbox) -> trax.Rectangle:
        return trax.Rectangle.create(*rect.unpack2bbox())
