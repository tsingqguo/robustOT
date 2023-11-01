from typing import TypeAlias

from .boxes import Box, Bbox, Center, Corner
from .point import Point
from .polygon import Polygon

Region: TypeAlias = Polygon | Box
