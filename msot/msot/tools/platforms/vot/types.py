from enum import Enum
from typing import Literal, NamedTuple, TypeAlias

import trax

VALID_CHANNEL_NAMES: TypeAlias = Literal["rgbd", "rgbt", "ir"]
VALID_CHANNEL_TYPES: TypeAlias = Literal["color", "depth", "ir"]


class TraxImageFormat(Enum):
    PATH = "path"
    URL = "url"
    BUFFER = "buffer"
    MEMORY = "memory"


class TraxRegionFormat(Enum):
    MASK = "mask"
    POLYGON = "polygon"
    RECTANGLE = "rectangle"
    SPECIAL = "special"


class TraxServerRequest(NamedTuple):
    type: Literal["initialize", "frame"]
    image: dict[VALID_CHANNEL_TYPES, trax.image.Image]
    objects: list[tuple[trax.Region, trax.Properties]] | None
    properties: trax.Properties
