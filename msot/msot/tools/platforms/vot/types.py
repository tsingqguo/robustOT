from enum import Enum
from typing import Literal, NamedTuple, TypeAlias

import trax

TraxStatus: TypeAlias = Literal["initialize", "frame", "quit"]

TraxChannelNames: TypeAlias = Literal["rgbd", "rgbt", "ir"]
TraxChannelTypes: TypeAlias = Literal["color", "depth", "ir"]


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
    type: TraxStatus
    image: dict[TraxChannelTypes, trax.image.Image]
    objects: list[tuple[trax.Region, trax.Properties]] | None
    properties: trax.Properties
