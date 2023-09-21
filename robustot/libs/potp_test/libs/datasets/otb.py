from ..dataset import Dataset
from ..video import Video
from ..types import Path


class OTB100Video(Video["OTB100Dataset"]):
    ...


class OTB100Dataset(Dataset[OTB100Video]):
    def __init__(self, h5fp: Path) -> None:
        super().__init__("OTB100", h5fp, OTB100Video)
