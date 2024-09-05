from ..dataset import Dataset
from ..video import Video
from ..types import Path


class UAVVideo(Video["UAVDataset"]):
    ...


class UAVDataset(Dataset[UAVVideo]):
    def __init__(self, h5fp: Path) -> None:
        super().__init__("UAV123", h5fp, UAVVideo)
