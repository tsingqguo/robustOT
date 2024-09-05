from ..dataset import Dataset
from ..video import Video
from ..types import Path


class LaSOTVideo(Video["LaSOTDataset"]):
    ...


class LaSOTDataset(Dataset[LaSOTVideo]):
    def __init__(self, h5fp: Path) -> None:
        super().__init__("LaSOT", h5fp, LaSOTVideo)
