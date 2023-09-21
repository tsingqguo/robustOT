from ..dataset import Dataset
from ..video import Video
from ..types import Path
from typing import Type, TypeVar

ND = TypeVar("ND", bound="_NFSDataset")
NV = TypeVar("NV", bound="_NFSVideo")

class _NFSVideo(Video[ND]):
    ...


class NFS30Video(_NFSVideo["NFS30Dataset"]):
    ...


class NFS240Video(_NFSVideo["NFS240Dataset"]):
    ...


class _NFSDataset(Dataset[NV]):
    def __init__(self, name: str, h5fp: Path, video_cls: Type[NV]) -> None:
        super().__init__(name, h5fp, video_cls)


class NFS30Dataset(_NFSDataset[NFS30Video]):
    def __init__(self, h5fp: Path) -> None:
        super().__init__("NFS30", h5fp, NFS30Video)


class NFS240Dataset(_NFSDataset[NFS240Video]):
    def __init__(self, h5fp: Path) -> None:
        super().__init__("NFS240", h5fp, NFS240Video)
