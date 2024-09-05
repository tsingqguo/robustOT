import h5py
import numpy as np
import numpy.typing as npt
from os import PathLike
# from abc import abstractmethod
from typing import (
    Generic,
    Iterator,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
)
from typing_extensions import Self

Path = Union[str, PathLike]
V = TypeVar("V", bound="Video")
D = TypeVar("D", bound="Dataset")


class Dataset(Protocol, Generic[V]):
    name: str
    video_vls: Type[V]
    _h5: Optional[h5py.File]

    def __init__(self, name: str, h5fp: Path, video_cls: Type[V]) -> None:
        ...

    def _get_video(self, v_name: str) -> V:
        ...

    def __getitem__(self, idx: Union[int, str]) -> V:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[V]:
        ...


VideoItem = tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]


class Video(Protocol, Generic[D]):
    name: str
    attrs: dict

    _dataset: D
    _node: h5py.Group
    _frames: h5py.Dataset
    _gt_bboxes: h5py.Dataset

    @classmethod
    def create(cls, dataset: D, node: h5py.Group, name: str) -> Self:
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> VideoItem:
        ...

    def __iter__(self) -> Iterator[VideoItem]:
        ...
