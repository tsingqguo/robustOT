import h5py
import os
from .types import Dataset as _Dataset, Path, V
from typing import Iterator, Optional, Type, Union


class Dataset(_Dataset[V]):
    name: str
    video_vls: Type[V]
    _h5: Optional[h5py.File]
    _videos: dict[str, V]

    def __init__(self, name: str, h5fp: Path, video_cls: Type[V]) -> None:
        if not os.path.exists(h5fp):
            raise FileNotFoundError(f"{h5fp} does not exist")

        self.name = name
        self.video_vls = video_cls
        self._h5 = h5py.File(h5fp, "r")
        self._videos = {}

    @property
    def h5fh(self) -> h5py.File:
        if self._h5 is None:
            raise ValueError(f"dataset is closed")
        else:
            return self._h5

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
            self._videos = {}

    def _get_video(self, v_name: str) -> V:
        if v_name not in self._videos:
            v_node = self.h5fh[v_name]
            if not isinstance(v_node, h5py.Group):
                raise ValueError(f"{v_name} is not a group")
            else:
                self._videos[v_name] = self.video_vls.create(
                    self, v_node, v_name
                )
        return self._videos[v_name]

    def __getitem__(self, idx: Union[int, str]) -> V:
        if isinstance(idx, int):
            v_name = list(self.h5fh.keys())[idx]
        elif isinstance(idx, str):
            v_name = idx
        else:
            raise TypeError(f"idx must be int or str, not {type(idx)}")
        return self._get_video(v_name)

    def __len__(self) -> int:
        return len(self.h5fh.keys())

    def __iter__(self) -> Iterator[V]:
        for i in range(len(self)):
            yield self[i]
