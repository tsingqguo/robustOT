from toolkit.datasets.video import Video
from typing import (
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)


class VariantNotFound(Exception):
    pass


V = TypeVar("V", bound=Video)


class Dataset(Generic[V]):
    name: str
    dataset_root: str
    video_class: Type[V]
    videos: Dict[str, V]

    tracker_path: str
    tracker_names: List[str]

    def __init__(self, name: str, dataset_root: str, video_class: Type[V]):
        self.name = name
        self.dataset_root = dataset_root
        self.video_class = video_class
        self.videos = {}  # FIXME: origin: self.videos = None

    def _create_video(self, *args, **kwargs) -> V:
        """protected method to create video instance"""
        kwargs = {
            **kwargs,
        }
        video = self.video_class(*args, **kwargs)
        return video

    def __getitem__(self, idx: Union[int, str]) -> V:
        if isinstance(idx, str):
            return self.videos[idx]
        elif isinstance(idx, int):
            return self.videos[sorted(list(self.videos.keys()))[idx]]

    def __len__(self) -> int:
        return len(self.videos)

    def __iter__(self) -> Iterator[V]:
        keys = sorted(list(self.videos.keys()))
        for key in keys:
            yield self.videos[key]

    def set_tracker(self, tracker_res_root: str, tracker_names: List[str]):
        self.tracker_path = tracker_res_root
        self.tracker_names = tracker_names
