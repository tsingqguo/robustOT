import cv2
import numpy as np
import torch
from enum import Enum
from pysot.utils.bbox import Center
from pyotp.utils.option import NONE, Option, Some
from toolkit.datasets.video import SeqImg, Video
from typing import (
    Any,
    Generic,
    NamedTuple,
    NoReturn,
    Optional,
    TypeVar,
    Union,
)

_int = Union[int, np.int8, np.int64]
U = TypeVar("U")
V = TypeVar("V", bound=Video)


class VisStyle(Enum):
    Newline = 0
    Inline = 1


class BBox:
    def __init__(self, x: _int, y: _int, w: _int, h: _int) -> None:
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)

    @property
    def lt(self) -> tuple[int, int]:
        """left-top point of the bounding box"""
        return self.x, self.y

    @property
    def lb(self) -> tuple[int, int]:
        """left-bottom point of the bounding box"""
        return self.x, self.y + self.h

    @property
    def rt(self) -> tuple[int, int]:
        """right-top point of the bounding box"""
        return self.x + self.w, self.y

    @property
    def rb(self) -> tuple[int, int]:
        """right-bottom point of the bounding box"""
        return self.x + self.w, self.y + self.h

    def unpack(self) -> tuple[int, int, int, int]:
        """unpack the bounding box to (x1, y1, w, h)"""
        return self.x, self.y, self.w, self.h

    def to_center(self) -> Center:
        return Center(
            x=self.x + self.w / 2,
            y=self.y + self.h / 2,
            w=float(self.w),
            h=float(self.h),
        )


class _Undefined:
    ...


UNDEFINED = _Undefined()


class DataCTR:
    _data_ctr: dict[str, tuple[bool, Any]]

    def __init__(self) -> None:
        self._data_ctr = {}

    def has(self, register: str) -> bool:
        return register in self._data_ctr

    def set_data(
        self, register: str, data: Any, mutable: bool = False
    ) -> Union[NoReturn, None]:
        _mutable, _ = self._data_ctr.get(register, (True, None))
        if _mutable:
            self._data_ctr[register] = (mutable, data)
        else:
            raise ValueError(f'DataCTR["{register}"] is not mutable')

    def get_data(
        self, register: str, default: Union[_Undefined, U] = UNDEFINED
    ) -> U:
        if isinstance(default, _Undefined):
            _, data = self._data_ctr[register]
        else:
            _, data = self._data_ctr.get(register, (True, default))
        return data  # type: ignore


class HistoricalFrame:
    bbox: list[Any]
    bbox_pred: Optional[list[Any]]
    aligned_bbox: tuple[float, float, float, float]
    """
    `cw`, `ch`, `w`, `h` in `np.float64`/`float`
    """
    img: SeqImg
    external_visualization: Optional[dict[str, tuple[cv2.Mat, VisStyle]]]

    def __init__(
        self,
        bbox: list[Any],
        bbox_pred: Optional[list[Any]],
        aligned_bbox: tuple[float, float, float, float],
        img: SeqImg,
        external_visualization: Optional[Any] = None,
    ) -> None:
        self.bbox = bbox
        self.bbox_pred = bbox_pred
        self.aligned_bbox = aligned_bbox
        self.img = img
        self.external_visualization = external_visualization


class HistoricalTracking:
    # img: np.ndarray
    # center_pos: npt.NDArray[np.float64]
    x_crop: Optional[torch.Tensor]
    """x_crop feed into the tracker model"""
    x_crop_gt: Optional[torch.Tensor]
    x_crop_trkpp_free: Optional[torch.Tensor]
    """x_crop feed into the tracker without trkpp processing (may already be attacked)"""
    z_crop: Optional[torch.Tensor]

    def __init__(
        self,
        x_crop: Optional[torch.Tensor] = None,
        x_crop_gt: Optional[torch.Tensor] = None,
        x_crop_trkpp_free: Optional[torch.Tensor] = None,
        z_crop: Optional[torch.Tensor] = None,
    ) -> None:
        self.x_crop = x_crop
        self.x_crop_gt = x_crop_gt
        self.x_crop_trkpp_free = x_crop_trkpp_free
        self.z_crop = z_crop

    def __str__(self):
        def _show_tensor(t: Optional[torch.Tensor]) -> str:
            if t is None:
                return "None"
            else:
                return f"{list(t.shape)} {t.dtype}"

        msg = [
            "HistoricalTracking",
            "\t.x_crop:",
            _show_tensor(self.x_crop),
            "\t.x_crop_gt:",
            _show_tensor(self.x_crop_gt),
            "\t.x_crop_trkpp_free:",
            _show_tensor(self.x_crop_trkpp_free),
            "\t.z_crop:",
            _show_tensor(self.z_crop),
        ]
        return "\n".join(msg)


class Historical(NamedTuple):
    frame: HistoricalFrame
    tracking: Option[HistoricalTracking]
    extra: DataCTR


class Released:
    ...


class AccessingReleasedError(Exception):
    ...


class HistoricalMgr(Generic[V]):
    _video: V
    _his: list[Historical | Released]

    _max_save_num: int
    """saving max n (except the first) historical data"""

    def __init__(self, video: V, mem_size: int = 50):
        self._video = video
        self._his = []
        self._max_save_num = mem_size

    @property
    def cur(self) -> Historical:
        return self[-1]

    @property
    def last(self) -> Option[Historical]:
        if len(self) > 1:
            return Some(self[-2])
        else:
            return NONE

    @property
    def frames(self) -> NoReturn:
        raise NotImplementedError("removed")

    @staticmethod
    def get_slice(
        items: list[U | Released],
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> list[U]:
        if start is None:
            start = 0
        else:
            start = max(0, start)
        if end is None:
            end = len(items)
        else:
            end = min(len(items), end)
        safe_items: list[U] = []
        for i in range(start, end):
            items.append(items[i])
        return safe_items

    def get_frames_slice(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> list[HistoricalFrame]:
        return [h.frame for h in self.get_slice(self._his, start, end)]

    @property
    def tracking_history(self) -> NoReturn:
        raise NotImplementedError("removed")

    def get_tracking_history_slice(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> list[Option[HistoricalTracking]]:
        return [h.tracking for h in self.get_slice(self._his, start, end)]

    def flush(self):
        self._his = []

    def next(
        self, frame: HistoricalFrame, tracking: Option[HistoricalTracking]
    ) -> Historical:
        self._his.append(
            Historical(
                frame=frame,
                tracking=tracking,
                extra=DataCTR(),
            )
        )
        for i in range(1, len(self) - self._max_save_num):
            self._his[i] = Released()
        return self.cur

    def __getitem__(self, idx: int) -> Historical:
        item = self._his[idx]
        if isinstance(item, Released):
            raise AccessingReleasedError()
        return item

    def __len__(self) -> int:
        return len(self._his)

    def __iter__(self):
        self._iter_num = 0
        return self

    def __next__(self) -> Historical:
        if self._iter_num < len(self._his):
            self._iter_num += 1
            hist = self._his[self._iter_num - 1]
            if isinstance(hist, Released):
                raise AccessingReleasedError()
            return hist
        else:
            raise StopIteration
