from __future__ import annotations
from typing import Callable, Generic, NamedTuple, Type, TypeVar, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch

from msot.trackers.base import TrackerState
from msot.trackers.types import ScaledCrop
from msot.utils.dataship import (
    DataCTR as DC,
    DataCTRAC as DCAC,
    DataShip as DS,
    VertDCAC,
)
from msot.utils.option import NONE, Option, Some
from msot.utils.region import Bbox, Region

from .roles import TDRoles

if TYPE_CHECKING:
    from .process import ProceesorAttrs

TS = TypeVar("TS", bound=TrackerState)
U = TypeVar("U", bound=DS)
V = TypeVar("V", bound="Frame | Tracking | Result | Analysis")


class Frame(DS):
    img: DCAC[npt.NDArray[np.uint8], TDRoles]
    gt: DCAC[Region, TDRoles]

    @property
    def valid_names(self) -> set[str]:
        return super().valid_names | {"img", "gt"}

    def __init__(
        self,
        img: npt.NDArray[np.uint8],
        gt: Region | None = None,
    ) -> None:
        self.img = DCAC(
            TDRoles,
            lambda role: role >= TDRoles.TEST,
            # lambda role: role & TDRoles.TRACKER | TDRoles.DEBUG == role,
            img,
            is_shared=True,
            is_mutable=False,
            allow_unbound=False,
        )
        self.gt = DCAC(
            TDRoles,
            lambda role: role >= TDRoles.TEST,
            is_shared=True,
            is_mutable=False,
            allow_unbound=True,
        )
        if gt is not None:
            self.gt.update(gt)


class Result(DS, Generic[TS]):
    tracker_state: TS
    pred: DC[Bbox]  # TODO: polygon
    best_score: DC[float | None]
    is_skip: DC[bool]

    def __init__(
        self,
        tracker_state: TS,
        pred: Bbox,
        best_score: float | None,
        is_skip: bool = False,
    ) -> None:
        self.tracker_state = tracker_state.smart_clone()  # TODO: fork dilemma
        self.pred = DC(
            pred,
            is_shared=True,
            is_mutable=False,
            allow_unbound=False,
        )
        self.best_score = DC(
            best_score,
            is_shared=True,
            is_mutable=False,
            allow_unbound=False,
        )
        self.is_skip = DC(
            is_skip,
            is_shared=True,
            is_mutable=False,
            allow_unbound=False,
        )


class Analysis(DS):
    overlap: DCAC[float | None, TDRoles]

    @property
    def valid_names(self) -> set[str]:
        return super().valid_names | {"overlap"}

    def __init__(self, overlap: float | None) -> None:
        self.overlap = DCAC(
            TDRoles,
            lambda role: role >= TDRoles.ANALYSIS,
            overlap,
            is_shared=False,
            is_mutable=False,
            allow_unbound=False,
        )


class Tracking(DS):
    scaled_z: VertDCAC[ScaledCrop, TDRoles, str | None]
    scaled_x: VertDCAC[ScaledCrop, TDRoles, str | None]
    processor_attrs: DC[dict[str, ProceesorAttrs]]

    @property
    def valid_names(self) -> set[str]:
        return super().valid_names | {
            "scaled_z",
            "scaled_x",
            "processor_attrs",
        }

    def __init__(
        self,
    ):
        self.scaled_z = VertDCAC(
            TDRoles,
            lambda role: role >= TDRoles.TRACKER,
            lambda role: role >= TDRoles.DEBUG,
        )
        self.scaled_x = VertDCAC(
            TDRoles,
            lambda role: role >= TDRoles.TRACKER,
            lambda role: role >= TDRoles.DEBUG,
        )
        self.processor_attrs = DC(
            is_shared=False,
            is_mutable=False,
            allow_unbound=True,
        )


class _Released(DS):
    def __repr__(self) -> str:
        return "(released)"


class _ReleasedError(Exception):
    ...


class _HistoricalDS(Generic[U]):
    _cur: Option[U]
    _list: list[U | _Released]
    _mem_size: int | None
    _locked: bool

    def __init__(self, mem_size: int | None = None) -> None:
        # self._cur = unbound
        self._list = []
        self._mem_size = mem_size
        self._locked = False

    @property
    def cur(self) -> Option[U]:
        if self._locked:
            raise RuntimeError("Unbound cur data after finalize")
        return self._cur

    @property
    def last(self) -> U:
        last = self._list[-1]
        if type(last) is _Released:
            raise _ReleasedError()
        return last  # type: ignore

    def finalize(self):
        self.next()
        del self._cur
        self._locked = True

    def next(self, data: U | Option = NONE):
        if self._locked:
            raise RuntimeError("next is not allowed after finalize")
        if not hasattr(self, "_cur"):
            ...
        else:
            if self._cur.is_none():
                raise RuntimeError("Unbound cur data before next")
            self._list.append(self.cur.unwrap())
            if self._mem_size is not None:
                for i in range(len(self._list) - self._mem_size):
                    self._list[i] = _Released()
        self._cur = Some(data) if not isinstance(data, Option) else data

    def __getitem__(self, idx: int) -> U | Option[U]:
        if idx == len(self._list):
            return self._cur
        else:
            item = self._list[idx]
            if type(item) is _Released:
                raise _ReleasedError()
        return item  # type: ignore

    def __len__(self) -> int:
        if not hasattr(self, "_cur"):
            return len(self._list)
        else:
            return len(self._list) + 1

    def __iter__(self):
        self._iter_num = 0
        return self

    def __next__(self) -> U | Option[U]:
        if self._iter_num < len(self):
            self._iter_num += 1
            return self[self._iter_num - 1]
        else:
            raise StopIteration

    def to_list(self, ignore_released: bool = False) -> list[U]:
        if not self._locked:
            raise RuntimeError("to_list is not allowed before finalize")
        lst = []
        for item in self._list:
            if type(item) is _Released:
                if ignore_released:
                    continue
                else:
                    raise _ReleasedError()
            else:
                lst.append(item)
        return lst


class _HistoricalTupled(NamedTuple):
    frame: Frame
    tracking: Tracking
    result: Result
    analysis: Analysis


class _HistoricalTupledOpt(NamedTuple):
    frame: Option[Frame]
    tracking: Option[Tracking]
    result: Option[Result]
    analysis: Option[Analysis]


class Historical:
    frame_cls: type[Frame]
    tracking_cls: type[Tracking]
    result_cls: type[Result]
    analysis_cls: type[Analysis]

    _frame: _HistoricalDS[Frame]
    _tracking: _HistoricalDS[Tracking]
    _result: _HistoricalDS[Result]
    _analysis: _HistoricalDS[Analysis]

    def __init__(
        self,
        frame_mem_size: int = 10,
        tracking_mem_size: int = 10,
    ) -> None:
        self.frame_cls = Frame
        self.tracking_cls = Tracking
        self.result_cls = Result
        self.analysis_cls = Analysis

        self._frame = _HistoricalDS(frame_mem_size)
        self._tracking = _HistoricalDS(tracking_mem_size)
        self._result = _HistoricalDS()
        self._analysis = _HistoricalDS()

    def _check_len(self):
        assert (
            len(self._frame)
            == len(self._tracking)
            == len(self._result)
            == len(self._analysis)
        )

    def next(
        self,
        frame: Frame | Option = NONE,
        tracking: Tracking | Option = NONE,
        result: Result | Option = NONE,
        analysis: Analysis | Option = NONE,
    ) -> _HistoricalTupledOpt:
        self._check_len()
        f = self._frame.next(frame)
        t = self._tracking.next(tracking)
        r = self._result.next(result)
        a = self._analysis.next(analysis)
        return self.cur

    def set_cur_(self, ds: V) -> V:
        if isinstance(ds, Frame):
            self._frame._cur = Some(ds)
        elif isinstance(ds, Tracking):
            self._tracking._cur = Some(ds)
        elif isinstance(ds, Result):
            self._result._cur = Some(ds)
        elif isinstance(ds, Analysis):
            self._analysis._cur = Some(ds)
        else:
            raise RuntimeError("Unsupported type for set_cur_")
        return ds

    def set_cur_frame(
        self,
        cb: Callable[[Type[Frame]], Frame],
    ) -> Frame:
        ds = cb(self.frame_cls)
        return self.set_cur_(ds)

    def set_cur_tracking(
        self, cb: Callable[[Type[Tracking]], Tracking]
    ) -> Tracking:
        ds = cb(self.tracking_cls)
        return self.set_cur_(ds)

    def set_cur_result(
        self,
        cb: Callable[[Type[Result]], Result],
    ) -> Result:
        ds = cb(self.result_cls)
        return self.set_cur_(ds)

    def set_cur_analysis(
        self,
        cb: Callable[[Type[Analysis]], Analysis],
    ) -> Analysis:
        ds = cb(self.analysis_cls)
        return self.set_cur_(ds)

    @property
    def cur(self) -> _HistoricalTupledOpt:
        self._check_len()
        return _HistoricalTupledOpt(
            self._frame.cur,
            self._tracking.cur,
            self._result.cur,
            self._analysis.cur,
        )

    @property
    def last(self) -> Option[_HistoricalTupled]:
        self._check_len()
        if len(self._frame) > 1:
            return Some(
                _HistoricalTupled(
                    self._frame.last,
                    self._tracking.last,
                    self._result.last,
                    self._analysis.last,
                )
            )
        else:
            return NONE

    def get_historical_result(self) -> _HistoricalDS[Result]:
        return self._result

    def get_historical_tracking(self) -> _HistoricalDS[Tracking]:
        return self._tracking

    def get_historical_frame(self) -> _HistoricalDS[Frame]:
        return self._frame

    def get_historical_analysis(self) -> _HistoricalDS[Analysis]:
        return self._analysis

    def finalize(self):
        self._frame.finalize()
        self._tracking.finalize()
        self._result.finalize()
        self._analysis.finalize()

    def __len__(self):
        return len(self._frame)
