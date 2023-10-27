from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from msot.trackers.base import (
    BaseTracker,
    TrackConfig,
    TrackerState,
    TrackResult,
)

from ..info import SequenceInfo
from ..utils.historical import Historical
from ..utils.process import InputProcess
from ..utils.result import TestResult

TC = TypeVar("TC", bound=TrackConfig)
TS = TypeVar("TS", bound=TrackerState)
TR = TypeVar("TR", bound=TrackResult)


@dataclass
class ParamsFrame(Generic[TC, TS, TR]):
    raw_tracker: BaseTracker[TC, TS, TR]
    historical: Historical
    sequence_info: SequenceInfo


@dataclass
class ParamsFrameInit(ParamsFrame[TC, TS, TR]):
    input_process: InputProcess


@dataclass
class ParamsFrameSkip(ParamsFrame[TC, TS, TR]):
    ...


@dataclass
class ParamsFrameTrack(ParamsFrame[TC, TS, TR]):
    input_process: InputProcess


@dataclass
class ParamsFramePost(ParamsFrame[TC, TS, TR]):
    ...


@dataclass
class ParamsFrameFinish(ParamsFrame[TC, TS, TR]):
    results_handler: TestResult | None


TestActionInit = Callable[[ParamsFrameInit], None]
TestActionTrack = Callable[[ParamsFrameTrack], TrackResult]
TestActionSkip = Callable[[ParamsFrameSkip], None]
TestActionPost = Callable[[ParamsFramePost], None]
TestActionFinish = Callable[[ParamsFrameFinish], None]
