import time
from enum import Enum, auto
from typing import Callable, Type, overload

import torch


class TimerType(Enum):
    STD = auto()
    CV2 = auto()
    CUDA = auto()


def get_current(type: TimerType):
    if type == TimerType.STD:
        return time.time()
    elif type == TimerType.CV2:
        import cv2

        return cv2.getTickCount()
    elif type == TimerType.CUDA:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        return (start, end)


def get_elapsed(start, type: TimerType):
    if type == TimerType.STD:
        end = get_current(type)
        return end - start
    elif type == TimerType.CV2:
        import cv2

        end = get_current(type)
        return (end - start) / cv2.getTickFrequency()
    elif type == TimerType.CUDA:
        start, end = start
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / 1000


class Timer:
    _type: TimerType
    _t: list

    @overload
    def __init__(self, tt: TimerType | None = None) -> None:
        ...

    @overload
    def __init__(
        self, tt: Callable[[Type[TimerType]], TimerType] | None = None
    ) -> None:
        ...

    def __init__(
        self,
        tt: TimerType | Callable[[Type[TimerType]], TimerType] | None = None,
    ) -> None:
        if tt is None:
            tt = TimerType.STD
        elif isinstance(tt, TimerType):
            ...
        elif callable(tt):
            tt = tt(TimerType)
        else:
            raise NotImplementedError
        self._type = tt

        if self._type == TimerType.CUDA:
            assert torch.cuda.is_available(), "CUDA is not available"

        self.reset()

    @property
    def elapsed(self) -> float:
        return get_elapsed(self._t[-1], self._type)

    def reset(self):
        self._t = [get_current(self._type)]


if __name__ == "__main__":
    t1 = Timer(TimerType.CV2)
    t2 = Timer(TimerType.STD)
    time.sleep(0.1)
    print(t1.elapsed)
    print(t2.elapsed)
