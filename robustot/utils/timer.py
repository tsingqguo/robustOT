import time
from enum import Enum
from typing import Callable, Type, overload


class TimerType(Enum):
    std = 0
    cv2 = 1


def get_current(type: TimerType):
    if type == TimerType.std:
        return time.time()
    elif type == TimerType.cv2:
        import cv2

        return cv2.getTickCount()


def get_elapsed(start, end, type: TimerType):
    if type == TimerType.std:
        return end - start
    elif type == TimerType.cv2:
        import cv2

        return (end - start) / cv2.getTickFrequency()


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
            tt = TimerType.std
        elif isinstance(tt, TimerType):
            ...
        elif callable(tt):
            tt = tt(TimerType)
        else:
            raise NotImplementedError
        self._type = tt
        self.reset()

    @property
    def elapsed(self):
        return get_elapsed(self._t[-1], get_current(self._type), self._type)

    def reset(self):
        self._t = [get_current(self._type)]


if __name__ == "__main__":
    t1 = Timer(TimerType.cv2)
    t2 = Timer(TimerType.std)
    time.sleep(0.1)
    print(t1.elapsed)
    print(t2.elapsed)
