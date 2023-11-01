from typing_extensions import Self

import numpy as np
import numpy.typing as npt


class Point:
    _pt: npt.NDArray[np.float_]

    def __init__(self, x, y) -> None:
        self._pt = np.array([x, y], dtype=np.float_)

    @classmethod
    def from_ndarray(cls, arr: np.ndarray) -> Self:
        assert arr.shape == (2,)
        return cls(arr[0], arr[1])

    @property
    def x(self) -> float:
        return self._pt[0]

    @property
    def y(self) -> float:
        return self._pt[1]

    def __str__(self) -> str:
        return "Point {{ x: {}, y: {} }}".format(self._pt[0], self._pt[1])

    def __array__(self) -> npt.NDArray[np.float_]:
        return self._pt
