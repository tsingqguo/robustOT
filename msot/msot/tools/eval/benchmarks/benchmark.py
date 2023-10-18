from __future__ import annotations
from typing import Any, Generic, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

from msot.libs.potp_test.libs.video import Video

PredTraj: TypeAlias = npt.NDArray[np.float_]

T = TypeVar("T")


class BenchmarkItem(Generic[T]):
    def __eq__(self, other) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def execute(
        self,
        video: Video,
        pred_traj: PredTraj,
        benchmark: Benchmark | None = None,
    ) -> T:
        raise NotImplementedError


class Benchmark:
    """benchmark for single sequence"""

    video: Video
    _bitems: list[BenchmarkItem]
    _caches: dict[str, Any]

    def __init__(
        self,
        video: Video,
        benchmark_items: list[BenchmarkItem] | None = None,
    ) -> None:
        self.video = video
        self._bitems = benchmark_items or []
        self._caches = {}

    def get_cache(self, key: str, default=None):
        return self._caches.get(key, default)

    def set_cache(self, key: str, val):
        self._caches[key] = val

    def _convert_perd(
        self, pred_traj: list[list[float]] | PredTraj
    ) -> PredTraj:
        if not isinstance(pred_traj, np.ndarray):
            pred_traj = np.array(pred_traj)
        else:
            pred_traj = pred_traj
        assert len(self.video._gt[()]) == len(pred_traj)
        return pred_traj

    def eval_on(
        self, item: BenchmarkItem[T], pred_traj: list[list[float]] | PredTraj
    ) -> T:
        return item.execute(self.video, self._convert_perd(pred_traj), self)

    def eval(self, pred_traj: list[list[float]] | PredTraj):
        pred_traj = self._convert_perd(pred_traj)
        return {item: self.eval_on(item, pred_traj) for item in self._bitems}
