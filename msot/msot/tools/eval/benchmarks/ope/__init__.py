import numpy as np
import numpy.typing as npt

from ..benchmark import Benchmark, BenchmarkItem, PredTraj, Video
from .analysis import convert_bbox_to_center, success_error, success_overlap


class OPEAccuracy(BenchmarkItem[npt.NDArray[np.float_]]):
    def execute(
        self,
        video: Video,
        pred_traj: PredTraj,
        benchmark: Benchmark | None,
    ) -> npt.NDArray[np.float_]:
        return success_overlap(video._gt[()], pred_traj)


class OPEPrecision(BenchmarkItem[npt.NDArray[np.float_]]):
    thlds: np.ndarray
    thlds_idx: int

    def __init__(self, thlds: np.ndarray, thlds_idx: int) -> None:
        self.thlds = thlds
        self.thlds_idx = thlds_idx

    def execute(
        self,
        video: Video,
        pred_traj: PredTraj,
        benchmark: Benchmark | None,
    ) -> npt.NDArray[np.float_]:
        return success_error(
            convert_bbox_to_center(video._gt[()]),
            convert_bbox_to_center(pred_traj),
            thlds=self.thlds,
        )[self.thlds_idx]
