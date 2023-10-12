import warnings
from copy import deepcopy
from typing import Sequence

import numpy as np
import numpy.typing as npt

from msot.utils.region import helpers as region_helpers
from msot.utils.region.polygon import Bounds

from .statics import ReservedResults


def silent_nanmean(a, **kwargs) -> np.float_ | npt.NDArray[np.float_]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, **kwargs)


def calculate_failures(
    pred: Sequence[Sequence[float]],
) -> list[int]:
    failures = [
        i
        for i, p in enumerate(pred)
        if len(p) == 1 and p[0] == ReservedResults.FAILURE
    ]
    return failures


def calculate_accuracy(
    gt: Sequence[Sequence[float]],
    pred: list[list[float]],
    bounds: Bounds | None = None,
    burn_in: int = 0,
) -> list[float]:
    if burn_in > 0:
        pred_ = deepcopy(pred)
        mask = [len(p) == 1 and p[0] == ReservedResults.INIT for p in pred]
        for i in range(len(mask)):
            if mask[i]:
                for j in range(burn_in):
                    if i + j < len(mask):
                        pred_[i + j] = [float(ReservedResults.SKIP.value)]
    else:
        pred_ = pred

    overlaps = []
    for g, p in zip(gt, pred_):
        try:
            g = region_helpers.eval_from_list(g)
            p = region_helpers.eval_from_list(p)
            overlaps.append(
                region_helpers.calculate_overlap_ratio(g, p, bounds)
            )

        except ValueError:
            overlaps.append(float("nan"))

    return overlaps
