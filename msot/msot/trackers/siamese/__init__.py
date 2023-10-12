from typing import TypeVar

import numpy as np
import numpy.typing as npt
import torch

from msot.utils.region import Point

from ..base import (
    BaseTracker,
    TrackerState as _TrackerState,
    TrackResult as _TrackResult,
)
from ..types import ScaledCrop, TrackSize
from .config import TrackConfig


class TrackerState(_TrackerState):
    ...


class TrackResult(_TrackResult):
    ...


C = TypeVar("C", bound=TrackConfig)
S = TypeVar("S", bound=TrackerState)
R = TypeVar("R", bound=TrackResult)


class SiameseTracker(BaseTracker[C, S, R]):
    @staticmethod
    def _get_img(
        im: npt.NDArray[np.uint8],
        center_pos: Point[float],
        original_size: int,
        avg_chans: npt.NDArray[np.float64],
    ) -> tuple[
        torch.Tensor,
        tuple[int, int, int, int],
        tuple[int, int, int, int],
        torch.Tensor,
    ]:
        if not isinstance(im, np.ndarray):
            raise TypeError(
                "[IMP_ERR] failed to handle non-ndarray im in method `SiameseTracker._get_img`"
            )
            # im_tensor = im_tensor.squeeze(0).permute(1, 2, 0) # from somewhere
        else:
            im_tensor = torch.from_numpy(im).float()
            avg_chans_tensor = torch.from_numpy(avg_chans)

        sz = original_size
        im_sz = im_tensor.size()
        c = (original_size + 1) / 2
        context_xmin: int = np.floor(center_pos.x - c + 0.5)
        context_xmax: int = context_xmin + sz - 1
        context_ymin: int = np.floor(center_pos.y - c + 0.5)
        context_ymax: int = context_ymin + sz - 1
        left_pad = int(max(0.0, -context_xmin))
        top_pad = int(max(0.0, -context_ymin))
        right_pad = int(max(0.0, context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0.0, context_ymax - im_sz[0] + 1))
        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        im_h, im_w, im_c = im.shape
        im_patch: torch.Tensor
        te_im = torch.zeros(
            im_h + top_pad + bottom_pad, im_w + left_pad + right_pad, im_c
        )
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            te_im[
                top_pad : top_pad + im_h, left_pad : left_pad + im_w, :
            ] = im_tensor
            if top_pad:
                te_im[
                    0:top_pad, left_pad : left_pad + im_w, :
                ] = avg_chans_tensor
            if bottom_pad:
                te_im[
                    im_h + top_pad :, left_pad : left_pad + im_w, :
                ] = avg_chans_tensor
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans_tensor
            if right_pad:
                te_im[:, im_w + left_pad :, :] = avg_chans_tensor
            im_patch = te_im[
                int(context_ymin) : int(context_ymax + 1),
                int(context_xmin) : int(context_xmax + 1),
                :,
            ]
        else:
            im_patch = im_tensor[
                int(context_ymin) : int(context_ymax + 1),
                int(context_xmin) : int(context_xmax + 1),
                :,
            ]

        if im_patch.shape[0] == 0 or im_patch.shape[1] == 0:
            raise ValueError(
                f"im_patch.shape[0] == 0 or im_patch.shape[1] == 0, im_patch.shape: {im_patch.shape}"
            )

        return (
            im_patch,
            (left_pad, top_pad, right_pad, bottom_pad),
            (context_xmin, context_xmax, context_ymin, context_ymax),
            te_im,
        )
