from __future__ import annotations
import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from pyotp.config.pysot import CFG
from pyotp.utils import tensor2mat
from pysot.models import ModelBuilder
from toolkit.datasets.video import SeqImg
from torch import Tensor
from typing import Any, Dict, List, Tuple, TypeVar, Union
from typing_extensions import Self


_T = TypeVar("_T", bound="BaseTracker")
"""for dummy dump"""


class BaseTracker:
    _is_dummy: bool = False
    anchors: npt.NDArray[np.float32]
    # anchor_num: int
    center_pos: npt.NDArray[np.float64]
    """
    set by track init -> `get_z_crop`
    """
    channel_average: npt.NDArray[np.float64]
    model: ModelBuilder
    size: npt.NDArray[np.float64]

    # data collection
    _saved: Dict[str, Any]

    def collect_saved(self) -> Dict[str, Any]:
        """
        return saved data and clean `self._saved`
        """
        data = self._saved
        self._saved = {}
        return data

    def init(self, img: np.ndarray, bbox: List[int]):
        # TODO: replace bbox from list to tuple
        """
        args:
            img(np.ndarray): BGR image
            bbox(list): [x, y, width, height]
        """
        raise NotImplementedError

    def _convert_bbox(
        self, delta: Tensor, anchor: npt.NDArray[np.float32]
    ) -> Tensor:
        raise NotImplementedError

    def _convert_score(self, score: Tensor) -> Tensor:
        raise NotImplementedError

    def track(self, img: np.ndarray):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        raise NotImplementedError

    @staticmethod
    def get_subwindow(
        im: SeqImg,
        pos: npt.NDArray[np.float64],  # self.center_pos
        model_sz: int,
        original_sz: int,
        avg_chans: npt.NDArray[np.float64],
    ) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def get_origin_img(
        im: np.ndarray,
        im_patch: Tensor,
        pos: npt.NDArray[np.float64],
        model_sz: int,
        original_sz: int,
        avg_chans: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.uint8]:
        raise NotImplementedError

    @classmethod
    def create_dummy(cls) -> Self:
        inst = cls.__new__(cls)
        inst._is_dummy = True
        inst._saved = {}
        return inst

    def clone(self):
        ...

    @staticmethod
    def dump_dummy_attrs(dummy: _T, target: _T) -> List[str]:
        if not dummy._is_dummy:
            raise TypeError(
                "failed to dump dummy attrs from non-dummy instance"
            )
        restored: List[str] = []
        for attr in dir(dummy):
            if attr.startswith("__"):
                continue
            attr_val = getattr(dummy, attr)
            if attr_val.__class__.__name__ in ["method", "function"]:
                continue
            if attr == "_is_dummy":
                continue
            setattr(target, attr, attr_val)
            restored.append(attr)
        return restored

    def load_attrs_from_dummy(
        self,
        dummy: Self,  # type:ignore PEP 673
    ):
        return self.dump_dummy_attrs(dummy, self)


class SiameseTracker(BaseTracker):
    @staticmethod
    def _get_img(
        im: SeqImg,
        pos: Union[float, list[float], npt.NDArray[np.float64]],
        original_size: int,
        avg_chans: npt.NDArray[np.float64],
    ) -> Tuple[
        Tensor, Tuple[int, int, int, int], Tuple[int, int, int, int], Tensor
    ]:
        if not isinstance(im, np.ndarray):
            raise TypeError(
                "[IMP_ERR] failed to handle non-ndarray im in method `SiameseTracker._get_img`"
            )
            # im_tensor = im_tensor.squeeze(0).permute(1, 2, 0) # from somewhere
        else:
            im_tensor = torch.from_numpy(im).float()
            avg_chans_tensor = torch.from_numpy(avg_chans)

        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_size
        im_sz = im_tensor.size()
        c = (original_size + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin: int = np.floor(pos[0] - c + 0.5)
        context_xmax: int = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin: int = np.floor(pos[1] - c + 0.5)
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
        # r, c, k = im.shape
        im_patch: Tensor
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

    @staticmethod
    def get_subwindow(
        im: SeqImg,
        pos: npt.NDArray[np.float64],  # self.center_pos
        model_size: int,
        original_size: int,
        avg_chans: npt.NDArray[np.float64],
    ) -> Tensor:
        im_patch, _, _, _ = SiameseTracker._get_img(
            im, pos, original_size, avg_chans
        )
        im_patch = im_patch.permute(2, 0, 1)
        im_patch = im_patch.unsqueeze(0)
        if CFG.cuda:
            im_patch = im_patch.cuda()
        if not np.array_equal(model_size, original_size):
            im_patch = F.interpolate(
                im_patch,
                size=(model_size, model_size),
                mode="bilinear",
                align_corners=False,
            )
        return im_patch

    @staticmethod
    def get_origin_img(
        im: np.ndarray,
        im_patch: Tensor,
        pos: npt.NDArray[np.float64],
        model_size: int,
        original_size: int,
        avg_chans: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.uint8]:
        im = im.copy()
        tim_patch, pad, contexts, te_im = SiameseTracker._get_img(
            im, pos, original_size, avg_chans
        )
        top_pad, bottom_pad, left_pad, right_pad = pad
        context_xmin, context_xmax, context_ymin, context_ymax = contexts

        if im_patch.shape[0] != 1:
            raise ValueError("im_patch.shape[0] != 1")
        im_patch_mat = tensor2mat(im_patch)
        te_im = torch.squeeze(te_im)
        te_im_nd = te_im.detach().cpu().numpy()
        te_im_nd: npt.NDArray[np.uint8] = te_im_nd.astype(np.uint8)

        r, c, _ = im.shape

        if not np.array_equal(model_size, original_size):
            im_patch_mat = cv2.resize(
                im_patch_mat, (tim_patch.shape[0], tim_patch.shape[1])
            )

        if any([top_pad, bottom_pad, left_pad, right_pad]):
            te_im_nd[
                int(context_ymin) : int(context_ymax + 1),
                int(context_xmin) : int(context_xmax + 1),
                :,
            ] = im_patch
            im = te_im_nd[top_pad : top_pad + r, left_pad : left_pad + c, :]
        else:
            im[
                int(context_ymin) : int(context_ymax + 1),
                int(context_xmin) : int(context_xmax + 1),
                :,
            ] = im_patch_mat

        return im
