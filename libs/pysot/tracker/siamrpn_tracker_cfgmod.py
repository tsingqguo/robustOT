from __future__ import annotations
import numpy as np
import numpy.typing as npt
import torch.nn.functional as F
from pyotp.config.pysot import CFG
from pyotp.modules.process import ProcessVisualizer, TrackerInputProcess
from pyotp.utils import unwrap_or
from pysot.models import ModelBuilder
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker
from toolkit.datasets.video import SeqImg
from torch import Tensor
from typing import (
    Callable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import os
import torch
from pyotp import ENV

N = TypeVar("N", bound=np.ndarray)
PostProcessFn = Callable[[List[Tensor]], Tensor]


class SiamRPNTracker(SiameseTracker):
    score: npt.NDArray[np.float32]
    trk_pp_list: List[TrackerInputProcess]
    trk_pp_visualizer: ProcessVisualizer

    def __init__(self, model: ModelBuilder):
        super(SiamRPNTracker, self).__init__()
        self.score_size = (
            (CFG.track.instance_size - CFG.track.exemplar_size)
            // CFG.anchor.stride
            + 1
            + CFG.track.base_size
        )
        self.anchor_num = len(CFG.anchor.ratios) * len(CFG.anchor.scales)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()
        #
        self.trk_pp_list = []
        self._saved = {}
        # data collection from BaseTracker
        self.data_collector = {}

    @staticmethod
    def generate_anchor(score_size: int) -> npt.NDArray[np.float32]:
        anchors = Anchors(
            CFG.anchor.stride, CFG.anchor.ratios, CFG.anchor.scales
        )
        anchor: np.ndarray = anchors.anchors  # type: ignore
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack(
            [(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1
        )
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = -(score_size // 2) * total_stride
        xx, yy = np.meshgrid(
            [ori + total_stride * dx for dx in range(score_size)],
            [ori + total_stride * dy for dy in range(score_size)],
        )
        xx, yy = (
            np.tile(xx.flatten(), (anchor_num, 1)).flatten(),
            np.tile(yy.flatten(), (anchor_num, 1)).flatten(),
        )
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(
            np.float32
        )
        return anchor

    @staticmethod
    def _convert_bbox(
        _delta: Tensor, anchor: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        _delta: `&Tensor`
        anchor: `&ndarray`
        """
        _delta = _delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta: npt.NDArray[np.float32] = _delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    @staticmethod
    def _convert_score(score: Tensor) -> npt.NDArray[np.float32]:
        """
        score: `&Tensor`
        """
        score = (
            score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        )
        return F.softmax(score, dim=1).data[:, 1].cpu().numpy()

    @staticmethod
    def _bbox_clip(
        cx: int, cy: int, width: int, height: int, boundary: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    # get_sx_sz_scalez
    @staticmethod
    def get_sizes(
        tracker: SiamRPNTracker
    ) -> Tuple[float, float, float]:
        """
        self: `&self`

        obtain sizes for **current frame**
        returns: `x_size`, `z_size`, `z_scale`
        """
        # w_z = tracker.size[0] + CFG.track.context_amount * np.sum(tracker.size)
        # h_z = tracker.size[1] + CFG.track.context_amount * np.sum(tracker.size)
        w_z = tracker.size[0] + CFG.track.context_amount * np.sum(tracker.size)
        h_z = tracker.size[1] + CFG.track.context_amount * np.sum(tracker.size)
        #
        # print('')
        # print('[debug] tracker.size:', tracker.size)
        # print('[debug] w_z:', w_z)
        # print('[debug] h_z:', h_z)
        z_size = np.sqrt(w_z * h_z)
        z_scale = CFG.track.exemplar_size / z_size
        #
        x_size = z_size * (CFG.track.instance_size / CFG.track.exemplar_size)
        # print('[debug] x_size:', x_size)
        # print('[debug] z_size:', z_size)
        # print('[debug] z_scale:', z_scale)

        return x_size, z_size, z_scale

    def get_z_crop(
        self, # bbox in format (lt_x, lt_y, w, h)
        img: SeqImg,
        bbox: List,
        immutable_dummy: Optional[SiamRPNTracker],
    ) -> Tensor:
        # if immutable_dummy is not None:
        #     mut_tracker = immutable_dummy
        # else:
        #     mut_tracker = self
        mut_tracker = unwrap_or(immutable_dummy, self)

        mut_tracker.center_pos = np.array(
            [bbox[0] + (bbox[2] - 1) / 2, bbox[1] + (bbox[3] - 1) / 2]
        )
        mut_tracker.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        (
            _,
            z_size,
            _,
        ) = self.get_sizes(mut_tracker)

        # calculate channel average
        mut_tracker.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(
            img,
            mut_tracker.center_pos,
            CFG.track.exemplar_size,
            round(z_size),
            mut_tracker.channel_average,
        )
        return z_crop

    def get_x_crop(
        self, img: SeqImg, immutable_dummy: Optional[SiamRPNTracker] = None
    ) -> Tuple[Tensor, float]:
        """
        self: `&self`
        """
        mut_tracker = unwrap_or(immutable_dummy, self)

        # print('')
        # print(f'[debug] tracker.size: {mut_tracker.size}; center: {mut_tracker.center_pos}')

        x_size, _, scale = self.get_sizes(mut_tracker)

        # print('')
        # print('[debug]  dummy:', immutable_dummy)
        # print('[debug] x_size:', x_size)

        x_crop = self.get_subwindow(
            img,
            mut_tracker.center_pos,
            CFG.track.instance_size,
            round(x_size),
            mut_tracker.channel_average,
        )
        return x_crop, scale

    def get_res_from_x_crop(
        self,
        img: SeqImg,
        x_crop: Tensor,
        scale: Union[npt.NDArray[np.float64], float],  # TODO: verify this
        immutable_dummy: Optional[SiamRPNTracker],
    ):
        # if immutable_dummy is not None:
        #     mut_tracker = immutable_dummy
        # else:
        #     mut_tracker = self
        mut_tracker = unwrap_or(immutable_dummy, self)
        mut_tracker._saved["x_crop_trkpp_free"] = x_crop.detach().cpu().clone()

        for i, trk_pp in enumerate(self.trk_pp_list):
            if i == 0:
                self.trk_pp_visualizer.add(x_crop, "clean")
            x_crop = trk_pp(x_crop)
            self.trk_pp_visualizer.add(x_crop, trk_pp.name)
        self.trk_pp_visualizer.save()

        outputs = self.model.track(x_crop)
        mut_tracker._saved["x_crop"] = x_crop.detach().cpu()
        """
        outputs['cls']:Tensor [B, 2 * anchors.len, 25, 25]
        outputs['loc']:Tensor [B, 4 * anchors.len, 25, 25]
        """

        # # REMOVE THIS:
        # _name = ""
        # for i, trk_pp in enumerate(self.trk_pp_list):
        #     if trk_pp.name:
        #         _name = trk_pp.name
        #
        # save_dir = os.path.join(
        #     ENV.experiments_path,
        #     "verification",
        #     f"{_name}_{self.trk_pp_visualizer.suffix}",
        #     self.trk_pp_visualizer._video.name,
        # )
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir, exist_ok=True)
        # torch.save(
        #     {
        #         "cls": outputs["cls"].detach().cpu(),
        #         "loc": outputs["loc"].detach().cpu(),
        #     },
        #     os.path.join(save_dir, f"{self.trk_pp_visualizer._idx}.pt"),
        # )
        # #

        score = self._convert_score(outputs["cls"])
        pred_bbox = self._convert_bbox(outputs["loc"], self.anchors)
        """
        score:     [25 * 25 * anchors.len,]
        pred_bbox: [4, 25 * 25 * anchors.len]
        """

        def change(r: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            return np.maximum(r, 1.0 / r)

        def sz(w: N, h: N) -> N:
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))  # type: ignore

        # scale penalty
        s_c = change(
            sz(pred_bbox[2, :], pred_bbox[3, :])
            / (sz(self.size[0] * scale, self.size[1] * scale))
        )

        # aspect ratio penalty
        r_c = change(
            (self.size[0] / self.size[1]) / (pred_bbox[2, :] / pred_bbox[3, :])
        )
        penalty: npt.NDArray[np.float32] = np.exp(
            -(r_c * s_c - 1) * CFG.track.penalty_k
        )
        """[25 * 25 * anchors.len,]"""
        pscore: npt.NDArray[np.float32] = penalty * score

        best_idx = np.argmax(
            # window penalty
            pscore * (1 - CFG.track.window_influence)
            + self.window * CFG.track.window_influence
        )
        # print('track.window_influence', CFG.track.window_influence)

        bbox = pred_bbox[:, best_idx] / scale
        lr = penalty[best_idx] * score[best_idx] * CFG.track.lr

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(
            cx, cy, width, height, img.shape[:2]
        )

        # update state
        mut_tracker.center_pos = np.array([cx, cy])
        mut_tracker.size = np.array([width, height])
        mut_tracker.score = score

        # print(f'[DBG] content of bbox(cx;cy):\n{bbox}')
        # self.trk_pp_visualizer.add_bbox_to_last(
        #     [
        #         int(CFG.track.instance_size / scale) - scale * bbox[0],
        #         int(CFG.track.instance_size / scale) - scale * bbox[1],
        #         # int((CFG.track.instance_size / scale) // 2),
        #         # int((CFG.track.instance_size / scale) // 2),
        #         int(1),
        #         int(1),
        #     ]
        #     # [
        #     #     int((CFG.track.instance_size / scale) // 2 - scale * bbox[0]),
        #     #     int((CFG.track.instance_size / scale) // 2 - scale * bbox[1]),
        #     #     int(scale * width),
        #     #     int(scale * height),
        #     # ]
        # )
        bbox = [cx - width / 2, cy - height / 2, width, height]
        # self.trk_pp_visualizer.add_bbox_to_last(bbox)
        best_score = score[best_idx]

        # print(f'[DBG] content of bbox(cx;cy):\n{bbox}')

        return {
            "bbox": bbox,
            "best_score": best_score,
            "scale": scale,
        }

    def feed_model_template(self, z_crop: Tensor) -> None:
        self.model.template(z_crop)
        self._saved["z_crop"] = z_crop.detach().cpu()

    def init(
        self,
        img: SeqImg,
        bbox, # bbox in format (lt_x, lt_y, w, h)
        immutable_dummy: Optional[Self],  # type:ignore PEP 673
    ) -> None:
        z_crop = self.get_z_crop(img, bbox, immutable_dummy)
        # z_crop = self.get_z_crop(img, bbox, None) # TODO: otherwise get_sizes with invalid
        self.feed_model_template(z_crop)

    def track(
        self,
        img: SeqImg,
        immutable_dummy: Optional[Self],  # type:ignore PEP 673
    ):
        """
        immutable_dummy: `Option<&mut SiamRPNTracker>`
        """
        x_crop, scale = self.get_x_crop(img, immutable_dummy)
        res = self.get_res_from_x_crop(img, x_crop, scale, immutable_dummy)
        return res
