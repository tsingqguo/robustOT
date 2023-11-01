from __future__ import annotations
from typing import Type

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from msot.libs.pysot.utils.anchor import Anchors, AnchorsCfg
from msot.models import TModel, TModelResult
from msot.utils.dataship import DataCTR as DC
from msot.utils.region import Box, Center, Point, Polygon, Region

from .. import (
    ScaledCrop,
    SiameseTracker,
    TrackerState as _TrackerState,
    TrackResult as _TrackResult,
    TrackSize,
)
from .config import TrackConfig


class TrackerState(_TrackerState):
    # immutable
    anchors: DC[npt.NDArray[np.float32]]
    window: DC[npt.NDArray[np.floating]]  # TODO:
    #
    channel_average: DC[np.ndarray]
    score: DC[npt.NDArray[np.float32]]

    @property
    def valid_names(self) -> set[str]:
        return super().valid_names | {
            "anchors",
            "window",
            "channel_average",
            "score",
        }

    def __init__(self) -> None:
        super().__init__()
        self.anchors = DC(
            is_shared=True,
            is_mutable=False,
            allow_unbound=True,
        )
        self.window = DC(
            is_shared=True,
            is_mutable=False,
            allow_unbound=True,
        )
        #
        self.channel_average = DC(
            is_shared=False,
            is_mutable=True,
            allow_unbound=True,
        )
        self.score = DC(
            is_shared=False,
            is_mutable=True,
            allow_unbound=True,
        )

    def reset(self) -> None:
        super().reset()
        self.channel_average.unbind()
        self.score.unbind()


class TrackResult(_TrackResult):
    ...


class SiamRPNTracker(SiameseTracker[TrackConfig, TrackerState, TrackResult]):
    def __init__(
        self,
        model: TModel,
        config: TrackConfig,
        state_cls: Type[TrackerState],
        state: TrackerState | None = None,
    ) -> None:
        super().__init__(model, config, state_cls, state)
        is_window_unbound = self.state.window.is_unbound()
        is_anchors_unbound = self.state.anchors.is_unbound()
        if is_window_unbound or is_anchors_unbound:
            score_size = (
                (self.config.instance_size - self.config.exemplar_size)
                // self.config.anchor.stride
                + 1
                + self.config.base_size
            )
            if is_window_unbound:
                hanning = np.hanning(score_size)
                window = np.outer(hanning, hanning)
                self.state.window.update(
                    np.tile(window.flatten(), self.config.anchor.anchor_num)
                )
            if is_anchors_unbound:
                self.state.anchors.update(
                    self.generate_anchor(self.config.anchor, score_size)
                )

    @staticmethod
    def generate_anchor(
        cfg: AnchorsCfg, score_size: int
    ) -> npt.NDArray[np.float32]:
        anchors = Anchors(cfg)
        x1, y1, x2, y2 = (
            anchors.anchors[:, 0],
            anchors.anchors[:, 1],
            anchors.anchors[:, 2],
            anchors.anchors[:, 3],
        )
        anchor = np.stack(
            [(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1
        )
        total_stride = anchors.config.stride
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
    def get_scale(
        size: np.ndarray | tuple | list,
        exemplar_size: int,
        context_amount: float,
    ) -> tuple[float, float]:
        w_z = size[0] + context_amount * np.sum(size)
        h_z = size[1] + context_amount * np.sum(size)
        z_size = np.sqrt(w_z * h_z)

        return z_size, exemplar_size / z_size

    @classmethod
    def get_sizes(cls, cfg: TrackConfig, st: TrackerState) -> TrackSize:
        z_size, scale = cls.get_scale(
            st.size.get(),
            cfg.exemplar_size,
            cfg.context_amount,
        )
        x_size = z_size * (cfg.instance_size / cfg.exemplar_size)

        return TrackSize(
            z_size=round(z_size), x_size=round(x_size), scale=scale
        )

    @classmethod
    def get_subwindow(
        cls,
        im: npt.NDArray[np.uint8],
        center_pos: Point,
        input_size: int,
        original_size: int,
        avg_chans: npt.NDArray[np.float_],
        device: torch.device,
    ) -> torch.Tensor:
        im_patch, _, _, _ = cls._get_img(
            im, center_pos, original_size, avg_chans
        )
        im_patch = im_patch.permute(2, 0, 1)
        im_patch = im_patch.unsqueeze(0)
        im_patch = im_patch.to(device)
        if not np.array_equal(input_size, original_size):
            im_patch = F.interpolate(
                im_patch,
                size=(input_size, input_size),
                mode="bilinear",
                align_corners=False,
            )
        return im_patch

    @staticmethod
    def _convert_bbox(
        _delta: torch.Tensor, anchor: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        _delta = _delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta: npt.NDArray[np.float32] = _delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    @staticmethod
    def _convert_score(score: torch.Tensor) -> npt.NDArray[np.float32]:
        score = (
            score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        )
        return F.softmax(score, dim=1).data[:, 1].cpu().numpy()

    @staticmethod
    def _center_box_clip(
        cx: float,
        cy: float,
        width: float,
        height: float,
        boundary: tuple,
    ) -> Center:
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return Center(cx, cy, width, height)

    @classmethod
    def get_template(
        cls,
        cfg: TrackConfig,
        st: TrackerState,
        img: npt.NDArray[np.uint8],
        gt: Region,
        device: torch.device,
    ) -> ScaledCrop:
        if isinstance(gt, Box):
            ...
        elif isinstance(gt, Polygon):
            gt = gt.to_corner()
        else:
            raise NotImplementedError

        st.center.update(gt.center)
        st.size.update(gt.size)
        st.channel_average.update(np.mean(img, axis=(0, 1)))

        size = cls.get_sizes(cfg, st)

        z_crop = cls.get_subwindow(
            img,
            st.center.val,
            cfg.exemplar_size,
            size.z_size,
            st.channel_average.val,
            device=device,
        )
        return ScaledCrop(z_crop, size)

    @classmethod
    def get_search(
        cls,
        cfg: TrackConfig,
        st: TrackerState,
        img: npt.NDArray[np.uint8],
        device: torch.device,
    ) -> ScaledCrop:
        size = cls.get_sizes(cfg, st)

        x_crop = cls.get_subwindow(
            img,
            st.center.val,
            cfg.instance_size,
            size.x_size,
            st.channel_average.val,
            device=device,
        )
        return ScaledCrop(x_crop, size)

    @classmethod
    def get_res_from_x_crop(
        cls,
        cfg: TrackConfig,
        st: TrackerState,
        scale: float,
        model_out: TModelResult,
        frame_size: tuple[int, ...],
    ) -> TrackResult:
        score = cls._convert_score(model_out.cls)
        pred_bbox = cls._convert_bbox(model_out.loc, st.anchors.val)
        """
        score:     [25 * 25 * anchors.len,]
        pred_bbox: [4, 25 * 25 * anchors.len]
        """

        def change(r: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            return np.maximum(r, 1.0 / r)

        def sz(
            w: npt.NDArray[np.float32], h: npt.NDArray[np.float32]
        ) -> npt.NDArray[np.float32]:
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(
            sz(pred_bbox[2, :], pred_bbox[3, :])
            / (sz(st.size.val[0] * scale, st.size.val[1] * scale))
        )

        # aspect ratio penalty
        r_c = change(
            (st.size.val[0] / st.size.val[1])
            / (pred_bbox[2, :] / pred_bbox[3, :])
        )
        penalty: npt.NDArray[np.float32] = np.exp(
            -(r_c * s_c - 1) * cfg.penalty_k
        )
        """[25 * 25 * anchors.len,]"""
        pscore: npt.NDArray[np.float32] = penalty * score

        best_idx = np.argmax(
            pscore * (1 - cfg.window_influence)
            + st.window.val * cfg.window_influence
        )

        bbox = pred_bbox[:, best_idx] / scale
        lr = penalty[best_idx] * score[best_idx] * cfg.lr

        # TODO:
        cx = bbox[0] + st.center.get().x
        cy = bbox[1] + st.center.get().y

        # smooth bbox
        width = st.size.val[0] * (1 - lr) + bbox[2] * lr
        height = st.size.val[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cbox = cls._center_box_clip(cx, cy, width, height, frame_size)

        # update state
        st.center.update(cbox.center)
        st.size.update(cbox.size)
        st.score.update(score)

        bbox = cbox.to_bbox()
        best_score = score[best_idx]

        return TrackResult(output=bbox, best_score=best_score)

    def init(
        self, img: npt.NDArray[np.uint8], gt: Region
    ) -> tuple[ScaledCrop, None]:
        scaled_z = self.get_template(
            self.config, self.state, img, gt, self.device
        )
        return (
            scaled_z,
            self.init_with_scaled_template(scaled_z.crop),
        )

    def track_with_scaled_search(
        self, x_crop: torch.Tensor, scale: float, frame_size: tuple[int, ...]
    ) -> TrackResult:
        out = self.model.track(self.state.z_feat.val, x_crop)
        res = self.get_res_from_x_crop(
            self.config, self.state, scale, out, frame_size
        )
        return res

    def track(
        self, img: npt.NDArray[np.uint8]
    ) -> tuple[ScaledCrop, TrackResult]:
        scaled_x = self.get_search(self.config, self.state, img, self.device)
        return (
            scaled_x,
            self.track_with_scaled_search(
                scaled_x.crop, scaled_x.size.scale, img.shape[:2]
            ),
        )
