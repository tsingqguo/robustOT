import time
import torch
from LRR import LRR
from pyotp.config.pysot import CFG
from pyotp.modules.process import TrackerInputProcess
from pyotp.typing import HistoricalMgr
from pyotp.utils import logging
from typing import Callable, Optional

logging.init_logger("global", level=logging.LT.Debug, ignore_exist=True)
LOG = logging.get_logger("global")


class NoDefTrkPP(TrackerInputProcess):
    def __init__(
        self,
        name: str = "ND",
    ) -> None:
        super().__init__(name)

    def forward(
        self, img: torch.Tensor, historical: HistoricalMgr
    ) -> torch.Tensor:
        return img


class LRRTrkPP(TrackerInputProcess):
    net: LRR
    model_n: int
    frame_cnt: int
    target_index: int
    """in [0, frame_cnt]"""
    text_template: Optional[str | Callable[[str], str]]

    _latency: list[float]
    _skip_per: int
    _skip_cnt: int

    def __init__(
        self,
        saved_fp: str,
        frame_cnt: int,
        text_template: Optional[str | Callable[[str], str]],
        name: str = "cvr_trkpp",
    ) -> None:
        super().__init__(name)
        self._latency = []  # TODO:
        self.frame_cnt = frame_cnt
        self.target_index = -1  # TODO:

        self.text_template = text_template

        device = torch.device("cuda")
        saved = torch.load(saved_fp, map_location=device)
        self.net = LRR(
            device,
            saved,
            height=CFG.track.instance_size,
            width=CFG.track.instance_size,
        )

    # def reset(self) -> None:
    #     super().reset()
    #     print(f"\n[debug] CVRTrkPP: avg latency: {np.mean(self._latency)*1000:.3f}ms")
    #     self._latency = []

    def forward(
        self, img: torch.Tensor, historical: HistoricalMgr
    ) -> torch.Tensor:
        t0 = time.time()
        inputs: list[torch.Tensor] = []
        if self.frame_cnt > 1:
            history = historical.get_tracking_history_slice(
                -self.frame_cnt, -1
            )
            for trk in history:
                if trk.is_some():
                    trk = trk.unwrap()
                    if trk.x_crop_trkpp_free is not None:
                        inputs.append(
                            trk.x_crop_trkpp_free.to(img.device) / 255.0
                        )

        inputs.append(img / 255.0)
        frames = torch.stack(inputs, dim=1)
        """[B, N, C, H, W]"""
        if self.text_template is None:
            text = historical._video.name.lower()
        elif callable(self.text_template):
            text = self.text_template(historical._video.name.lower())
        else:
            text = self.text_template.format(historical._video.name.lower())

        img_def = self.net.forward(frames, text)

        img_def = img_def[:, self.target_index, :, :, :]

        img_def = img_def * 255
        self._latency.append(time.time() - t0)
        return img_def
