import torch
import numpy as np
from pyotp.modules.process import CP, PP
from pyotp.typing import HistoricalMgr
from pyotp.utils import calculate_points_inf


class BBoxInf(PP):
    def __init__(self, name: str = "bbox_inf") -> None:
        super().__init__(name)

    def forward(
        self, input: torch.Tensor, historical: HistoricalMgr
    ) -> torch.Tensor:
        if historical.last.is_some():
            extra = historical.cur.extra

            x_cur, y_cur, _, _ = historical.cur.frame.aligned_bbox
            x_prev, y_prev, _, _ = historical.last.unwrap().frame.aligned_bbox

            distance, rot = calculate_points_inf(
                (x_prev, y_prev), (x_cur, y_cur)
            )

            extra.set_data("bbox_center_pts_distance", distance)
            extra.set_data("bbox_center_pts_rotation", rot)
            extra.set_data("bbox_center_pts_rotation_degree", np.rad2deg(rot))

        return input
