import numpy as np
import torch
import visdom
from pysot.tracker.siamrpn_tracker_cfgmod import SiamRPNTracker
from attackers.spark.SPARK.cfg import CFG


class SiamRPNTracker_SPARK(SiamRPNTracker):
    def track(
        self,
        img: np.ndarray,
        x_crop: torch.Tensor = [],
        immutable_dummy=None,
        is_perturbed: bool = False,
    ):
        """
        unique for SPARK
        """
        if isinstance(img, np.ndarray):
            img_h, img_w = img.shape[0], img.shape[1]
        else:
            img_h, img_w = img.size()[2], img.size()[3]

        _x_crop, scale_z = self.get_x_crop(img)
        if not is_perturbed:
            x_crop = _x_crop

        res = self.get_res_from_x_crop(img, x_crop, scale_z, immutable_dummy)
        return res
