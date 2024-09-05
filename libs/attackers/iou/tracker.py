import numpy as np
from pysot.tracker.siamrpn_tracker_cfgmod import SiamRPNTracker
from pyotp.config.pysot import PYSOT_Config


class IoU_SiamRPNTracker:
    raw: SiamRPNTracker
    cfg: PYSOT_Config

    def __init__(self, tracker: SiamRPNTracker, cfg: PYSOT_Config) -> None:
        self.raw = tracker
        self.cfg = cfg

    def track_fixed(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        tracker = self.raw
        w_z = tracker.size[0] + self.cfg.track.context_amount * np.sum(
            tracker.size
        )
        h_z = tracker.size[1] + self.cfg.track.context_amount * np.sum(
            tracker.size
        )
        s_z = np.sqrt(w_z * h_z)
        scale_z = self.cfg.track.exemplar_size / s_z
        s_x = s_z * (
            self.cfg.track.instance_size / self.cfg.track.exemplar_size
        )
        x_crop = tracker.get_subwindow(
            img,
            tracker.center_pos,
            self.cfg.track.instance_size,
            round(s_x),
            tracker.channel_average,
        )

        outputs = tracker.model.track(x_crop)

        score = tracker._convert_score(outputs["cls"])
        pred_bbox = tracker._convert_bbox(outputs["loc"], tracker.anchors)

        def change(r):
            return np.maximum(r, 1.0 / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(
            sz(pred_bbox[2, :], pred_bbox[3, :])
            / (sz(tracker.size[0] * scale_z, tracker.size[1] * scale_z))
        )

        # aspect ratio penalty
        r_c = change(
            (tracker.size[0] / tracker.size[1])
            / (pred_bbox[2, :] / pred_bbox[3, :])
        )
        penalty = np.exp(-(r_c * s_c - 1) * self.cfg.track.penalty_k)
        pscore = penalty * score

        # window penalty
        pscore = (
            pscore * (1 - self.cfg.track.window_influence)
            + tracker.window * self.cfg.track.window_influence
        )
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * self.cfg.track.lr

        cx = bbox[0] + tracker.center_pos[0]
        cy = bbox[1] + tracker.center_pos[1]

        # smooth bbox
        width = tracker.size[0] * (1 - lr) + bbox[2] * lr
        height = tracker.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = tracker._bbox_clip(
            cx, cy, width, height, img.shape[:2]
        )

        # udpate state
        # tracker.center_pos = np.array([cx, cy])
        # tracker.size = np.array([width, height])

        bbox = [cx - width / 2, cy - height / 2, width, height]
        best_score = score[best_idx]
        return {"bbox": bbox, "best_score": best_score}
