from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from msot.trackers.base import (
    BaseTracker,
    TrackConfig,
    TrackerState,
    TrackResult,
)
from msot.trackers.siamese.siamrpnpp import SiamRPNTracker
from msot.utils.region import Bbox, Center, Point
from msot.utils.region.utils import bbox_overlap_ratio

from .utils import rpn_smoothL1


Tracker: TypeAlias = BaseTracker[TrackConfig, TrackerState, TrackResult]


def center_to_delta(
    ct: Center,
    anchor: npt.NDArray[np.float32],
    scale: float,
    center_pos: Point,
    perturb: bool = False,
):
    ct = ct - center_pos

    if perturb:
        rate_xy1 = np.random.uniform(0.3, 0.5)
        rate_xy2 = np.random.uniform(0.3, 0.5)
        rate_wd = np.random.uniform(0.7, 0.9)

        ct = Center(
            ct.cx - ct.w * rate_xy1,
            ct.cy - ct.h * rate_xy2,
            ct.w * rate_wd,
            ct.h * rate_wd,
        )

    delta = np.tile(np.array(ct.unpack2center()) * scale, (anchor.shape[0], 1))

    delta[:, 0] = (delta[:, 0] - anchor[:, 0]) / anchor[:, 2]
    delta[:, 1] = (delta[:, 1] - anchor[:, 1]) / anchor[:, 3]

    delta[:, 2] = np.log(delta[:, 2]) / anchor[:, 2]
    delta[:, 3] = np.log(delta[:, 3]) / anchor[:, 3]

    return delta


def perturb_forward(
    tracker: BaseTracker,
    search: torch.Tensor,
    scale: float,
    gt_bbox: Bbox,
):
    if not isinstance(tracker, SiamRPNTracker):
        raise TypeError("tracker must be SiamRPNTracker")

    search = search.cuda()

    xf = tracker.model.backbone(search)

    if tracker.model.config.mask.mask:
        raise NotImplementedError

    if tracker.model.neck is not None:
        xf = tracker.model.neck(xf)

    cls: torch.Tensor
    loc: torch.Tensor

    cls, loc = tracker.model.rpn_head.forward(tracker.state.z_feat.get(), xf)

    ## ----------------- ##
    iou_hi = 0.6
    iou_low = 0.3

    cvt_ct = tracker._convert_bbox(loc, tracker.state.anchors.val)
    cvt_ct = cvt_ct / scale
    cvt_ct = Center(*cvt_ct)
    cvt_ct += tracker.state.center.val

    label = bbox_overlap_ratio(
        np.array(cvt_ct.unpack2bbox()),
        np.array(gt_bbox.unpack2bbox()),
    )

    y_pos = np.where(label > iou_hi, 1, 0)
    y_pos = torch.from_numpy(y_pos).cuda().long()
    y_neg = np.where(label < iou_low, 1, 0)
    y_neg = torch.from_numpy(y_neg).cuda().long()
    pos_index = np.where(y_pos.cpu() == 1)
    neg_index = np.where(y_neg.cpu() == 0)
    index = np.concatenate((pos_index, neg_index), axis=1)

    # make pseudo lables
    y_pos_pseudo = np.where(label > iou_hi, 0, 1)
    y_pos_pseudo = torch.from_numpy(y_pos_pseudo).cuda().long()
    y_neg_pseudo = np.where(label < iou_low, 1, 0)
    y_neg_pseudo = torch.from_numpy(y_neg_pseudo).cuda().long()

    y_truth = y_pos
    y_pseudo = y_pos_pseudo

    score = cls.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)

    cls_truth_loss = -F.cross_entropy(score[index], y_truth[index])
    cls_pseudo_loss = -F.cross_entropy(score[index], y_pseudo[index])
    cls_loss = (cls_truth_loss - cls_pseudo_loss) * (1)
    ## ----------------- ##
    delta1 = loc.permute(1, 2, 3, 0).contiguous().view(4, -1)
    # gt_center = gt_corner.to_center()
    gt_center = gt_bbox.to_center()
    gt_cen = center_to_delta(
        gt_center,
        tracker.state.anchors.get(),
        scale,
        tracker.state.center.get(),
        perturb=False,
    )
    gt_cen_pseudo = center_to_delta(
        gt_center,
        tracker.state.anchors.get(),
        scale,
        tracker.state.center.get(),
        perturb=True,
    )

    loc_truth_loss = -rpn_smoothL1(delta1, gt_cen, y_pos)
    loc_pseudo_loss = -rpn_smoothL1(delta1, gt_cen_pseudo, y_pos)
    loc_loss = (loc_truth_loss - loc_pseudo_loss) * (5)

    total_loss = cls_loss + loc_loss

    outputs = {
        "label": label,
        "total_loss": total_loss,
        "cls_loss": cls_loss,
        "cls_truth_loss": cls_truth_loss,
        "cls_pseudo_loss": cls_pseudo_loss,
        "loc_loss": loc_loss,
        "loc_truth_loss": loc_truth_loss,
        "loc_pseudo_loss": loc_pseudo_loss,
    }

    if tracker.model.config.mask.mask:
        raise NotImplementedError("mask loss not implemented yet")

    return outputs
