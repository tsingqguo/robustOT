import numpy as np
import torch
from typing import overload, Union


@overload
def iou_overlap(pred_bbox: list[float], gt_bbox: list[float]) -> float:
    ...


@overload
def iou_overlap(pred_bbox: np.ndarray, gt_bbox: np.ndarray) -> float:
    ...


@overload
def iou_overlap(
    pred_bbox: torch.Tensor, gt_bbox: torch.Tensor
) -> torch.Tensor:
    ...


def iou_overlap(pred_bbox, gt_bbox):
    if isinstance(pred_bbox, list):
        return _iou_overlap_np(pred_bbox, gt_bbox)
    elif isinstance(pred_bbox, np.ndarray):
        return _iou_overlap_np(pred_bbox, gt_bbox)
    elif isinstance(pred_bbox, torch.Tensor):
        return _iou_overlap_tensor(pred_bbox, gt_bbox)
    else:
        raise TypeError(f"pred_bbox type {type(pred_bbox)} not supported")


def _iou_overlap_tensor(
    pred_bbox: torch.Tensor, gt_bbox: torch.Tensor
) -> torch.Tensor:
    x_a = torch.max(pred_bbox[0], gt_bbox[0])
    y_a = torch.max(pred_bbox[1], gt_bbox[1])
    x_b = torch.min(pred_bbox[0] + pred_bbox[2], gt_bbox[0] + gt_bbox[2])
    y_b = torch.min(pred_bbox[1] + pred_bbox[3], gt_bbox[1] + gt_bbox[3])
    inter_area = torch.max(x_b - x_a + 1, torch.zeros_like(x_b)) * torch.max(
        y_b - y_a + 1, torch.zeros_like(y_b)
    )
    box_a_area = (pred_bbox[2] + 1) * (pred_bbox[3] + 1)
    box_b_area = (gt_bbox[2] + 1) * (gt_bbox[3] + 1)
    iou = inter_area / (box_a_area + box_b_area - inter_area)
    return iou


def _iou_overlap_np(
    pred_bbox: Union[list[float], np.ndarray],
    gt_bbox: Union[list[float], np.ndarray],
) -> float:
    pred_bb = np.array(pred_bbox)
    gt_bb = np.array(gt_bbox)
    x_a = max(pred_bb[0], gt_bb[0])
    y_a = max(pred_bb[1], gt_bb[1])
    x_b = min(pred_bb[0] + pred_bb[2], gt_bb[0] + gt_bb[2])
    y_b = min(pred_bb[1] + pred_bb[3], gt_bb[1] + gt_bb[3])
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    box_a_area = (pred_bb[2] + 1) * (pred_bb[3] + 1)
    box_b_area = (gt_bb[2] + 1) * (gt_bb[3] + 1)
    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    return iou
