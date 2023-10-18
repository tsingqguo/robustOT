import numpy as np
import numpy.typing as npt


def _bbox_overlap_ratio(
    bbox1: npt.NDArray[np.float_], bbox2: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:

    if bbox1.ndim == 1:
        bbox1 = bbox1[None, :]
    if bbox2.ndim == 1:
        bbox2 = bbox2[None, :]

    left = np.maximum(bbox1[:, 0], bbox2[:, 0])
    right = np.minimum(bbox1[:, 0] + bbox1[:, 2], bbox2[:, 0] + bbox2[:, 2])
    top = np.maximum(bbox1[:, 1], bbox2[:, 1])
    bottom = np.minimum(bbox1[:, 1] + bbox1[:, 3], bbox2[:, 1] + bbox2[:, 3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = bbox1[:, 2] * bbox1[:, 3] + bbox2[:, 2] * bbox2[:, 3] - intersect
    iou = np.clip(intersect / union, 0, 1)

    return iou


def bbox_overlap_ratio(
    bbox1: npt.NDArray[np.float_], bbox2: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    bbox1 = np.transpose(bbox1)

    if bbox1.ndim == 1:
        bbox1 = bbox1[None, :]
    if bbox2.ndim == 1:
        bbox2 = bbox2[None, :]

    left = np.maximum(bbox1[:, 0], bbox2[:, 0])
    right = np.minimum(bbox1[:, 0] + bbox1[:, 2], bbox2[:, 0] + bbox2[:, 2])
    top = np.maximum(bbox1[:, 1], bbox2[:, 1])
    bottom = np.minimum(bbox1[:, 1] + bbox1[:, 3], bbox2[:, 1] + bbox2[:, 3])

    intersect = np.maximum(0, right - left + 1) * np.maximum(
        0, bottom - top + 1
    )
    union = (
        (bbox1[:, 2] + 1) * (bbox1[:, 3] + 1)
        + (bbox2[:, 2] + 1) * (bbox2[:, 3] + 1)
        - intersect
    )
    iou = np.clip(intersect / union, 0, 1)

    return iou
