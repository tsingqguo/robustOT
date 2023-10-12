import numpy as np
import numpy.typing as npt

from msot.utils.region.utils import _bbox_overlap_ratio


def convert_bbox_to_center(bboxes) -> npt.NDArray[np.float64]:
    return np.array(
        [
            (bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
            (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2),
        ],
        dtype=np.float64,
    ).T


def success_overlap(
    gt_bboxes: npt.NDArray[np.float_],
    pred_bboxes: npt.NDArray[np.float_],
) -> npt.NDArray[np.float64]:
    assert gt_bboxes.shape == pred_bboxes.shape
    f_cnt = len(gt_bboxes)

    mask = np.sum(gt_bboxes[:, 2:] > 0, axis=1) == 2

    iou = np.ones(f_cnt) * (-1)
    iou[mask] = _bbox_overlap_ratio(gt_bboxes[mask], pred_bboxes[mask])

    thlds = np.arange(0, 1.05, 0.05)
    success = np.zeros(len(thlds))

    for i in range(len(thlds)):
        success[i] = np.sum(iou > thlds[i]) / f_cnt

    return success


def success_error(
    gt_centers: npt.NDArray[np.float64],
    pred_centers: npt.NDArray[np.float64],
    thlds: np.ndarray,
) -> npt.NDArray[np.float64]:
    assert gt_centers.shape == pred_centers.shape
    f_cnt = len(gt_centers)

    mask = np.sum(gt_centers > 0, axis=1) == 2

    dist = np.ones(len(gt_centers)) * (-1)
    dist[mask] = np.sqrt(
        np.sum(np.power(gt_centers[mask] - pred_centers[mask], 2), axis=1)
    )

    success = np.zeros(len(thlds))

    for i in range(len(thlds)):
        success[i] = np.sum(dist <= thlds[i]) / f_cnt

    return success
