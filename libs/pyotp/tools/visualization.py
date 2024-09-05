import cv2
import os
import numpy as np
from enum import Enum
from pyotp.typing import (
    BBox,
    Historical,
    HistoricalMgr,
    NamedTuple,
    VisStyle,
)
from pyotp.utils import (
    calculate_points_inf,
    create_video,
    draw_arrow,
    merge_imgs,
    save_imgs,
    tensor2mat,
    unwrap_or,
)
from pyotp.utils.visualization import Color, COLORS, create_canvas, add_border
from pyotp.utils.option import NONE, Option, Some
from typing import Dict, List, Optional, Set, Tuple


class ValidVis(Enum):
    Frame = 0
    SearchRegion = 1
    All = 2


class _PotentialLoaded(NamedTuple):
    frames: Option[List[Option[np.ndarray]]]
    search_regions: Option[List[Option[np.ndarray]]]
    pred_bboxes: Option[List[Option[BBox]]]


def helper_compare_with(folder: str, video_name: str, length: int):
    pred_bboxes_path = os.path.join(folder, video_name, "pred_bboxes.npy")
    try:
        pred_bboxes = np.load(pred_bboxes_path, allow_pickle=True)
        pred_bboxes = Some(
            [Some(BBox(*x)) if x != 0 else NONE for x in pred_bboxes.tolist()]
        )
    except FileNotFoundError:
        pred_bboxes = NONE
    frames = []
    x_crops = []
    for i in range(length):
        frame_fp = os.path.join(folder, video_name, f"{i:05d}.jpg")
        if os.path.exists(frame_fp):
            frame = cv2.imread(
                frame_fp,
                # cv2.IMREAD_COLOR,
            )
            frames.append(Some(frame))
        else:
            frames.append(NONE)
        x_crop_fp = os.path.join(folder, video_name, f"{i:05d}_x.jpg")
        if os.path.exists(x_crop_fp):
            x_crop = cv2.imread(
                x_crop_fp,
                # cv2.IMREAD_COLOR,
            )
            x_crops.append(Some(x_crop))
        else:
            x_crops.append(NONE)
    frames = Some(frames) if any(x.is_some() for x in frames) else NONE
    x_crops = Some(x_crops) if any(x.is_some() for x in x_crops) else NONE
    return _PotentialLoaded(
        frames=frames,
        search_regions=x_crops,
        pred_bboxes=pred_bboxes,
    )


class TrackingVisualizer:
    _historical: HistoricalMgr
    _extra_bbox: Option[Dict[str, Tuple[List[Option[BBox]], Color]]]
    _extra_x_crop: Option[Dict[str, Tuple[List[Option[np.ndarray]], Color]]]
    draw_gt_bbox: bool
    draw_pred_bbox: bool
    search_region_border_width: int

    def __init__(self, historical: HistoricalMgr) -> None:
        self._historical = historical
        self._extra_bbox = NONE
        self._extra_x_crop = NONE
        self.draw_gt_bbox = True
        self.draw_pred_bbox = True
        self.search_region_border_width = 2

    def __len__(self) -> int:
        return len(self._historical)

    @property
    def extra_bbox(
        self,
    ) -> Option[Dict[str, Tuple[List[Option[BBox]], Color]]]:
        return self._extra_bbox

    @extra_bbox.setter
    def extra_bbox(
        self, value: Dict[str, Tuple[List[Option[BBox]], Color]]
    ) -> None:
        for name, extra in value.items():
            assert len(extra[0]) == len(
                self
            ), f"{name} has {len(extra[0])} elements, but should be {len(self)}"
        self._extra_bbox = Some(value)

    def _get_extra_bbox(self, i: int) -> Dict[str, Tuple[Option[BBox], Color]]:
        if self.extra_bbox.is_none():
            return {}
        extra_bbox = self.extra_bbox.unwrap()
        return {
            name: (extra[0][i], extra[1]) for name, extra in extra_bbox.items()
        }

    @property
    def extra_x_crop(
        self,
    ) -> Option[Dict[str, Tuple[List[Option[np.ndarray]], Color]]]:
        return self._extra_x_crop

    @extra_x_crop.setter
    def extra_x_crop(
        self, value: Dict[str, Tuple[List[Option[np.ndarray]], Color]]
    ) -> None:
        for name, extra in value.items():
            assert len(extra[0]) == len(
                self
            ), f"{name} has {len(extra[0])} elements, but should be {len(self)}"
        self._extra_x_crop = Some(value)

    def _get_extra_x_crop(
        self, i: int
    ) -> Dict[str, Tuple[Option[np.ndarray], Color]]:
        if self.extra_x_crop.is_none():
            return {}
        extra_x_crop = self.extra_x_crop.unwrap()
        return {
            name: (extra[0][i], extra[1])
            for name, extra in extra_x_crop.items()
        }

    def _get_hist(self, i: int) -> Historical:
        return self._historical[i]

    def get_frame(self, i: int) -> np.ndarray:
        info = self._get_hist(i).frame
        img = info.img.copy()
        cx, cy, w, h = list(map(int, info.aligned_bbox))
        cx = cx - (w - 1) // 2
        cy = cy - (h - 1) // 2
        gt_bbox = BBox(cx, cy, w, h)
        if self.draw_gt_bbox:
            cv2.rectangle(
                img,
                gt_bbox.lt,
                gt_bbox.rb,
                COLORS["green"].to_bgr(),
                5,
            )
        if self.draw_pred_bbox and info.bbox_pred is not None:
            cx, cy, w, h = list(map(int, info.bbox_pred))
            pred_bbox = BBox(cx, cy, w, h)
            cv2.rectangle(
                img,
                pred_bbox.lt,
                pred_bbox.rb,
                COLORS["red"].to_bgr(),
                5,
            )
        if self.extra_bbox.is_some():
            for name, (bbox, color) in self._get_extra_bbox(i).items():
                # TODO: name
                if bbox.is_some():
                    bbox = bbox.unwrap()
                    cv2.rectangle(
                        img,
                        bbox.lt,
                        bbox.rb,
                        color.to_bgr(),
                        5,
                    )
        return img

    def get_search_region(self, i: int) -> Option[np.ndarray]:
        info = self._get_hist(i).tracking
        if info.is_some():
            x_crop = info.unwrap().x_crop
            if x_crop is not None:
                x_crop = tensor2mat(x_crop)
                x_crop = cv2.resize(x_crop, (256, 256))
                return Some(x_crop)
            else:
                return NONE
        else:
            return NONE

    def get_output(
        self, i: int, search_region_size_ratio: float = 0.8
    ) -> np.ndarray:
        frame = self.get_frame(i)
        searches: Dict[str, Tuple[Option[np.ndarray], Color]] = {
            "this": (self.get_search_region(i), COLORS["red"]),
            **self._get_extra_x_crop(i),
        }
        search_canvases = []
        for name, (search, color) in searches.items():
            search_canvas = create_canvas(
                frame.shape[0],
                frame.shape[1],
                COLORS["black"].to_bgr(),
            )
            # search = self.get_search_region(i)
            messages = [name]
            if search.is_some():
                search = search.unwrap()
                width = int(
                    min(frame.shape[0], frame.shape[1])
                    * search_region_size_ratio
                )
                _rw = width - self.search_region_border_width * 2
                search = cv2.resize(search, (_rw, _rw))
                search = add_border(
                    search,
                    self.search_region_border_width,
                    color,
                )
                padding_x = (frame.shape[1] - width) // 2
                padding_y = (frame.shape[0] - width) // 2
                search_canvas[
                    padding_y : padding_y + width,
                    padding_x : padding_x + width,
                ] = search
            else:
                messages.append("lost")
            for i, msg in enumerate(messages):
                cv2.putText(
                    search_canvas,
                    msg,
                    (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=color.to_bgr(),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
            search_canvases.append(search_canvas)

        return merge_imgs(
            [frame, *search_canvases],
            fixed_width=1 + len(search_canvases)
            if frame.shape[0] > frame.shape[1]
            else 1,
        )

    def save_to_video(self, name: str):
        create_video(
            os.path.expandvars("$PYOTP_EXP/vis"),
            name,
            [self.get_frame(i) for i in range(len(self))],
            10,
        )

    def save_to_folder(
        self,
        folder: str,
        name: str,
        items: Set[Tuple[str, ValidVis]],
    ):
        base = os.path.join(folder, name)
        if not os.path.exists(base):
            os.makedirs(base)
        for i in range(len(self)):
            for item in items:
                suffix, vis = item
                if vis == ValidVis.Frame:
                    cv2.imwrite(
                        f"{base}/{i:05d}{suffix}.jpg",
                        self.get_frame(i),
                    )
                elif vis == ValidVis.SearchRegion:
                    search = self.get_search_region(i)
                    if search.is_some():
                        cv2.imwrite(
                            f"{base}/{i:05d}{suffix}.jpg",
                            search.unwrap(),
                        )
                elif vis == ValidVis.All:
                    cv2.imwrite(
                        f"{base}/{i:05d}{suffix}.jpg",
                        self.get_output(i),
                    )

    def save_pred_bboxes(self, folder, name: str):
        pred_bboxes = []
        for i, hist in enumerate(self._historical):
            pred = hist.frame.bbox_pred
            if pred is None:
                pred = 0
            pred_bboxes.append(pred)
        pred_bboxes = np.array(pred_bboxes, dtype=object)
        base = os.path.join(folder, name)
        if not os.path.exists(base):
            os.makedirs(base)
        np.save(f"{base}/pred_bboxes.npy", pred_bboxes)


def test_visualization(
    historical: HistoricalMgr,
) -> None:
    # visualization
    frames = []
    gt_size = None
    last_bbox: Optional[BBox] = None
    for i, his in enumerate(historical):
        info = his.frame
        info_tracking = his.tracking
        info_extra = his.extra
        if gt_size is None:
            gt_size = (info.img.shape[0], info.img.shape[1])
            w = max(gt_size)
            gt_size = (w, w)

        img_ctr = np.ones((*gt_size, 3), dtype=np.uint8) * 255
        img = info.img
        cx, cy, w, h = list(map(int, info.aligned_bbox))
        cx = cx - (w - 1) // 2
        cy = cy - (h - 1) // 2
        gt_bbox = BBox(cx, cy, w, h)
        cv2.rectangle(
            img,
            gt_bbox.lt,
            gt_bbox.rb,
            (120, 255, 80),
            5,
        )
        if info.bbox_pred is not None:
            cx, cy, w, h = list(map(int, info.bbox_pred))
            pred_bbox = BBox(cx, cy, w, h)
            cv2.rectangle(
                img,
                pred_bbox.lt,
                pred_bbox.rb,
                (158, 146, 252),
                5,
            )
        img_ctr[: img.shape[0], : img.shape[1], :3] = img
        img_ctr = cv2.resize(img_ctr, (512, 512))
        cv2.putText(
            img_ctr,
            f"frame: {i: >4}",
            (15, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(150, 150, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        if last_bbox is not None:
            arrow_imgs = []
            for pos in ["lt", "rt", "rb", "lb"]:
                distance, rot = calculate_points_inf(
                    last_bbox.__getattribute__(pos),
                    gt_bbox.__getattribute__(pos),
                )
                arrow_img = draw_arrow(128, 128, distance, rot, comment=pos)
                arrow_imgs.append(arrow_img)
            arrow_img = merge_imgs(arrow_imgs, fixed_width=2)
            arrow_tot = create_canvas(256, 256)
            arrow_img = merge_imgs([arrow_img, arrow_tot], fixed_width=2)
        else:
            arrow_img = create_canvas(256, 512)
        last_bbox = gt_bbox

        if info_tracking.is_some():
            info_tracking = info_tracking.unwrap()
            x_crop = info_tracking.x_crop
            x_crop_gt = info_tracking.x_crop_gt
        else:
            x_crop = None
            x_crop_gt = None
        if x_crop is not None:
            x_crop = tensor2mat(x_crop)
            x_crop = cv2.resize(x_crop, (256, 256))
        else:
            x_crop = create_canvas(256, 256, color=0)
        if x_crop_gt is not None:
            x_crop_gt = tensor2mat(x_crop_gt)
            x_crop_gt = cv2.resize(x_crop_gt, (256, 256))
        else:
            x_crop_gt = create_canvas(256, 256, color=0)

        crop = merge_imgs([x_crop_gt, x_crop], fixed_width=2)
        rhs_img = merge_imgs([crop, arrow_img], fixed_width=1)

        frame = merge_imgs([img_ctr, rhs_img], fixed_width=2)

        reg1 = info_extra.get_data("vis_reg1", None)
        reg2 = info_extra.get_data("vis_reg2", None)
        # if reg1 is not None and reg2 is not None:
        if True:
            reg1 = unwrap_or(
                reg1,
                create_canvas(512, 512),
            )
            reg2 = unwrap_or(
                reg2,
                create_canvas(512, 512),
            )
            reg1 = cv2.resize(reg1, (512, 512))
            reg2 = cv2.resize(reg2, (512, 512))
            reg = merge_imgs([reg1, reg2], fixed_width=2)
            frame = merge_imgs([frame, reg], fixed_width=1)

        cv2.imwrite(
            os.path.expandvars(f"$PYOTP_PATH/video/{str(i).zfill(4)}.jpg"),
            frame,
        )
        frames.append(frame)
