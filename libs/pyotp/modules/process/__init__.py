import cv2
import numpy as np
import os
import torch
from collections import OrderedDict
from typing import Any, Callable, Optional, Union


# from pysot.tracker.siamrpn_tracker_cfgmod import SiamRPNTracker
from pyotp.typing import HistoricalMgr
from pyotp.utils import tensor2mat, save_imgs
from pyotp.utils.visualization import BBox, Color, COLORS, add_text


class VP:
    name: str
    _cuda: bool = True

    def __init__(self, name: str) -> None:
        self.name = name

    # def cuda(self) -> None:
    #     self._cuda = True

    def reset(self) -> None:
        ...


class CP(VP):
    ...


class PP(VP):
    def forward(
        self, input: Union[torch.Tensor, np.ndarray], historical: HistoricalMgr
    ) -> torch.Tensor:
        raise NotImplementedError


class TrackerInputProcess:
    linked_tracker: Any  # SiamRPNTracker
    linked_historical_mgr: HistoricalMgr

    def __init__(self, name: str) -> None:
        self.name = name

    def reset(self) -> None:
        ...

    def forward(
        self, input: torch.Tensor, historical: HistoricalMgr
    ) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        return self.forward(input, self.linked_historical_mgr)


class ProcessVisualizerSaveOptions:
    highlight_best: bool
    highlight_best_border_color: Color
    title: bool
    title_fg: Color
    info: bool
    info_fg: Color
    bbox: bool
    bbox_border_color: Color

    def __init__(
        self,
        highlight_best: bool = True,
        highlight_best_border_color: Color = COLORS["blue"],
        title: bool = True,
        title_fg: Color = COLORS["white"],
        info: bool = True,
        info_fg: Color = COLORS["white"],
        bbox: bool = True,
        bbox_border_color: Color = COLORS["green"],
    ) -> None:
        self.highlight_best = highlight_best
        self.highlight_best = False
        self.highlight_best_border_color = highlight_best_border_color
        self.title = title
        # self.title = False
        self.title_fg = title_fg
        self.info = info
        # self.info = False
        self.info_fg = info_fg
        self.bbox = bbox
        self.bbox_border_color = bbox_border_color


from toolkit.datasets.video import Video


class ProcessVisualizer:
    visualize: bool
    save_options: ProcessVisualizerSaveOptions
    vis_saved: dict[str, np.ndarray]
    info_saved: dict[str, list[str]]
    bbox_saved: dict[str, list[tuple[BBox, Optional[Color]]]]
    #
    save_dir: str
    fname: str
    #
    suffix: Optional[str]
    """useful in group mode -> using grp_name as suffix"""
    batch_mode: bool
    _clean_exist: bool
    _last_add: Optional[str]
    _idx: int  # do not use this
    _video: Video

    def __init__(
        self,
        visualize: bool,
        clean: Optional[Union[np.ndarray, torch.Tensor]] = None,
        save_options: Optional[ProcessVisualizerSaveOptions] = None,
    ) -> None:
        self.visualize = visualize
        self.vis_saved = {}
        self.info_saved = {}
        self.bbox_saved = {}
        self.suffix = None
        self.batch_mode = False
        self._clean_exist = False
        self._last_add = None
        if clean is not None:
            self.add(clean, "clean")
        if save_options is None:
            save_options = ProcessVisualizerSaveOptions()
        self.save_options = save_options

    def add(self, img: Union[np.ndarray, torch.Tensor], name: str) -> None:
        if self.visualize:
            if name == "clean":
                if not self._clean_exist:
                    self._last_add = name
                    self._clean_exist = True
                else:
                    return
            elif self.suffix is not None:
                self._last_add = name
                name = "_".join([name, self.suffix])

            if isinstance(img, torch.Tensor):
                v = tensor2mat(img)
            else:
                v = img
            self.vis_saved[name] = v.astype(np.uint8).copy()

    def add_bbox_to_last(
        self,
        bbox: Union[list[float], list[int], BBox],
        color: Optional[Union[Color, tuple]] = None,
    ) -> None:
        if self.visualize and self._last_add is not None:
            name = self._last_add
            if self.suffix is not None and name != "clean":
                name = "_".join([name, self.suffix])
            if name not in self.vis_saved:
                raise ValueError(f"{name} not in vis_saved")

            if isinstance(color, tuple):
                color = Color(color)

            if isinstance(bbox, BBox):
                self.bbox_saved.setdefault(name, []).append((bbox, color))
            else:
                self.bbox_saved.setdefault(name, []).append(
                    (BBox(*[int(x) for x in bbox]), color)
                )

    def add_info(self, info: str, name: Optional[str]) -> None:
        """assign info to all processors' name ending with suffix if name is `None`"""
        if self.visualize:
            if name is None:
                if self.suffix is not None:
                    for n in self.vis_saved.keys():
                        if n.endswith(self.suffix):
                            self.add_info(info, n[: -len(self.suffix) - 1])
                else:
                    raise ValueError(
                        "can not decide processor names without suffix(group name)"
                    )
            else:
                if self.suffix is not None:
                    name = "_".join([name, self.suffix])
                if name not in self.info_saved:
                    self.info_saved[name] = []
                self.info_saved[name].append(info)

    def save(
        self,
        sort_fn: Optional[Callable[[str], int]] = None,
        force: bool = False,
    ) -> None:
        if self.visualize:
            if not self.batch_mode or force:
                os.makedirs(self.save_dir, exist_ok=True)
                vis_with_info = {}
                if sort_fn is not None:
                    names = sorted(
                        [n for n in self.vis_saved.keys() if n != "clean"],
                        key=sort_fn,
                    )
                    # names = ["clean", *names]
                    names = [*names, "clean"]
                else:
                    names = self.vis_saved.keys()

                font_height = 12
                wrap_length = 20
                for name in names:
                    im = self.vis_saved[name].copy()
                    if self.save_options.title:
                        im = add_text(
                            im,
                            name,
                            color=self.save_options.title_fg,
                            font_height=font_height,
                            wrap_length=wrap_length,
                        )
                    if (
                        self.save_options.info
                        and self.info_saved.get(name) is not None
                    ):
                        text = "\n".join(self.info_saved[name][::-1])
                        im = add_text(
                            im,
                            text,
                            color=self.save_options.info_fg,
                            font_height=font_height,
                            wrap_length=wrap_length,
                            from_bottom=True,
                        )
                    if self.save_options.bbox and name in self.bbox_saved:
                        bboxes = self.bbox_saved[name]
                        for bbox, color in bboxes:
                            if color is None:
                                color = self.save_options.bbox_border_color
                            cv2.rectangle(
                                im,
                                bbox.lt,
                                bbox.rb,
                                color.to_bgr(),
                                2,
                            )
                    vis_with_info[name] = im
                if len(vis_with_info.keys()) > 0:
                    save_imgs(
                        [im for im in vis_with_info.values()],
                        os.path.join(self.save_dir, self.fname),
                    )
