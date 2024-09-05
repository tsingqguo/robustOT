from enum import Enum
import cv2
import datetime
import math
import numpy as np
import numpy.typing as npt
import pathlib
import shutil
import subprocess
import sys
import termtables
import time
import torch
import torch.nn.functional as F
import warnings
from .colorfmt import *
from os import path
from torch import Tensor
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NoReturn,
    Optional,
    Tuple,
    Type,
    TypeVar,
    TypedDict,
    Union,
)
from pyotp.utils.config import Config
from pyotp.utils.legacy_support import merge_from_file


PIT = TypeVar("PIT")


class PG:
    p_bar: str
    overflow: str = ".."
    refresh_thld: float = 0.1

    def clean_line(self):
        sys.stdout.write(f"\r{self.get_width()*' '}")

    # def get_average(self, x: List[float]) -> float:
    #     if len(x) == 0:
    #         return -1
    #     return sum(x) / len(x)

    # def get_delta(self, x: List[float]) -> List[float]:
    #     return [x[n] - x[n - 1] for n in range(1, len(x))]

    def get_width(self) -> int:
        return shutil.get_terminal_size().columns

    def format_time(self, sec: float):
        if False:
            pass
        elif sec > 120 and sec <= 1800:
            t = f"{sec/60:.2f}min"
        elif sec < 1e-2 and sec >= 1e-5:
            t = f"{sec*1e3:.2f}ms"
        elif sec < 1e-5 and sec >= 1e-8:
            t = f"{sec*1e6:.2f}us"
        elif sec < 1e-8:
            t = f"{sec*1e9:.2f}ns"
        else:
            t = f"{sec:.2f}s"
        return t

    def print(self, msg: str, inline: bool = False, raw: bool = False):
        self.clean_line()
        if inline:
            _msg = msg
            if len(_msg) > self.get_width():
                _msg = (
                    _msg[: self.get_width() - len(self.overflow)]
                    + self.overflow
                )
            sys.stdout.write(f"\r{_msg}")
        else:
            if raw:
                sys.stdout.write(f"\r{msg}")
            else:
                sys.stdout.write(f"\r{msg}\n")

    def pipe(self, msg, raw: bool = True):
        self.print(str(msg), inline=False, raw=raw)
        self.print(self.p_bar, inline=True, raw=False)

    def set_progress_bar(self, text: str):
        self.p_bar = text
        self.print(self.p_bar, inline=True)

    def __call__(
        self, items: Iterable[PIT], total: Optional[int] = None
    ) -> Iterable[PIT]:
        if total is not None:
            tot = total
        else:
            try:
                _items = list(items)
                tot = len(_items)
                items = _items
            except TypeError as err:
                # tot = sum(1 for _ in items)
                raise err
            # TODO:
            # tot = len(list(items))
            # tot = sum(1 for _ in items)

        t_o = time.time()
        t_p = t_o
        for i, item in enumerate(items):
            t_c = time.time()
            t_d = t_c - t_p
            if i == 0:
                self.set_progress_bar(f"current: {i+1}/{tot}")
            elif t_d > self.refresh_thld or i == tot - 1:
                if i > 0:
                    t_avg = (t_c - t_o) / i
                    eta = t_avg * (tot - i)
                    self.set_progress_bar(
                        f"current: {i+1}/{tot}; {self.format_time(t_avg)}/it ETA:{self.format_time(eta)}"
                    )
                else:
                    self.set_progress_bar(f"current: {i+1}/{tot}")
                t_p = t_c
            else:
                pass
            yield item


class TableArgs(TypedDict, total=False):
    style: str
    padding: Tuple[int, int]
    alignment: str


def add_gaussian_noise(im, mean: float, var: float):
    row, col, ch = im.shape
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch)) * 255
    gauss = gauss.reshape(row, col, ch)
    noisy = im + gauss
    return noisy


def auto_get_config_file(snapshot_path: str) -> Optional[str]:
    config_path = path.join(path.dirname(snapshot_path), "config.yaml")
    if path.isfile(config_path):
        return config_path
    else:
        return None


_Num = Union[int, float]


def calculate_points_inf(
    pts1: Tuple[_Num, _Num], pts2: Tuple[_Num, _Num]
) -> Tuple[float, float]:
    """
    pts1<older> -> pts2<newer>
    return distance and clockwise rotation(actually radian) between two points
    """
    x1, y1 = pts1
    x2, y2 = pts2
    y1 = -y1
    y2 = -y2

    rot = math.atan2((y2 - y1), (x2 - x1))
    distance = math.dist((x1, y1), (x2, y2))

    # if distance > 40:
    #     print(f'[INF] {pts1}->{pts2} distance:{distance}, rot:{rot}={np.rad2deg(rot)}')

    return distance, rot


def create_video(
    save_path: str,
    video_name: str,
    frames: list[cv2.Mat],
    fps: int,
    size: Optional[tuple[int, int]] = None,
):
    if size is None:
        f = frames[0]
        if f.ndim == 4:
            f = f[0]
        h, w, _ = f.shape
    else:
        h, w = size

    output = path.join(save_path, video_name + ".mp4")
    output = path.abspath(output)
    vw = cv2.VideoWriter(
        output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (h, w),
    )
    for f in frames:
        vw.write(f)
    vw.release()


def draw_arrow(
    w: int,
    h: int,
    severity: float,
    rot: float,
    draw_ring: bool = True,
    comment: Optional[str] = None,
    min_severity: float = 0,
    max_severity: float = 20,
    circle_width: int = 10,
):
    MAX_RADIUS = 0.9

    delta = max_severity - min_severity
    severity = (severity - min_severity) / delta
    severity = max(0, severity)
    severity = min(1, severity)

    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    start_x = w // 2
    start_y = h // 2
    end_x = start_x + int((severity * start_x * MAX_RADIUS) * np.cos(rot))
    end_y = start_y + int((severity * start_y * MAX_RADIUS) * np.sin(rot))
    # end_x = start_x + int(thickness * np.cos(rot))
    # end_y = start_y + int(thickness * np.sin(rot))
    if draw_ring:
        cv2.circle(
            img,
            (start_x, start_y),
            int(start_x * 0.5),
            (159, 165, 255),
            circle_width,
            lineType=cv2.LINE_AA,
        )
    cv2.arrowedLine(
        img,
        (start_x, start_y),
        (end_x, end_y),
        (0, 0, 0),
        3,
        line_type=cv2.LINE_AA,
        tipLength=0.2
        # 0,
        # 0.1,
    )
    if comment is not None:
        cv2.putText(
            img,
            comment,
            (10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return img


def exec_py_file(fp: str):
    exec(open(fp, mode="r").read())


def flatten_iterable(items: Iterable, valid_types: List[Type]) -> List:
    flt = []
    for i in items:
        if isinstance(i, Iterable) and type(i) in valid_types:
            flt = [*flt, *flatten_iterable(i, valid_types)]
        else:
            flt.append(i)
    return flt


def get_current_dt(format: str) -> str:
    return datetime.datetime.now().strftime(format)


def print_table(
    headers: List[str],
    data: Iterable[Iterable],
    fmt: Dict[str, Callable[[Any], str]] = {},
    arbitrary_fmt: List[Callable[[Any], Union[str, NoReturn]]] = [],
    print_args: TableArgs = {},
) -> None:
    D: List[List] = []
    for row in data:
        d_r = []
        for item in row:
            d_r.append(item)
        D.append(d_r)

    for a_cb in arbitrary_fmt:
        for r_i, row in enumerate(D):
            for i, item in enumerate(row):
                try:
                    D[r_i][i] = a_cb(item)
                except Exception:
                    pass

    class ItemNotFound(Exception):
        pass

    TT = TypeVar("TT")

    def find_header(
        array: List[TT], target: TT, start: int
    ) -> Union[int, NoReturn]:
        try:
            return array.index(target, start)
        except ValueError:
            raise ItemNotFound

    for h, cb in fmt.items():
        s = 0
        while True:
            try:
                c_i = find_header(headers, h, s)
                s = c_i + 1
                for r_i, row in enumerate(D):
                    D[r_i][c_i] = cb(row[c_i])
            except ItemNotFound:
                break
            except Exception as err:
                print(f"ERR: formatter failure:\n\t{str(err)}")

    default_print_args = {
        "style": termtables.styles.markdown,
        "padding": (0, 1),
        "alignment": "c",
    }

    termtables.print(D, header=headers, **{**default_print_args, **print_args})


_CVTT = Literal[255, 1]


def load_img2tensor(p: str, r_to: _CVTT):
    """return in [0,1]"""
    n: np.ndarray = cv2.imread(p)
    n = n.transpose(2, 0, 1)
    n_t = torch.from_numpy(n)
    n_t = n_t[None, :, :, :]
    if r_to == 255:
        n_t = n_t.clamp(0, 255)
    else:
        n_t = n_t / 255
        n_t = n_t.clamp(0, 1)
    return n_t


def mat2tensor(mat: np.ndarray, no_batch: bool = False) -> torch.Tensor:
    img = torch.from_numpy(mat).float()
    if img.ndim == 3:
        if no_batch:
            img = img.permute(2, 0, 1)
            return img
        else:
            img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2)
    return img


def tensor2img_ndarray(t: Tensor, from_t: _CVTT) -> np.ndarray:
    if from_t != 255:
        t = t * 255
    t = t.clamp(0, 255)
    if len(t.size()) == 4:
        t = t[0]
    n: np.ndarray = t.cpu().detach().numpy()
    n = n.transpose(1, 2, 0)
    return n


def tensor2mat(
    t: Tensor, allow_batch: bool = False, color_cvt: Optional[int] = None
) -> npt.NDArray[np.uint8]:
    if t.ndim == 4:
        if allow_batch:
            ...
        elif t.shape[0] == 1:
            t = t[0]
        else:
            raise ValueError(
                f"tensor2mat: tensor has more than 1 batch (shape={t.shape})"
            )
    elif t.ndim == 3:
        return tensor2mat(
            t.unsqueeze(0),
            color_cvt=color_cvt,
        )
    elif t.ndim == 5 and t.shape[0] == 1:
        return tensor2mat(
            t[0],
            allow_batch=allow_batch,
            color_cvt=color_cvt,
        )
    else:
        raise ValueError(
            f"tensor2mat: tensor has invalid shape (shape={t.shape})"
        )

    t = t.clamp(0, 255)
    n: np.ndarray = t.cpu().detach().numpy()
    if n.ndim == 3:
        n = n.transpose(1, 2, 0)
        if color_cvt is not None:
            n = cv2.cvtColor(n, color_cvt)
    elif n.ndim == 4:
        _n = []
        for sub in n:
            sub = sub.transpose(1, 2, 0)
            if color_cvt is not None:
                sub = cv2.cvtColor(sub, color_cvt)
            _n.append(sub)
        n = np.stack(_n, axis=0)
    n = n.astype(np.uint8)
    return n


def save_tensor2img(t: Tensor, p: str, from_t: _CVTT):
    n = tensor2img_ndarray(t, from_t)
    cv2.imwrite(p, n)


def merge_imgs(
    imgs: List[np.ndarray],
    ratio: float = 1,
    fixed_width: Optional[int] = None,
) -> npt.NDArray[np.uint8]:
    fill_c = 255
    L = len(imgs)
    if L == 1:
        return imgs[0]
    else:
        w = max([im.shape[0] for im in imgs])
        h = max([im.shape[1] for im in imgs])
        c = imgs[0].shape[2]  # TODO:
        if fixed_width is not None:
            w_cnt = fixed_width
        else:
            w_cnt = round(math.sqrt(L * ratio))
        h_cnt = math.ceil(L / w_cnt)
        ln = []
        for h_i in range(h_cnt):
            ln_imgs = imgs[h_i * w_cnt : (h_i + 1) * w_cnt]
            for w_i, ln_im in enumerate(ln_imgs):
                w_local = ln_im.shape[0]
                h_local = ln_im.shape[1]
                if w_local != w:
                    w_l = round((w - w_local) / 2)
                    w_r = w - w_l - w_local
                    ln_im = np.pad(
                        ln_im,
                        ((w_l, w_r), (0, 0), (0, 0)),
                        mode="constant",
                        constant_values=fill_c,
                    )
                if h_local != h:
                    h_t = round((h - h_local) / 2)
                    h_b = h - h_t - h_local
                    ln_im = np.pad(
                        ln_im,
                        ((0, 0), (h_t, h_b), (0, 0)),
                        mode="constant",
                        constant_values=fill_c,
                    )
                ln_imgs[w_i] = ln_im
            while len(ln_imgs) < w_cnt:
                ln_imgs.append(fill_c * np.ones((w, h, c), dtype=np.uint8))
            ln.append(np.hstack(ln_imgs))
        img = np.vstack(ln)
        return img


def save_imgs(
    imgs: Union[List[np.ndarray], np.ndarray],
    path_str: str,
    ratio: float = 1,
    fixed_width: Optional[int] = None,
):
    if isinstance(imgs, np.ndarray) and imgs.ndim == 4:
        imgs = [imgs[i] for i in range(imgs.shape[0])]
    if isinstance(imgs, np.ndarray) and imgs.ndim == 3:
        imgs = [imgs]
    elif isinstance(imgs, list):
        ...
    else:
        raise ValueError(f"invalid shape of imgs: {imgs.shape} (must be 4)")
    img = merge_imgs(imgs, ratio, fixed_width)
    cv2.imwrite(path_str, img)


_C = TypeVar("_C", bound=Config)
_ConfAssign = Callable[[_C], None]


def read_config(config_path: str, config: _C) -> _C:
    if not path.exists(config_path):
        raise FileNotFoundError(f"config file not found: {config_path}")
    ext = pathlib.Path(config_path).suffix[1:]
    if ext in ["yaml", "yml"]:
        merge_from_file(config, config_path)
    elif ext == "py":
        ns = {}
        exec(open(config_path).read(), {}, ns)
        base_config: Optional[Union[str, Iterable[str]]] = ns.get(
            "base_config"
        )
        assign: Optional[_ConfAssign[_C]] = ns.get("config_assign")
        if base_config is not None:
            if isinstance(base_config, str):
                read_config(base_config, config)
            else:
                for bc in base_config:
                    read_config(bc, config)
        if assign is None:
            raise ValueError(
                "can not found `fn config_assign` in given config file"
            )
        assign(config)
    else:
        raise ValueError(f"read_config: invalid config file extension: {ext}")
    return config


def silent_nanmean(a, **kwargs) -> Union[float, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, **kwargs)


from logging import Logger


def summary_tensor(
    t: Tensor,
    name: Optional[str] = None,
    size: Optional[tuple[int, int]] = None,
    logger: Optional[Logger] = None,
    indent: int = 0,
    scale_value: bool = False,
    reset_torch_print_options: Optional[dict] = None,
) -> torch.Tensor:
    def _indent(size: int) -> str:
        return " " * 4 * size

    h, w = t.shape[-2:]

    torch.set_printoptions(
        precision=1,
        linewidth=200,
        sci_mode=False,
        # profile='short',
    )

    if size is not None:
        org_h, org_w = h, w
        h, w = size
        if h == -1:
            h = int(w * org_h / org_w)
        elif w == -1:
            w = int(h * org_w / org_h)
        if h != org_h or w != org_w:
            t = F.interpolate(
                t,
                size=(h, w),
                mode="nearest",
                # mode="bilinear",
                # align_corners=True,
            )
            if scale_value:
                ratio = math.sqrt(h * w / (org_h * org_w))
                t = t * ratio

    if name is None:
        name = ""
    else:
        name = f' "{name}"'
    val = f"\n".join(
        ["", *[f"{_indent(indent + 1)}{s}" for s in str(t).split("\n")]]
    )
    msg = f"{_indent(indent)}\n".join(
        [
            f"summary tensor{name}:",
            f" - shape: {t.shape}",
            f" - dtype: {t.dtype}",
            f" -   val: {val}",
        ]
    )
    if logger:
        logger.info(msg)
    else:
        print(msg)
    if reset_torch_print_options is not None:
        torch.set_printoptions(**reset_torch_print_options)
    else:
        torch.set_printoptions(profile="default")
    return t


T = TypeVar("T")


def unwrap_or(x: Optional[T], default: T) -> T:
    if x is None:
        return default
    return x


def random_pick(
    arr: list[int],
    pick_length: int,
    gap_tolerance: int = 1,
    weighting: bool = True,
) -> Optional[list[int]]:
    candidates: list[list[int]] = [[]]

    cache = candidates[-1]
    for i in range(len(arr)):
        num = arr[i]
        if len(cache) == 0:
            cache.append(num)
        else:
            if num - cache[-1] <= gap_tolerance + 1:
                cache.append(num)
            else:
                candidates.append([num])
                cache = candidates[-1]

    candidates = list(filter(lambda x: len(x) >= pick_length, candidates))

    if len(candidates) == 1:
        pool = candidates[0]
    elif len(candidates) == 0:
        return None
    else:
        can_idx = np.arange(len(candidates))
        if weighting:
            w = [len(x) for x in candidates]
            w /= np.sum(w)
        else:
            w = None

        pick = np.random.choice(can_idx, p=w)
        pool = candidates[pick]

    start = np.random.randint(0, max(len(pool) - pick_length, 1))
    return pool[start : start + pick_length]


def pick_from_frame_id(arr: list[int], pick_size: int, start_id: int):
    start_idx = arr.index(start_id)
    return arr[start_idx : start_idx + pick_size]
