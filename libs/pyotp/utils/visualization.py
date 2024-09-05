import cv2
import numpy as np
import numpy.typing as npt
from ..typing import BBox
from typing import Optional, TypeVar, Union

I = TypeVar("I", bound=np.ndarray)


class Color:
    _rgb: tuple[int, int, int]

    def __init__(self, rgb: tuple[int, int, int]) -> None:
        self._rgb = rgb

    def to_bgr(self) -> tuple[int, int, int]:
        return self._rgb[::-1]

    def to_rgb(self) -> tuple[int, int, int]:
        return self._rgb


COLORS = {
    "white": Color(rgb=(255, 255, 255)),
    "black": Color(rgb=(0, 0, 0)),
    "red": Color(rgb=(252, 146, 158)),
    "green": Color(rgb=(80, 255, 120)),
    "blue": Color(rgb=(94, 167, 223)),
    "magenta": Color(rgb=(165, 94, 223)),
    "yellow": Color(rgb=(223, 193, 94)),
}


def add_border(
    im: np.ndarray, width: int, color: Color, inner: bool = False
) -> npt.NDArray[np.uint8]:
    if inner:
        canvas_size = (im.shape[1], im.shape[0])
    else:
        canvas_size = (im.shape[1] + 2 * width, im.shape[0] + 2 * width)
    canvas = create_canvas(
        canvas_size[0],
        canvas_size[1],
        color=color,
    )
    if inner:
        canvas[width : im.shape[0] - width, width : im.shape[1] - width] = im[
            width : im.shape[0] - width, width : im.shape[1] - width
        ]
    else:
        canvas[width : width + im.shape[0], width : width + im.shape[1]] = im
    return canvas


def add_text(
    im: np.ndarray,
    text: str,
    color: Color = COLORS["white"],
    margin_left: int = -1,
    margin_top: int = -1,
    # org: tuple[int, int] = (0, 0),
    font_height: int = -1,
    font_face: int = cv2.FONT_HERSHEY_DUPLEX,
    font_scale: int = 1,
    thickness: int = 1,
    start_ln: int = 0,
    wrap_length: int = 20,
    from_bottom: bool = False,
) -> np.ndarray:
    ln = start_ln
    (fw, fh), _ = cv2.getTextSize(
        "",
        fontFace=font_face,
        fontScale=font_scale,
        thickness=thickness,
    )
    if font_height == -1:
        font_height = int(im.shape[0] * 0.05)
    # https://stackoverflow.com/a/73666899
    factor = (fh - 1) / font_scale
    font_scale = (font_height - thickness) / factor

    if margin_left == -1:
        margin_left = int(im.shape[1] * 0.05)
    if margin_top == -1:
        margin_top = int(im.shape[0] * 0.05)

    for t in text.split("\n"):
        for ss in range(0, len(t), wrap_length):
            name_wrap = t[ss : ss + wrap_length]
            org_x = margin_left
            if from_bottom:
                org_y = im.shape[0] - (margin_top + font_height) * (ln + 1)
            else:
                org_y = (margin_top + font_height) * (ln + 1)
            im = im.copy()
            cv2.putText(
                im,
                name_wrap,
                (org_x, org_y),
                fontFace=font_face,
                fontScale=font_scale,
                color=color.to_bgr(),
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
            ln += 1
    return im


def create_canvas(
    height: int,
    width: int,
    color: Union[int, tuple[int, int, int], Color] = 255,
) -> npt.NDArray[np.uint8]:
    """color is `BGR`(tuple) or brightness(int) or `Color`"""
    if isinstance(color, Color):
        color = color.to_bgr()
    return np.full((height, width, 3), color, dtype=np.uint8)
