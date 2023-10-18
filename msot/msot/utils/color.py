from __future__ import annotations
from enum import IntEnum
from typing import Union


class ColorType(IntEnum):
    Fg = 0
    Bg = 1


class Colors(IntEnum):
    Black = 0
    Red = 1
    Green = 2
    Yellow = 3
    Blue = 4
    Magenta = 5
    Cyan = 6
    White = 7


class Styles(IntEnum):
    Bold = 1
    Dim = 2
    Underline = 4
    Blink = 5


_C = int


def _get_text(t: Union[_Color, str]) -> str:
    if isinstance(t, _Color):
        t = t.text
    return t


def _get_ascii_escape_code(param: str) -> str:
    return f"\x1b[{param}m"


def _get_color_param(c: _C, ct: ColorType) -> str:
    if isinstance(c, int) and c >= 0 and c < 16:
        if ct is ColorType.Fg:
            b = 30
        else:
            b = 40
        return str(b + c)
    elif isinstance(c, int) and c >= 16 and c < 256:
        if ct is ColorType.Fg:
            return f"38;5;{c}"
        else:
            return f"48;5;{c}"
    else:
        raise ValueError


def _get_style_param(s: Styles):
    if s is Styles.Bold:
        return "1"
    elif s is Styles.Dim:
        return "2"
    elif s is Styles.Underline:
        return "4"
    elif s is Styles.Blink:
        return "5"
    else:
        raise ValueError


class _Color:
    text: str

    def __init__(self, text: str) -> None:
        self.text = text

    @staticmethod
    def cvt_colorable(target: Colorable) -> _Color:
        if isinstance(target, _Color):
            target = target
        elif isinstance(target, str):
            target = _Color(target)
        else:
            raise ValueError
        return target

    @staticmethod
    def _reset(t: Union[_Color, str]) -> str:
        return _get_text(t) + "\x1b[0m"

    def _stylize(self, style: Styles) -> None:
        self.text = _get_ascii_escape_code(_get_style_param(style)) + self.text

    def _colorize(self, color: _C, color_type: ColorType) -> None:
        self.text = (
            _get_ascii_escape_code(_get_color_param(color, color_type))
            + self.text
        )

    def fg(self, color: _C) -> None:
        self._colorize(color, ColorType.Fg)

    def bg(self, color: _C) -> None:
        self._colorize(color, ColorType.Bg)

    def bold(self) -> None:
        self._stylize(Styles.Bold)

    def dim(self) -> None:
        self._stylize(Styles.Dim)

    def underline(self) -> None:
        self._stylize(Styles.Underline)

    def blink(self) -> None:
        self._stylize(Styles.Blink)

    def __str__(self) -> str:
        return self._reset(self.text)


Colorable = Union[_Color, str]


def fg(target: Colorable, color: _C) -> _Color:
    target = _Color.cvt_colorable(target)
    target.fg(color)
    return target


def bg(target: Colorable, color: _C) -> _Color:
    target = _Color.cvt_colorable(target)
    target.bg(color)
    return target


def bold(target: Colorable) -> _Color:
    target = _Color.cvt_colorable(target)
    target.bold()
    return target


def dim(target: Colorable) -> _Color:
    target = _Color.cvt_colorable(target)
    target.dim()
    return target


def underline(target: Colorable) -> _Color:
    target = _Color.cvt_colorable(target)
    target.underline()
    return target


def blink(target: Colorable) -> _Color:
    target = _Color.cvt_colorable(target)
    target.blink()
    return target
