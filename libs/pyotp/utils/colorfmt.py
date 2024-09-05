
from enum import Enum
from typing import Optional


class EscapeCCode(Enum):
    Black = "\x1b[30m"
    Red = "\x1b[31m"
    Green = "\x1b[32m"
    Yellow = "\x1b[33m"
    Blue = "\x1b[34m"
    Magenta = "\x1b[35m"
    Cyan = "\x1b[36m"
    White = "\x1b[37m"
    BrightBlack = "\x1b[90m"
    BrightRed = "\x1b[91m"
    BrightGreen = "\x1b[92m"
    BrightYellow = "\x1b[93m"
    BrightBlue = "\x1b[94m"
    BrightMagenta = "\x1b[95m"
    BrightCyan = "\x1b[96m"
    BrightWhite = "\x1b[97m"
    Reset = "\x1b[0m"
    NONE = -1


def to_color(text, color: Optional[EscapeCCode] = None) -> str:
    if color is None:
        return text
    elif color is EscapeCCode.NONE:
        return text
    else:
        return f'{color.value}{text}{EscapeCCode.Reset.value}'

def _to_color(text, code: str) -> str:
    return f"{code}{text}\x1b[0m"


def to_red(text) -> str:
    return _to_color(text, "\x1b[31m")


def to_green(text) -> str:
    return _to_color(text, "\x1b[32m")


def to_yellow(text) -> str:
    return _to_color(text, "\x1b[33m")


def to_blue(text) -> str:
    return _to_color(text, "\x1b[34m")
