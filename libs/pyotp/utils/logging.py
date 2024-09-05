from __future__ import annotations
import logging
import os
from .colorfmt import *
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set, Type, TypedDict, Union


class _LoggerException(Exception):
    ...


class LoggerNotExistError(_LoggerException):
    ...


class LoggerReInitError(_LoggerException):
    ...


class LoggerMultipleFileHandlerError(_LoggerException):
    ...


class _LoggerInfo(TypedDict):
    fh: List[logging.FileHandler]
    once_filter: LogOnceFilter
    rank_filter: RankFilter


_LOGGERS: Dict[str, _LoggerInfo] = {}


class BlockFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return False


class LogOnceFilter(logging.Filter):
    enable: bool
    logged_codelines: Set[str]

    def __init__(self, enable: bool, name: str = ""):
        super().__init__(name)
        self.enable = enable
        self.logged_codelines = set()

    def _filter_codeline(self, record: logging.LogRecord) -> bool:
        cl = f"{record.pathname}#{record.lineno}"
        if cl in self.logged_codelines:
            return False
        self.logged_codelines.add(cl)
        return True

    def filter(self, record: logging.LogRecord) -> bool:
        if record.__dict__.get("once", False):
            return self._filter_codeline(record)
        elif self.enable:
            return self._filter_codeline(record)
        else:
            return True


class RankFilter(logging.Filter):
    valid_rank: Set[int]

    def __init__(self, valid_rank: Set[int], name: str = ""):
        super().__init__(name)
        self.valid_rank = valid_rank

    def filter(self, _: logging.LogRecord) -> bool:
        rank = int(os.environ.get("RANK", -1))
        return rank in self.valid_rank


class LT(IntEnum):
    Debug = logging.DEBUG
    Info = logging.INFO
    Warn = logging.WARN
    Error = logging.ERROR
    Fatal = logging.FATAL


class M_Formatter(logging.Formatter):
    _formatters: Dict[int, logging.Formatter]
    no_color: bool = False

    @staticmethod
    def _build_fmt(levelno: int, no_color: bool = False) -> logging.Formatter:
        rank = int(os.environ.get("RANK", "-1"))
        level_name: str = "%(levelname)s"

        level_color: Optional[EscapeCCode]
        rank_color: Optional[EscapeCCode]
        name_color: Optional[EscapeCCode]
        time_color: Optional[EscapeCCode]
        file_color: Optional[EscapeCCode]
        msg_color: Optional[EscapeCCode]

        if not no_color:
            if levelno == LT.Debug.value:
                level_color = EscapeCCode.BrightBlue
            elif levelno == LT.Info.value:
                level_color = EscapeCCode.BrightGreen
            elif levelno == LT.Warn.value:
                level_color = EscapeCCode.BrightYellow
                level_name = "WARN"
            elif levelno == LT.Error.value:
                level_color = EscapeCCode.BrightRed
            elif levelno == LT.Fatal.value:
                level_color = EscapeCCode.BrightRed
                level_name = "FATAL"
            else:
                raise ValueError(f"Unknown levelno: {levelno}")
            rank_color = EscapeCCode.BrightRed
            name_color = EscapeCCode.BrightMagenta
            time_color = EscapeCCode.BrightBlack
            file_color = None
            msg_color = None

        else:
            level_color = None
            rank_color = None
            name_color = None
            time_color = None
            file_color = None
            msg_color = None

        rank_str = "" if rank < 0 else f"@R={to_color(str(rank), rank_color)} "
        format = (
            f"[{to_color(level_name, level_color)}] "
            # + '' if rank < 0 else f"R:{to_color(str(rank), rank_color)} "
            + rank_str
            + to_color("%(name)s ", name_color)
            + to_color("%(filename)s#%(lineno)d ", file_color)
            + to_color("%(asctime)s", time_color)
            + to_color(f"\n%(message)s", msg_color)
        )
        return logging.Formatter(format)

    def format(self, record):
        try:
            self.__getattribute__("_formatters")
        except AttributeError:
            self._formatters = {}

        formatter = self._formatters.get(record.levelno)
        if formatter is None:
            formatter = self._build_fmt(record.levelno, self.no_color)
            self._formatters[record.levelno] = formatter
        return formatter.format(record)


def init_logger(
    name: str,
    level: LT = LT.Info,
    formatter: Optional[Type[logging.Formatter]] = M_Formatter,
    ignore_exist=False,
) -> logging.Logger:
    try:
        _ = _get_logger_info(name)
        if not ignore_exist:
            raise LoggerReInitError(f"logger {name} already exists")
        else:
            return logging.getLogger(name)
    except LoggerNotExistError:
        logger = logging.getLogger(name)
        logger.setLevel(level.value)

        sh = logging.StreamHandler()
        sh.setLevel(level.value)
        if formatter is not None:
            sh.setFormatter(formatter())
        logger.addHandler(sh)

        once_filter = LogOnceFilter(False)
        logger.addFilter(once_filter)
        rank_filter = RankFilter({-1})
        logger.addFilter(rank_filter)

        _LOGGERS[name] = {
            "fh": [],
            "once_filter": once_filter,
            "rank_filter": rank_filter,
        }
        return logger


def _get_logger_info(name: str) -> _LoggerInfo:
    if name not in _LOGGERS:
        raise LoggerNotExistError(f"logger {name} does not found")
    else:
        return _LOGGERS[name]


def get_logger(name: str, rank: Optional[int] = None) -> logging.Logger:
    _get_logger_info(name)
    return logging.getLogger(name)


def set_valid_rank(name: str, rank: Set[int]) -> None:
    info = _get_logger_info(name)
    info["rank_filter"].valid_rank = rank


def set_log_once(name: str, log_once: bool) -> None:
    info = _get_logger_info(name)
    if log_once:
        info["once_filter"].enable = log_once


def add_file_handler(
    name: str,
    fp: str,
    append: bool = True,
    level: LT = LT.Info,
    formatter: Optional[Type[logging.Formatter]] = None,
    allow_multiple_fh: bool = False,
) -> None:
    info = _get_logger_info(name)
    if not allow_multiple_fh and len(info["fh"]) > 0:
        raise LoggerMultipleFileHandlerError(
            f"logger {name} already has a file handler"
        )

    logger = logging.getLogger(name)

    fh = logging.FileHandler(
        fp,
        mode="a" if append else "w",
        encoding="utf-8",
    )
    fh.setLevel(level.value)

    if formatter is None:
        _formatter = M_Formatter()
        _formatter.no_color = True
    else:
        _formatter = formatter()

    fh.setFormatter(_formatter)

    logger.addHandler(fh)
    info["fh"].append(fh)
