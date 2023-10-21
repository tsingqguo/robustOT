import logging
from enum import Enum

from .color import Colors, fg


class LogLevel(Enum):
    # names are logging._nameToLevel.keys()
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warn"
    ERROR = "error"


class MSOTLogFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": Colors.BrightBlue,
        "INFO": Colors.BrightGreen,
        "WARNING": Colors.BrightYellow,
        "ERROR": Colors.BrightRed,
        "CRITICAL": Colors.BrightRed,
    }

    def __init__(
        self,
        datefmt: str | None = None,
    ) -> None:
        super().__init__("%(asctime)s", datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        _ = super().format(record)

        level = fg(
            record.levelname, self.COLORS.get(record.levelname, Colors.White)
        ).bold()
        asctime = fg(record.asctime, Colors.White)
        return f"{asctime} {level}: {record.getMessage()}"


class MSOTLogHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
        self.setFormatter(MSOTLogFormatter())


def setup(log_level: LogLevel) -> logging.Logger:
    logger = logging.getLogger("msot")
    logger.setLevel(log_level.value.upper())

    for handler in logger.handlers:
        logger.removeHandler(handler)

    logger.addHandler(MSOTLogHandler())

    return logger


def set_level(log_level: LogLevel) -> None:
    logging.getLogger("msot").setLevel(log_level.value.upper())


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name or "msot")
