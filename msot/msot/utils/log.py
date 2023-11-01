import logging
from enum import Enum

from .color import Colors, fg


class LogLevel(Enum):
    # names are logging._nameToLevel.keys()
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warn"
    ERROR = "error"


class LogOnceFilter(logging.Filter):
    def __init__(self, name: str = "logonce") -> None:
        super().__init__(name)
        self._logged = set()

    def filter(self, record: logging.LogRecord) -> bool:
        if (
            record.args is not None
            and isinstance(record.args, dict)
            and record.args.get("log_once", None)
        ):
            key = f"{record.filename}.{record.lineno}"
            if key in self._logged:
                return False
            else:
                self._logged.add(key)
                return True
        else:
            return True


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
        return f"\r{asctime} {level}: {record.getMessage()}"


class MSOTLogHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
        self.setFormatter(MSOTLogFormatter())
        self.addFilter(LogOnceFilter())


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
