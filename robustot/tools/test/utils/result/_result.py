import json
import os
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Generic, NamedTuple, Type, TypedDict, TypeVar

T = TypeVar("T", bound=dict[str, str])


class RuningStatus(Enum):
    NONE = 0
    STARTED = auto()
    EXPIRED = auto()
    DONE = auto()


class ResultMedium(Enum):
    NONE = 0
    FILE = auto()


class ResultStyle(Enum):
    ARBITRARY = 0
    VOT_ST = auto()
    VOT_LT = auto()


class TestResult:
    dataset_name: str
    tracker_name: str
    test_name: str
    # sequence_name: str
    medium: ResultMedium
    style: ResultStyle


@dataclass
class _TRFileMark:
    create_at: float
    host: str
    pid: int

    def dump(self) -> str:
        return json.dumps(self.__dict__)


class _TRFileMarkNotMatch(Exception):
    ...


class _TRFile(TestResult, Generic[T]):
    _result_root: str
    _templates: T
    template_builder: Callable[[T, str, ResultStyle], T]

    def __init__(
        self,
        result_root: str,
        templates: T,
        template_builder: Callable[[T, str, ResultStyle], T],
    ) -> None:
        self._result_root = result_root
        self._templates = templates
        self.template_builder = template_builder

    @property
    def test_result_root(self) -> str:
        return os.path.join(
            self._result_root,
            self.dataset_name,
            self.tracker_name,
            self.test_name,
        )

    def get_result_path(self, sequence_name: str) -> str:
        if self.style is ResultStyle.VOT_ST:
            return os.path.join(
                self.test_result_root,
                sequence_name,
            )
        elif self.style is ResultStyle.VOT_LT:
            return os.path.join(
                self.test_result_root,
                sequence_name,
                "longterm",
            )
        else:
            return self.test_result_root

    @property
    def result_file_template(self) -> str:
        if (
            self.style is ResultStyle.VOT_ST
            or self.style is ResultStyle.VOT_LT
        ):
            return "{}_001.txt"
        else:
            return "{}.txt"

    def create_mark(self) -> _TRFileMark:
        return _TRFileMark(
            time.time(),
            os.uname()[1],
            os.getpid(),
        )

    def load_mark(self, mark: str) -> _TRFileMark:
        m = json.loads(mark)
        return _TRFileMark(
            m["create_at"],
            m["host"],
            m["pid"],
        )

    def detect_exist(
        self, sequence_name: str, timeout_thld: int
    ) -> RuningStatus:
        if not os.path.exists(self.test_result_root):
            return RuningStatus.NONE

        result_seq_paths = self.template_builder(
            self._templates, sequence_name, self.style
        )
        result_seq_path = result_seq_paths["main"]  # FIXME:

        if not os.path.exists(result_seq_path):
            return RuningStatus.NONE

        result_fp = os.path.join(
            result_seq_path,
            self.result_file_template.format(sequence_name),
        )

        if not os.path.exists(result_fp):
            return RuningStatus.NONE

        with open(result_fp, "r") as f:
            try:
                mark = self.load_mark(f.read(500))  # TODO: /dev/null
            except json.JSONDecodeError:
                return RuningStatus.DONE

            if time.time() - mark.create_at > timeout_thld:
                return RuningStatus.EXPIRED
            else:
                return RuningStatus.STARTED

    def place_mark(self, sequence_name: str) -> None:
        result_seq_paths = self.template_builder(
            self._templates, sequence_name, self.style
        )
        result_seq_path = result_seq_paths["main"]  # FIXME:
        result_fp = os.path.join(
            result_seq_path,
            self.result_file_template.format(sequence_name),
        )
        mark = self.create_mark()
        if not os.path.exists(result_seq_path):
            os.makedirs(result_seq_path)
        with open(result_fp, "w") as f:
            f.write(mark.dump())
        time.sleep(1)
        with open(result_fp, "r") as f:
            if self.load_mark(f.read(500)) != mark:
                raise _TRFileMarkNotMatch

    def try_init(
        self,
        sequence_name: str,
        timeout_thld: int,
        overwrite_done: bool = False,
    ) -> bool:
        status = self.detect_exist(sequence_name, timeout_thld)
        if status is RuningStatus.STARTED:
            return False

        if status is RuningStatus.DONE and not overwrite_done:
            return False

        # for:
        # - NONE
        # - EXPIRED
        # - DONE with overwrite_done
        try:
            self.place_mark(sequence_name)
            return True
        except _TRFileMarkNotMatch:
            return False
