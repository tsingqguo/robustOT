from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

from msot.tools.eval.benchmarks.statics import ReservedResults

from ..historical import Analysis, Result
from ..roles import TDRoles


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
    medium: ResultMedium
    style: ResultStyle

    save_results: Callable[[_TRFile, str, list[Result], list[Analysis]], None]

    def __init__(
        self,
        dataset_name: str,
        tracker_name: str,
        test_name: str,
        medium: ResultMedium,
        style: ResultStyle | None,
    ) -> None:
        self.dataset_name = dataset_name
        self.tracker_name = tracker_name
        self.test_name = test_name
        self.medium = medium
        if style is not None:
            self.style = style
        else:
            # FIXME:
            if dataset_name in ["VOT2019", "VOT2018", "GOT-10k"]:
                self.style = ResultStyle.VOT_ST
            elif dataset_name in ["VOT2018-LT"]:
                self.style = ResultStyle.VOT_LT
            else:
                self.style = ResultStyle.ARBITRARY

        self.save_results = lambda *args: None

    @staticmethod
    def create_file_based(
        root: str,
        dataset_name: str,
        tracker_name: str,
        test_name: str,
        style: ResultStyle | None = None,
    ) -> _TRFile:
        return _TRFile(
            root,
            dataset_name,
            tracker_name,
            test_name,
            style,
        )

    def try_init(
        self,
        sequence_name: str,
        timeout_thld: int,
        overwrite_done: bool = False,
    ) -> bool:
        raise NotImplementedError

    def save_results_default(
        self,
        sequence_name: str,
        results: list[Result],
        analysis: list[Analysis],
    ):
        raise NotImplementedError


@dataclass
class _TRFileMark:
    create_at: float
    host: str
    pid: int

    def dump(self) -> str:
        return json.dumps(self.__dict__)


class _TRFileMarkNotMatch(Exception):
    ...


class _TRFile(TestResult):
    _result_root: str

    def __init__(
        self,
        result_root: str,
        dataset_name: str,
        tracker_name: str,
        test_name: str,
        style: ResultStyle | None = None,
    ) -> None:
        super().__init__(
            dataset_name,
            tracker_name,
            test_name,
            ResultMedium.FILE,
            style,
        )

        self._result_root = result_root

    @property
    def test_result_root(self) -> str:
        return os.path.join(
            self._result_root,
            self.dataset_name,
            self.tracker_name,
            self.test_name,
        )

    def get_result_path(self, sequence_name: str) -> str:
        """get result path for given sequence"""
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

        result_seq_path = self.get_result_path(sequence_name)

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
        result_seq_path = self.get_result_path(sequence_name)
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
        except FileExistsError:
            return False
        except _TRFileMarkNotMatch:
            return False
        except json.JSONDecodeError:
            return False

    def save_results_default(
        self,
        sequence_name: str,
        results: list[Result],
        analysis: list[Analysis],
    ):
        result_dir = self.get_result_path(sequence_name)

        pred = []
        for r, a in zip(results, analysis):
            bbox = r.pred.val
            if self.style is ResultStyle.VOT_ST:
                if r.best_score.val is not None:
                    overlap = a.overlap.get(TDRoles.ANALYSIS)
                    if overlap is None:
                        raise ValueError("overlap is None")
                    if overlap > 0:
                        pred.append(
                            ",".join(
                                map(lambda a: f"{a:.4f}", bbox.unpack())
                            )
                        )
                    else:
                        pred.append(str(ReservedResults.FAILURE.value))
                elif r.is_skip.val:
                    pred.append(str(ReservedResults.SKIP.value))
                else:
                    pred.append(str(ReservedResults.INIT.value))
            else:
                pred.append(",".join(map(str, map(int, bbox.unpack()))))

        with open(
            os.path.join(
                result_dir,
                self.result_file_template.format(sequence_name),
            ),
            "w",
        ) as f:
            f.write("\n".join(pred) + "\n")

        # extra with GOT-10k and VOT2018-LT
        # TODO:
