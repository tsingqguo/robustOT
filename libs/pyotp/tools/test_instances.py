import re
from pyotp.env import ENV
from pyotp.modules.process import TrackerInputProcess, CP, PP, VP
from pyotp.modules.process.trk_pp import (
    LRRTrkPP,
    NoDefTrkPP,
)
from pyotp.tools.test import (
    ExecExtraAttr,
    Test,
    get_dataset,
    ValidDatasetNames,
)
from pyotp.typing import HistoricalMgr
from pyotp.utils import logging
from toolkit.datasets.dataset import Dataset
from typing import (
    Callable,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    TypedDict,
    Union,
)


_TI_Processors = List[Union[TrackerInputProcess, VP]]


class GroupedInfo(NamedTuple):
    processors: _TI_Processors
    sort_fn: Optional[Callable[[str], int]] = None


GroupedProcessors = Dict[str, _TI_Processors]
# GroupedProcessors = Dict[str, GroupedInfo]
ValidDatasets = Union[List[Dataset], List[ValidDatasetNames]]

logging.init_logger("global", level=logging.LT.Debug, ignore_exist=True)
LOG = logging.get_logger("global")


class TestInstance:
    test: Test
    _dyn_suffix: str
    user_suffix: str
    group_spawn_mode: bool
    extra_processors: Dict[str, _TI_Processors]

    def __init__(self, test: Test) -> None:
        self.test = test
        self.user_suffix = ""
        self.group_spawn_mode = False
        self.extra_processors = {}

    @property
    def suffix(self) -> str:
        return "_ti" + self._dyn_suffix + self.user_suffix

    def _processors_builder(self) -> _TI_Processors:
        raise NotImplementedError

    def _run(
        self,
        datasets: ValidDatasets,
        processors: _TI_Processors,
        test_attr_params: Dict,
        sequences: List[str] = [],
        groups: Optional[GroupedProcessors] = None,
    ):
        """dummy when group_spawn_mode is enabled"""
        if not self.group_spawn_mode:
            self.test.args.variant_suffix = self.suffix
            for di, dataset in enumerate(datasets):
                if isinstance(dataset, str):
                    dataset = get_dataset(dataset)

                self.test.remove_all_processors()
                for p in processors:
                    self.test.add_processor(p)

                for grp, p_list in self.extra_processors.items():
                    for p in p_list:
                        self.test.add_processor(p, group=grp)

                _extra: ExecExtraAttr = {
                    "datasets_progress": (di + 1, len(datasets))
                }
                self.test.exec(
                    dataset,
                    test_attr_params=test_attr_params,
                    sequences=sequences,
                    extra_attr={**_extra},
                )
        else:
            if groups is None:
                raise ValueError("groups is required in group_spawn_mode")
            suffix = self.suffix
            if suffix.startswith("_ti_"):
                suffix = suffix[4:]
            groups[suffix] = processors
            LOG.info(f"spawned group {suffix}")

    def run(
        self,
        datasets: ValidDatasets,
        test_attr_params: Dict = {},
        sequences: List[str] = [],
        groups: Optional[GroupedProcessors] = None,
    ):
        raise NotImplementedError

    def _run_groups(
        self,
        datasets: ValidDatasets,
        grouped_processors: GroupedProcessors,
        test_attr_params: Dict,
        sequences: List[str] = [],
        exec_extra_attrs: ExecExtraAttr = {},
        group_results_name: str = "grouped_BX",
    ):
        if not group_results_name.startswith("_ti_"):
            group_results_name = "_ti_" + group_results_name
        self.test.args.variant_suffix = group_results_name

        for di, dataset in enumerate(datasets):
            if isinstance(dataset, str):
                dataset = get_dataset(dataset)

            self.test.remove_all_processors()
            for grp, trkpp_list in grouped_processors.items():
                for p in trkpp_list:
                    # FIXME:
                    if isinstance(p, TrackerInputProcess):
                        self.test.add_processor(p, group=grp)
                    else:
                        self.test.add_processor(p)  # FIXME:

            for grp, p_list in self.extra_processors.items():
                for p in p_list:
                    self.test.add_processor(p, group=grp)

            _extra: ExecExtraAttr = {
                "datasets_progress": (di + 1, len(datasets))
            }
            self.test.exec(
                dataset,
                test_attr_params=test_attr_params,
                sequences=sequences,
                extra_attr={
                    **_extra,
                    **exec_extra_attrs,
                },
            )


class TI_NoDef(TestInstance):
    def _processors_builder(self) -> _TI_Processors:
        nodef = NoDefTrkPP()
        return [nodef]
        # return []

    def run(
        self,
        datasets: ValidDatasets,
        test_attr_params: Dict = {},
        sequences: List[str] = [],
        groups: Optional[GroupedProcessors] = None,
    ):
        self._dyn_suffix = "_nodef"

        self._run(
            datasets,
            self._processors_builder(),
            test_attr_params,
            sequences=sequences,
            groups=groups,
        )


# def _lrr_text_template(text: str) -> str:
#     template = "A photo of a {}."
#     pattern = re.compile("[^a-zA-Z]")
#     text = pattern.sub("", text)
#     return template.format(text)


class TI_LRR(TestInstance):
    args_saved: dict[str, str]
    args_FRAME_CNT: list[int]
    static_text_template: Optional[
        str | Callable[[str], str]
    ] = "A photo of a {}."
    # static_text_template: Optional[
    #     str | Callable[[str], str]
    # ] = staticmethod(_lrr_text_template)

    def _processors_builder(
        self,
        saved_fp: str,
        frame_cnt: int,
    ) -> _TI_Processors:
        trkpp_lrr = LRRTrkPP(
            saved_fp,
            frame_cnt,
            text_template=self.static_text_template,
        )
        return [trkpp_lrr]

    def run(
        self,
        datasets: ValidDatasets,
        test_attr_params: Dict = {},
        sequences: List[str] = [],
        groups: Optional[GroupedProcessors] = None,
    ):
        for name, saved in self.args_saved.items():
            for frame_cnt in self.args_FRAME_CNT:
                self._dyn_suffix = f"_trkpp_LRR_{name}_FN={frame_cnt}"
                self._run(
                    datasets,
                    self._processors_builder(
                        saved,
                        frame_cnt,
                    ),
                    test_attr_params,
                    sequences=sequences,
                    groups=groups,
                )
