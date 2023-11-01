from __future__ import annotations
from enum import Enum, Flag
from typing import Generic, Type, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm

from msot.libs.potp_test.libs.dataset import Dataset
from msot.libs.potp_test.libs.video import Video
from msot.libs.pysot.utils.bbox import get_axis_aligned_bbox
from msot.models import TModel
from msot.trackers.base import (
    BaseTracker,
    TrackConfig,
    TrackerState,
    TrackResult,
)
from msot.utils.region import Box, Bbox, Polygon, Region
from msot.utils.region.helpers import calculate_overlap_ratio

from .action import (
    ParamsFrameFinish,
    ParamsFrameInit,
    ParamsFramePost,
    ParamsFrameSkip,
    ParamsFrameTrack,
    TestActionFinish,
    TestActionInit,
    TestActionPost,
    TestActionSkip,
    TestActionTrack,
    action_init,
    action_empty,
    action_finish,
    action_skip,
    action_track,
)
from .args import Args
from .config import TestConfig
from .info import SequenceInfo
from .utils.process import (
    InputProcess,
    ProcessSearch,
    ProcessTemplate,
    Processor as _Processor,
    ProceesorAttrs,
    ProceesorConfig,
)
from .utils.historical import Historical
from .utils.result import TestResult
from .utils.roles import TDRoles

A = TypeVar("A", bound=Args)
TA = TypeVar("TA", bound="TestAttributes")
TC = TypeVar("TC", bound=TrackConfig)
TS = TypeVar("TS", bound=TrackerState)
TR = TypeVar("TR", bound=TrackResult)
V = TypeVar("V", bound=Video)
VA = TypeVar("VA", bound=Flag)

# S = TypeVar("S", bound="Shared")
# class Shared(DS):
#     ...

Processor: TypeAlias = _Processor[ProceesorAttrs, ProceesorConfig]


class TrackActionType(Enum):
    INIT = "init"
    SKIP = "skip"
    TRACK = "track"


class TestAttributes(Generic[A]):
    _args: A

    start_at: int
    """start frame index"""

    restart_overlap_thld: float | None
    """restart if overlap ratio is le than this value (vot-st only)"""
    restart_skips: int | None
    """skip frames when restarting (vot-st only)"""

    historical: Historical
    input_process: InputProcess
    results_handler: TestResult | None
    sequence_info: SequenceInfo

    def __init__(
        self,
        args: A,
        video: V | None = None,
        dataset: Dataset[V, VA] | None = None,
        processors: list[Processor] | None = None,
    ):
        self._args = args

        self.start_at = 0
        self.historical = Historical(
            # TODO: configurable mem size
        )

        self.input_process = InputProcess()
        for p in args.processors or []:
            self.input_process.add(p)
        for p in processors or []:
            self.input_process.add(p)

        d_name = dataset.name if dataset is not None else "unknown_dset"
        if d_name in ["VOT2018", "VOT2019", "VOT2020"]:  # TODO:
            self.restart_overlap_thld = 0
            self.restart_skips = 5
        else:
            self.restart_overlap_thld = None
            self.restart_skips = None

        self.sequence_info = SequenceInfo(
            name=video.name if video is not None else None,
            attributes=None,  # TODO:
        )

        if args.variant_name is None:
            variant_name = "baseline"
        else:
            variant_name = args.variant_name

        if args.output_dir is None:
            self.results_handler = None
        else:
            self.results_handler = TestResult.create_file_based(
                args.output_dir,
                d_name,
                args.config.tracker.name,
                variant_name + args.variant_suffix,
                style=None,
            )


# class Test(Generic[TC, TS, TR]):
class Test(Generic[A, TA]):
    args: A
    test_attrs_cls: Type[TA]
    raw_tracker: BaseTracker

    action_init: TestActionInit
    action_skip: TestActionSkip
    action_track: TestActionTrack
    action_post: TestActionPost
    action_finish: TestActionFinish

    def __init__(
        self,
        args: A,
        test_attrs_cls: Type[TA] = TestAttributes,
    ):
        self.args = args
        self.test_attrs_cls = test_attrs_cls

        model = TModel.load_from_config(
            config=args.config.tracker.model,
            device=torch.device(args.config.tracker.backend.value),
        )
        model = model.eval()

        from msot.trackers.siamese.siamrpnpp import (
            TrackConfig as SiamRPNPPConfig,
            TrackerState as SiamRPNPPTrackerState,
            SiamRPNTracker,
        )

        if isinstance(args.config.tracker.track, SiamRPNPPConfig):
            self.raw_tracker = SiamRPNTracker(
                model,
                args.config.tracker.track,
                SiamRPNPPTrackerState,
                state=None,
            )
        else:
            raise NotImplementedError

        self.action_init = action_init
        self.action_skip = action_skip
        self.action_track = action_track
        self.action_post = action_empty
        self.action_finish = action_finish

    def exec(self):
        ...


class TestServer(Generic[A, TA]):
    test: Test[A, TA]
    _attrs: TA | None

    def __init__(self, test: Test[A, TA]):
        self.test = test
        self.reset()

    def init(
        self,
        dataset: Dataset | None = None,
        video: Video | None = None,
        test_attr_params: dict | None = None,
    ) -> bool:
        self._attrs = self.test.test_attrs_cls(
            **{
                **{
                    "args": self.test.args,
                    "video": video,
                    "dataset": dataset,
                },
                **(test_attr_params or {}),
            }
        )
        if self._attrs.results_handler is not None:
            return self._attrs.results_handler.try_init(
                video.name if video is not None else "unknown_seq",
                self.test.args.result_timeout_thld,
                self.test.args.force,
            )
        else:
            return True

    def next(self, frame: npt.NDArray[np.uint8], gt: Region | None):
        if self._attrs is None:
            raise RuntimeError("test server not initialized")

        self._attrs.historical.next()
        idx = len(self._attrs.historical) - 1

        cur_frame = self._attrs.historical.set_cur_frame(
            lambda F: F(frame, gt),
        )

        if idx == 0:
            self._attrs.sequence_info.size.update(frame.shape[:2])
        else:
            self._attrs.sequence_info.check_size(frame.shape[:2])

        if idx == self._attrs.start_at:
            tat = TrackActionType.INIT
        elif idx < self._attrs.start_at:
            tat = TrackActionType.SKIP
        else:
            tat = TrackActionType.TRACK

        if tat is TrackActionType.INIT:
            assert gt is not None
            # TODO: group-ip
            group_tracker = self.test.raw_tracker.fork()
            group_tracker.state_reset()  # IMPORTANT: for init
            self._attrs.input_process.prepare(
                {},  # FIXME: always empty attrs for init
                group_tracker,
                self._attrs.historical,
                ProcessTemplate(
                    self.test.raw_tracker.config.exemplar_size, gt
                ),
            )
            params = ParamsFrameInit(
                raw_tracker=group_tracker,
                historical=self._attrs.historical,
                sequence_info=self._attrs.sequence_info,
                input_process=self._attrs.input_process,
            )
            self.test.action_init(params)
            self.test.raw_tracker.absorb(group_tracker)  # IMPORTANT:

            if isinstance(gt, Box):
                pred = gt.to_bbox()
            elif isinstance(gt, Polygon):
                pred = gt.to_corner().to_bbox()
            else:
                raise NotImplementedError
            self._attrs.historical.set_cur_result(
                lambda R: R(self.test.raw_tracker.state, pred, None),
            )

        if tat is TrackActionType.SKIP:
            params = ParamsFrameSkip(
                raw_tracker=self.test.raw_tracker,
                historical=self._attrs.historical,
                sequence_info=self._attrs.sequence_info,
            )
            self.test.action_skip(params)
            self._attrs.historical.set_cur_result(
                lambda R: R(
                    self.test.raw_tracker.state,
                    Bbox(-1, -1, -1, -1),
                    None,
                    is_skip=True,
                ),
            )

        if tat is TrackActionType.TRACK:
            processor_attrs = (
                self._attrs.historical.last.unwrap().tracking.processor_attrs
            )
            # TODO: group-ip
            group_tracker = self.test.raw_tracker.fork()
            self._attrs.input_process.prepare(
                processor_attrs.get(default={}),
                group_tracker,
                self._attrs.historical,
                ProcessSearch(self.test.raw_tracker.config.instance_size),
            )
            params = ParamsFrameTrack(
                raw_tracker=group_tracker,
                historical=self._attrs.historical,
                sequence_info=self._attrs.sequence_info,
                input_process=self._attrs.input_process,
            )
            result = self.test.action_track(params)
            self.test.raw_tracker.absorb(group_tracker)  # IMPORTANT:
            self._attrs.historical.set_cur_result(
                lambda R: R(
                    self.test.raw_tracker.state,
                    result.output,
                    result.best_score,
                ),
            )

        if not cur_frame.gt.is_unbound():
            pred = self._attrs.historical.cur.result.unwrap().pred.get()
            overlap = calculate_overlap_ratio(
                pred, cur_frame.gt.get(TDRoles.TEST)
            )
            # TODO: collapsable variables
        else:
            overlap = None

        if self._attrs.historical.cur.tracking.is_some():
            process_costs = list(
                map(
                    lambda pa: (pa[0], pa[1].cost.get(default=None)),
                    self._attrs.historical.cur.tracking.unwrap()
                    .processor_attrs.get(default={})
                    .items(),
                )
            )
        else:
            process_costs = []

        self._attrs.historical.set_cur_analysis(
            lambda A: A(
                overlap=overlap,
                process_costs=process_costs,
            ),
        )

        if (
            tat is TrackActionType.TRACK
            and self._attrs.restart_overlap_thld is not None
            and overlap is not None
            and overlap <= self._attrs.restart_overlap_thld
        ):
            if self._attrs.restart_skips is None:
                raise RuntimeError(
                    "restart_skips is required when enable restart threshold"
                )
            self._attrs.start_at = idx + self._attrs.restart_skips

        params = ParamsFramePost(
            raw_tracker=self.test.raw_tracker,
            historical=self._attrs.historical,
            sequence_info=self._attrs.sequence_info,
        )
        self.test.action_post(params)

    def finish(self):
        if self._attrs is None:
            raise RuntimeError("test server not initialized")

        self._attrs.historical.finalize()

        params = ParamsFrameFinish(
            raw_tracker=self.test.raw_tracker,
            historical=self._attrs.historical,
            sequence_info=self._attrs.sequence_info,
            results_handler=self._attrs.results_handler,
        )
        self.test.action_finish(params)
        self.reset()

    def reset(self):
        self._attrs = None


def run_dataset(test: Test, dataset: Dataset):
    for idx, video in enumerate(dataset):
        run_video(test, video)


def run_video(
    test: Test[A, TA],
    video: Video,
    test_attr_params: dict | None = None,
):
    server = TestServer(test)
    success = server.init(
        video=video,
        test_attr_params=test_attr_params,
    )

    if not success:
        print(f"skip video {video.name}")
        return

    pbar = tqdm(
        video,
        desc=f"{video.name:>10}",
        ncols=60,
    )

    for _, (img, gt) in enumerate(pbar):
        ct = get_axis_aligned_bbox(np.array(gt))
        bbox = Bbox(
            ct.cx - (ct.w - 1) / 2,
            ct.cy - (ct.h - 1) / 2,
            ct.w,
            ct.h,
        )

        server.next(img, bbox)

    server.finish()
