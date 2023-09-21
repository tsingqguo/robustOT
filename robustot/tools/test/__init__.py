from __future__ import annotations
import os
import time
from dataclasses import dataclass, field
from typing import Generic, Type, TypeVar

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from msot.data.datasets import get_dataset
from msot.libs.potp_test.libs.dataset import Dataset
from msot.libs.potp_test.libs.video import Video
from msot.libs.pysot.utils.bbox import get_axis_aligned_bbox
from msot.models import TModel, ModelConfig, TModelResult
from msot.trackers.base import (
    BaseTracker,
    TrackConfig,
    TrackerState,
    TrackResult,
)
from msot.utils.boxes import Bbox
from msot.utils.dataship import DataCTR as DC, DataShip as DS
from msot.utils.option import NONE, Option, Some
from msot.utils.timer import Timer

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
    action_track,
)
from .args import Args
from .config import TestConfig
from .info import SequenceInfo
from .utils.process import InputProcess, ProcessSearch, ProcessTemplate
from .utils.historical import Historical
from .utils.result import TestResult
from .utils.roles import TDRoles

A = TypeVar("A", bound=Args)
TA = TypeVar("TA", bound="TestAttributes")
TC = TypeVar("TC", bound=TrackConfig)
TS = TypeVar("TS", bound=TrackerState)
TR = TypeVar("TR", bound=TrackResult)
V = TypeVar("V", bound=Video)

# S = TypeVar("S", bound="Shared")
# class Shared(DS):
#     ...


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
    result_utils: TestResult
    sequence_info: SequenceInfo

    def __init__(
        self,
        args: A,
        video: V | None = None,
        dataset: Dataset[V] | None = None,
    ):
        self._args = args

        self.start_at = 0
        self.historical = Historical(
            # TODO: configurable mem size
        )

        self.input_process = InputProcess()
        # TODO: add processors
        from msot.tools.test.utils.process.builtin import (
            DefaultCrop,
            DefaultCropConfig,
        )
        from robbox.attackers.csa import (
            CSAConfig,
            CSAProcessor,
            AttackOn,
            AttackType,
        )

        csa_config = CSAConfig(
            AttackOn.TEMPLATE | AttackOn.SEARCH,
            AttackType.COOLING_SHRINKING,
            "../tracker_experiments/CSA/checkpoints",
        )
        self.input_process.add(DefaultCrop(None, DefaultCropConfig()))
        self.input_process.add(CSAProcessor("csa_atk", csa_config))

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

        self.result_utils = TestResult.create_file_based(
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

        model = TModel(args.config.tracker.model)
        model = TModel.load_pretrained(model)  # FIXME:

        if args.config.tracker.cuda:
            model = model.cuda()
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
        self.action_skip = action_empty
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
        return self._attrs.result_utils.try_init(
            video.name if video is not None else "unknown_seq",
            self.test.args.result_timeout_thld,
            self.test.args.force,
        )

    def next(self, frame: npt.NDArray[np.uint8], gt_box):
        if self._attrs is None:
            raise RuntimeError("test server not initialized")

        self._attrs.historical.next()
        idx = len(self._attrs.historical) - 1

        ct = get_axis_aligned_bbox(np.array(gt_box))
        bbox = Bbox(
            ct.cx - (ct.w - 1) / 2,
            ct.cy - (ct.h - 1) / 2,
            ct.w,
            ct.h,
        )

        _frame = self._attrs.historical.set_cur_frame(
            lambda F: F(frame, bbox, ct),
        )

        if idx == 0:
            self._attrs.sequence_info.size.update(frame.shape[:2])
        else:
            self._attrs.sequence_info.check_size(frame.shape[:2])

        if idx == self._attrs.start_at:
            # TODO: group-ip
            group_tracker = self.test.raw_tracker.fork()
            group_tracker.state_reset() # IMPORTANT: for init
            self._attrs.input_process.prepare(
                {},  # FIXME: always empty attrs for init
                group_tracker,
                self._attrs.historical,
                ProcessTemplate(
                    self.test.raw_tracker.config.exemplar_size, bbox
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
            self._attrs.historical.set_cur_result(
                lambda R: R(self.test.raw_tracker.state, bbox, None),
            )
        elif idx < self._attrs.start_at:
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
        else:
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

        overlap = (
            self._attrs.historical.cur.result.unwrap().pred.val.overlap_ratio(
                self._attrs.historical.cur.frame.unwrap().bbox.get(
                    TDRoles.TEST
                )
            )
        )  # TODO: ignore skipped / init;
        # TODO: collapsable variables
        self._attrs.historical.set_cur_analysis(
            lambda A: A(
                overlap=overlap,
            ),
        )

        if (
            self._attrs.restart_overlap_thld is not None
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
            result_utils=self._attrs.result_utils,
        )
        self.test.action_finish(params)

    def reset(self):
        self._attrs = None


def run_dataset(test: Test, dataset: Dataset):
    for idx, video in enumerate(dataset):
        print(f"[{idx+1}/{len(dataset)}] running", video.name)
        run_video(test, video, dataset=dataset)


def run_video(
    test: Test[A, TA],
    video: Video,
    dataset: Dataset | None = None,
    test_attr_params: dict | None = None,
):
    server = TestServer(test)
    success = server.init(
        dataset=dataset,
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

    for _, (img, gt_bbox) in enumerate(pbar):
        server.next(img, gt_bbox)

    server.finish()
