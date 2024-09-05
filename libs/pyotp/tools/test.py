from __future__ import annotations
import cv2
import numpy as np
import numpy.typing as npt
import os
import sys
import time
import toolkit.datasets as Datasets
import torch
from enum import Enum
from pyotp import ENV
from pyotp.config.pysot import CFG, PYSOT_Config
from pyotp.modules.process import (
    CP,
    PP,
    VP,
    ProcessVisualizer,
    ProcessVisualizerSaveOptions,
    TrackerInputProcess,
)
from pyotp.typing import (
    DataCTR,
    HistoricalFrame,
    HistoricalMgr,
    HistoricalTracking,
)
from pyotp.utils import (
    auto_get_config_file,
    get_current_dt,
    save_imgs,
    tensor2mat,
    unwrap_or,
)
from pyotp.utils.analysis import iou_overlap
from pyotp.utils.legacy_support import merge_from_file
from pyotp.utils.option import Some, NONE, Option
from pyotp.utils.visualization import Color, COLORS, add_border
from pysot.models.model_builder_cfgmod import ModelBuilder
from pysot.tracker.siamrpn_tracker_cfgmod import SiamRPNTracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from tqdm import tqdm
from toolkit.datasets import (
    DatasetFactory,
    ValidDatasetNames,
    ValidDatasetNames_LaSOT,
    ValidDatasetNames_NFS,
    ValidDatasetNames_OTB,
    ValidDatasetNames_VOT,
    ValidDatasetNames_VOTLT,
    ValidDatasetNames_UAV,
)
from toolkit.datasets.dataset import Dataset
from toolkit.datasets.video import SeqImg, Video
from toolkit.utils.region import vot_overlap, vot_float2str
from typed_cap import Cap
from typing import (
    overload,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    TypedDict,
    Union,
)


class TrackingOutputs(TypedDict):
    bbox: Any
    polygon: Optional[Any]
    #
    best_score: Optional[Any]


class TestCliArgs:
    # @alias=c
    config: Optional[str]
    """path to config"""

    # @alias=d
    dataset: Optional[list[ValidDatasetNames]]
    """name of dataset"""

    # @alias=f
    force: bool = False
    """ignore the existing results"""

    # @alias=g
    gpu_id: int = 0
    """gpu id"""

    # @alias=o
    output_dir: str = "results"
    """directory for saving testing results"""

    save_frame_info: bool = False
    """saving frame info includes frame_img & bboxes"""

    # @alias=s
    snapshot: str
    """path to the snapshot"""

    # @alias=n
    snapshot_name: str

    # @alias=v
    variant_suffix: Optional[str]
    """suffix for variant"""

    # @alias=V
    visualize: bool = False
    """saving visualize as video"""


A = TypeVar("A", bound="TestArgs")
C = TypeVar("C", bound=PYSOT_Config)
T = TypeVar("T", bound=SiamRPNTracker)  # TODO:
TA = TypeVar("TA", bound="TestAttributes")
V = TypeVar("V", bound=Video)


class TestArgs(Generic[T, C]):
    config: C
    dataset: Optional[list[ValidDatasetNames]]
    force: bool
    model_name: str
    output_dir: str
    process_visualizer_save_options: Optional[ProcessVisualizerSaveOptions]
    save_frame_info: bool
    snapshot_path: str
    snapshot_name: str
    video: List[str]
    visualize: bool
    variant_name: str = "baseline"
    variant_suffix: str = ""
    tracker_cb: Optional[Callable[[T], None]] = None


class TestCBParamVideo(Generic[A, T, V, TA]):
    args: A
    raw_tracker: T
    attributes: TA
    v_idx: int
    video: V
    dataset: Dataset[V]
    session_data: DataCTR
    messenger: Callable[[str], None]

    def __init__(
        self,
        args: A,
        raw_tracker: T,
        attributes: TA,
        v_idx: int,
        video: V,
        dataset: Dataset[V],
        session_data: DataCTR,
        messenger: Callable[[str], None],
    ) -> None:
        self.args = args
        self.raw_tracker = raw_tracker
        self.attributes = attributes
        self.v_idx = v_idx
        self.video = video
        self.dataset = dataset
        self.session_data = session_data
        self.messenger = messenger


class TestCBParamFrame(TestCBParamVideo[A, T, V, TA]):
    idx: int
    img: np.ndarray  # TODO:
    historical: HistoricalMgr
    gt_bbox: Any

    def __init__(
        self,
        args: A,
        raw_tracker: T,
        attributes: TA,
        v_idx: int,
        video: V,
        dataset: Dataset[V],
        session_data: DataCTR,
        messenger: Callable[[str], None],
        idx: int,
        img: np.ndarray,
        historical: HistoricalMgr,
        gt_bbox,
    ) -> None:
        self.idx = idx
        self.img = img
        self.historical = historical
        self.gt_bbox = gt_bbox
        super().__init__(
            args,
            raw_tracker,
            attributes,
            v_idx,
            video,
            dataset,
            session_data,
            messenger,
        )


class TestCBParamFrameInit(TestCBParamFrame[A, T, V, TA]):
    cx: Any
    cy: Any
    w: Any
    h: Any
    gt_bbox_: List[Any]
    tic: Any

    def __init__(
        self,
        args: A,
        raw_tracker: T,
        attributes: TA,
        v_idx: int,
        video: V,
        dataset: Dataset[V],
        session_data: DataCTR,
        messenger: Callable[[str], None],
        idx: int,
        img: np.ndarray,
        historical: HistoricalMgr,
        gt_bbox,
        cx,
        cy,
        w,
        h,
        gt_bbox_: List[Any],
        tic,
    ) -> None:
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.gt_bbox_ = gt_bbox_
        self.tic = tic
        super().__init__(
            args,
            raw_tracker,
            attributes,
            v_idx,
            video,
            dataset,
            session_data,
            messenger,
            idx,
            img,
            historical,
            gt_bbox,
        )


class TestCBParamFrameDone(TestCBParamFrame[A, T, V, TA]):
    pred_bbox: List[Any]

    def __init__(
        self,
        args: A,
        raw_tracker: T,
        attributes: TA,
        v_idx: int,
        video: V,
        dataset: Dataset[V],
        session_data: DataCTR,
        messenger: Callable[[str], None],
        idx: int,
        img: np.ndarray,
        historical: HistoricalMgr,
        gt_bbox,
        pred_bbox: List[Any],
    ) -> None:
        self.pred_bbox = pred_bbox
        super().__init__(
            args,
            raw_tracker,
            attributes,
            v_idx,
            video,
            dataset,
            session_data,
            messenger,
            idx,
            img,
            historical,
            gt_bbox,
        )


class TestCBReturn(NamedTuple):
    ...


class AlignedBbox(NamedTuple):
    bbox: Any
    polygon: Any
    score: Any
    best_score: Any


# CB_P = TypeVar("CB_P")
# CB_R = TypeVar("CB_R")
# TestCB = Callable[[CB_P], CB_R]


class DoneStatus(Enum):
    NONE = 0
    Continue = 1
    Break = 2


class TestAttributes(Generic[A, V]):
    _args: A
    _video: V
    _dataset: Dataset[V]
    _results_root: str
    """
    root of current session's results
    """
    _seq_result_root: str
    """
    root of current sequence's result
    """
    is_vot_st: bool
    start_frame: int
    toc: int
    pred_bboxes: List[Union[int, List[float]]]
    done_status: DoneStatus
    #
    lost_number: int
    """VOT-ST only"""
    total_lost: int
    """VOT-ST only"""
    scores: List[Any]
    """OPE only"""
    track_times: List[Any]
    """OPE only"""

    # atk only
    atk_latency: list[float]

    @staticmethod
    def _result_file_detect(
        result_path: str, timeout: float, force: bool
    ) -> Option[DoneStatus]:
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                lines = f.readlines()
                if len(lines) == 1:
                    start_at = int(lines[0].split(",")[0])
                    if (time.time() - start_at) < timeout:
                        return Some(DoneStatus.Continue)
                    else:
                        ...
                else:
                    if not force:
                        return Some(DoneStatus.Continue)
            return NONE
        else:
            raise FileNotFoundError(
                f"result path for method `exist_result_detect` must exist: {result_path}"
            )

    @staticmethod
    def _result_file_create(result_path: str) -> None:
        result_root = os.path.dirname(result_path)
        if not os.path.exists(result_root):
            os.makedirs(result_root)
        with open(result_path, "w") as f:
            f.write(f"{int(time.time())},{os.uname()[1]}")

    def result_file_delete(self, video_name: str):
        result_path = os.path.join(
            self._seq_result_root,
            self._results_fp_template.format(video_name),
        )
        if os.path.exists(result_path):
            os.remove(result_path)
            return result_path
        else:
            return None

    def _done_status_check_fn(self) -> Optional[DoneStatus]:
        raise NotImplementedError()

    def check_done_status(self) -> DoneStatus:
        self.done_status = DoneStatus.NONE
        status = self._done_status_check_fn()
        if status is not None:
            self.done_status = status
        return self.done_status

    def set_results_root(
        self,
        args: A,
        video: V,
        dataset: Dataset[V],
        model_name: Optional[str] = None,
    ) -> None:
        """
        set properties:
        - self._results_root
        - self._seq_result_root
        """
        self._results_root = os.path.join(
            args.output_dir,
            dataset.name,
            model_name or args.model_name,
            args.variant_name + args.variant_suffix,
        )
        if dataset.name in ["VOT2019", "VOT2018", "GOT-10k"]:
            self._seq_result_root = os.path.join(
                self._results_root,
                video.name,
            )
            self._results_fp_template = "{}_001.txt"
        elif dataset.name in ["VOT2018-LT"]:
            self._seq_result_root = os.path.join(
                self._results_root,
                "longterm",
                video.name,
            )
            self._results_fp_template = "{}_001.txt"
        else:
            self._seq_result_root = self._results_root
            self._results_fp_template = "{}.txt"

    def __init__(
        self,
        args: A,
        video: V,
        dataset: Dataset[V],
        is_vot_st: Optional[bool] = None,
    ) -> None:
        def is_vot_st_dataset(dataset: str) -> bool:
            if dataset in ["VOT2016", "VOT2018", "VOT2019"]:
                return True
            else:
                return False

        self._args = args
        self._video = video
        self._dataset = dataset
        self.start_frame = 0
        self.toc = 0
        self.pred_bboxes = []
        if is_vot_st is None:
            self.is_vot_st = is_vot_st_dataset(self._dataset.name)
        else:
            self.is_vot_st = is_vot_st
        if self.is_vot_st:
            self.lost_number = 0
            self.total_lost = 0
        else:
            self.scores = []
            self.track_times = []

        self.atk_latency = []

        # set results root
        self.set_results_root(args, video, dataset)

        def _done_check_fn() -> Optional[DoneStatus]:
            seq_result_fp = os.path.join(
                self._seq_result_root,
                self._results_fp_template.format(video.name),
            )
            TIMEOUT_THLD = 60 * 60 * 24  # TODO: make this configurable
            # TIMEOUT_THLD = 60 * 60 * 2 # 2 hours
            # TIMEOUT_THLD = 60 * 5 # 5 mins
            if os.path.exists(seq_result_fp):
                status = self._result_file_detect(
                    seq_result_fp, TIMEOUT_THLD, args.force
                )
                if status.is_some():
                    return status.unwrap()
                else:
                    # timeout, overwrite
                    self._result_file_create(seq_result_fp)
            else:
                try:
                    self._result_file_create(seq_result_fp)
                except FileExistsError:
                    time.sleep(1)
                    return _done_check_fn()

        self._done_status_check_fn = _done_check_fn


def _check_file_exist(fp: str):
    if not os.path.exists(fp):
        # raise FileNotFoundError
        print(f"[ERR]: target file {fp} not found")
        exit(1)


TC = TypeVar("TC", bound=TestCliArgs)


def args_parse(
    cap_T: Type[TC], argv: list[str] = sys.argv[1:]
) -> Tuple[List[str], TC]:
    cap = Cap(cap_T)
    parsed = cap.parse(argv)
    return parsed.args, parsed.val


def initial_testing_args(
    cliargs: TestCliArgs,
    argv: List[str],
    args: TestArgs[T, C],
    config: C = CFG,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cliargs.gpu_id)

    torch.set_num_threads(1)
    #
    args.snapshot_path = cliargs.snapshot
    args.snapshot_name = cliargs.snapshot_name
    _check_file_exist(args.snapshot_path)
    config_path: str = ""
    if cliargs.config is not None:
        config_path = cliargs.config
    else:
        conf_path = auto_get_config_file(args.snapshot_path)
        if conf_path is None:
            raise FileNotFoundError(
                f"Tried to find config file with snapshot {args.snapshot_path}, but failed"
            )
        else:
            config_path = conf_path
    _check_file_exist(config_path)
    merge_from_file(config, config_path)
    args.config = config
    args.dataset = cliargs.dataset
    args.force = cliargs.force
    args.output_dir = cliargs.output_dir
    save_options = None
    save_options = ProcessVisualizerSaveOptions()
    # save_options.title = False
    # save_options.info = False
    save_options.bbox = False
    args.process_visualizer_save_options = save_options
    args.save_frame_info = cliargs.save_frame_info
    if cliargs.variant_suffix is not None:
        args.variant_suffix = cliargs.variant_suffix
    args.video = argv
    args.visualize = cliargs.visualize
    args.model_name = args.snapshot_path.split("/")[-1].split(".")[0]


@overload
def get_dataset(
    name: ValidDatasetNames_LaSOT, fixed_dataset_root: Optional[str] = None
) -> Datasets.LaSOTDataset:
    ...


@overload
def get_dataset(
    name: ValidDatasetNames_NFS, fixed_dataset_root: Optional[str] = None
) -> Datasets.NFSDataset:
    ...


@overload
def get_dataset(
    name: ValidDatasetNames_OTB, fixed_dataset_root: Optional[str] = None
) -> Datasets.OTBDataset:
    ...


@overload
def get_dataset(
    name: ValidDatasetNames_VOT, fixed_dataset_root: Optional[str] = None
) -> Datasets.VOTDataset:
    ...


@overload
def get_dataset(
    name: ValidDatasetNames_VOTLT, fixed_dataset_root: Optional[str] = None
) -> Datasets.VOTLTDataset:
    ...


@overload
def get_dataset(
    name: ValidDatasetNames_UAV, fixed_dataset_root: Optional[str] = None
) -> Datasets.UAVDataset:
    ...


@overload
def get_dataset(
    name: ValidDatasetNames, fixed_dataset_root: Optional[str] = None
) -> Dataset:
    ...


# TODO: fixed_dataset_root impl
def get_dataset(
    name: str, fixed_dataset_root: Optional[str] = None
) -> Dataset:
    dataset_root: str
    load_img: bool = False

    if name in ["VOT2016", "VOT2018", "VOT2019"]:
        dataset_root = os.path.join(ENV.dset_root_testing, name)
    elif name == "OTB100":
        dataset_root = os.path.join(ENV.dset_root_testing, name)
    elif name == "UAV123":
        dataset_root = os.path.join(
            ENV.dset_root_testing, "UAV123", "data_seq", "UAV123"
        )
    elif name == "LaSOT":
        dataset_root = os.path.join(
            ENV.dset_root_testing, "lasot", "LaSOTTesting"
        )
    elif name in ["NFS30", "NFS240"]:
        dataset_root = os.path.join(
            ENV.dset_root_testing,
            "NFS",
        )
    else:
        raise NotImplementedError(
            f"dataset <{name}> are not been supported for now."
        )

    return DatasetFactory.create_dataset(
        name=name,
        dataset_root=dataset_root,
        load_img=load_img,
    )


def save_track_tracking_result_default(params: TestCBParamVideo) -> None:
    video = params.video
    dataset = params.dataset
    attr = params.attributes

    result_fp = os.path.join(
        attr._seq_result_root,
        attr._results_fp_template.format(video.name),
    )

    if attr.is_vot_st:
        with open(result_fp, "w") as f:
            for x in attr.pred_bboxes:
                if isinstance(x, int):
                    f.write(f"{x}\n")
                else:
                    f.write(
                        ",".join([vot_float2str("%.4f", i) for i in x]) + "\n"
                    )
    else:
        with open(result_fp, "w") as f:
            for x in attr.pred_bboxes:
                f.write(",".join([str(i) for i in x]) + "\n")

    # extra
    if dataset.name in ["VOT2018-LT"]:
        conf_fp = os.path.join(
            attr._seq_result_root, f"{video.name}_001_confidence.value"
        )
        with open(conf_fp, "w") as f:
            for x in attr.scores:
                f.write("\n") if x is None else f.write(f"{x:.6f}\n")
        time_fp = os.path.join(attr._seq_result_root, f"{video.name}_time.txt")
        with open(time_fp, "w") as f:
            for x in attr.track_times:
                f.write(f"{x:.6f}\n")
    if dataset.name in ["GOT-10k"]:
        time_fp = os.path.join(attr._seq_result_root, f"{video.name}_time.txt")
        with open(time_fp, "w") as f:
            for x in attr.track_times:
                f.write(f"{x:.6f}\n")


class VPReg:
    _cp_list: List[Tuple[str, CP]]
    _pp_list: List[Tuple[str, PP]]
    _trkpp_list: List[Tuple[str, TrackerInputProcess]]

    def __init__(self) -> None:
        self._cp_list = []
        self._pp_list = []
        self._trkpp_list = []

    def add(
        self,
        processor: Union[VP, TrackerInputProcess],
        group: str,
    ) -> None:
        if isinstance(processor, PP):
            self._pp_list.append((group, processor))
        elif isinstance(processor, CP):
            self._cp_list.append((group, processor))
        elif isinstance(processor, TrackerInputProcess):
            self._trkpp_list.append((group, processor))
        else:
            raise ValueError(
                f"Unknown processor type: {processor.__class__.name}"
            )

    def get_pp_list(self, group: str) -> List[PP]:
        return [pp for g, pp in self._pp_list if group == g]

    def get_pp_groups(self) -> Set[str]:
        return {g for g, _ in self._pp_list}

    def get_cp_list(self, group: str) -> List[CP]:
        return [cp for g, cp in self._cp_list if group == g]

    def get_cp_groups(self) -> Set[str]:
        return {g for g, _ in self._cp_list}

    def get_trkpp_list(self, group: str) -> List[TrackerInputProcess]:
        return [trkpp for g, trkpp in self._trkpp_list if group == g]

    def get_trkpp_groups(self) -> Set[str]:
        return {g for g, _ in self._trkpp_list}

    def remove_all(self) -> None:
        self._cp_list.clear()
        self._pp_list.clear()
        self._trkpp_list.clear()

    def reset_all(self) -> None:
        for _, cp in self._cp_list:
            cp.reset()
        for _, pp in self._pp_list:
            pp.reset()
        for _, trkpp in self._trkpp_list:
            trkpp.reset()


class TrkppGroupResults(Generic[T]):
    outputs: TrackingOutputs
    overlap_ratio: float
    dummy: T

    def __init__(
        self,
        outputs: TrackingOutputs,
        overlap_ratio: float,
        dummy: T,
    ) -> None:
        self.outputs = outputs
        self.overlap_ratio = overlap_ratio
        self.dummy = dummy


class _GroupResults:
    _iou_results: Dict[str, Dict[int, float]]

    def __init__(self) -> None:
        self._iou_results = {}

    def add_iou_result(
        self, group_name: str, frame_id: int, iou_overlap: float
    ) -> None:
        self._iou_results.setdefault(group_name, {})[frame_id] = iou_overlap

    def save(self, result_root_dir: str) -> None:
        os.makedirs(result_root_dir, exist_ok=True)
        for group in self._iou_results.keys():
            fp = os.path.join(
                result_root_dir,
                f"iou_{group}.txt",
            )
            values: List[float] = []
            for i in range(max(self._iou_results[group].keys()) + 1):
                values.append(self._iou_results[group].get(i, 0.0))
            with open(fp, "w") as f:
                f.write("\n".join([f"{v:.6f}" for v in values]))


class ExecExtraAttr(TypedDict, total=False):
    groups_sort_fn: Callable[[str], Any]
    datasets_progress: tuple[int, int]  # cur, tot


# class Test(Generic[A, T]):
class Test(Generic[T, A, V, TA]):
    args: A
    tracker_cls: Type[T]
    test_attr_cls: Type[TA]
    historical_mgr: HistoricalMgr
    """only accessable during Test.exec"""
    raw_tracker: T

    callback_init: Callable[
        [TestCBParamFrameInit[A, T, V, TA], Optional[T]],
        None,
    ]
    callback_track: Callable[
        [TestCBParamFrame[A, T, V, TA], Optional[T], Optional[dict]],
        TrackingOutputs,
    ]  # TODO:
    callback_skip: Callable[
        [TestCBParamFrame[A, T, V, TA]],
        None,
    ]
    callback_frame_done: Callable[
        [TestCBParamFrameDone[A, T, V, TA]],
        Optional[DoneStatus],
    ]
    callback_video_done: Callable[
        [TestCBParamVideo[A, T, V, TA]],
        Optional[DoneStatus],
    ]

    processor_reg: VPReg

    def __init__(
        self,
        args: A,
        tracker_cls: Type[T] = SiamRPNTracker,
        test_attr_cls: Type[TA] = TestAttributes[A, V],
    ) -> None:
        self.args = args
        self.tracker_cls = tracker_cls
        self.test_attr_cls = test_attr_cls
        self.processor_reg = VPReg()

        def trk_init(
            params: TestCBParamFrameInit[A, T, V, TA],
            dummy: Optional[T],
        ) -> None:
            tracker = params.raw_tracker
            img = params.img
            gt_bbox_ = params.gt_bbox_
            #
            tracker.init(img, gt_bbox_, dummy)

        def trk_track(
            params: TestCBParamFrame[A, T, V, TA],
            dummy: Optional[T],
            extra_options: Optional[dict] = None,
        ) -> TrackingOutputs:
            tracker = params.raw_tracker
            img = params.img
            #
            if extra_options is not None:
                extra_options.pop("extra_options", False)
            return tracker.track(img, dummy)  # type: ignore

        def trk_video_done(params: TestCBParamVideo[A, T, V, TA]) -> None:
            v_idx = params.v_idx
            video = params.video
            session_data = params.session_data
            attr = params.attributes
            #
            save_track_tracking_result_default(params)
            if attr.is_vot_st:
                tot_lost = (
                    session_data.get_data("vot_tot_lost_number", 0)
                    + attr.lost_number
                )
                session_data.set_data(
                    "vot_tot_lost_number",
                    tot_lost,
                    mutable=True,
                )
                Test.show_details(
                    v_idx,
                    video.name,
                    attr.toc,
                    len(video),
                    attr.lost_number,
                    tot_lost,
                )
            # else:
            #     Test.show_details(
            #         v_idx,
            #         video.name,
            #         attr.toc,
            #         len(video),
            #     )

        def empty_cb(params: Any) -> None:
            ...

        self.callback_init = trk_init
        self.callback_track = trk_track
        self.callback_skip = empty_cb
        self.callback_frame_done = empty_cb
        self.callback_video_done = trk_video_done

    def initial_tracker(self) -> T:
        model = ModelBuilder()
        model = load_pretrain(model, self.args.snapshot_path).cuda().eval()
        self.raw_tracker = self.tracker_cls(model)
        return self.raw_tracker

    @staticmethod
    def show_details(
        video_idx: int,
        video_name: str,
        duration: float,
        frame_cnt: int,
        lost_number: Optional[int] = None,
        total_lost: Optional[int] = None,
    ):
        msg = [
            f"[{str(video_idx + 1).zfill(3)}]",
            f"Video: {video_name:12s}",
            f"Time: {duration:5.1f}s",
            f"Speed: {(frame_cnt / duration):3.1f}fps",
        ]
        if lost_number is not None:
            msg.append(f"Lost: {lost_number}")
        if total_lost is not None:
            msg.append(f"tot {total_lost}")
        print("\r" + " ".join(msg))

    def add_processor(
        self,
        processor: Union[VP, TrackerInputProcess],
        group: str = "default",
    ):
        if isinstance(processor, TrackerInputProcess):
            processor.linked_tracker = self.raw_tracker
        self.processor_reg.add(processor, group)

    def remove_all_processors(self):
        self.processor_reg.remove_all()

    @property
    def history(self) -> HistoricalMgr:
        return self.historical_mgr

    def exec(
        self,
        dataset: Dataset[V],
        # dataset: Dataset,
        # cb:TestCB[TestCBParamInit[A, SiamRPNTracker, V], None],
        skip_frames: int = 5,
        test_attr_params: Dict = {},
        sequences: List[str] = [],
        extra_attr: ExecExtraAttr = {},
    ):
        variant = self.args.variant_name + self.args.variant_suffix
        print("-----------------------------")
        print(f"|  `Test.exec` is running")
        dset_prog = extra_attr.get("datasets_progress", None)
        dset_prog = (
            f" ({dset_prog[0]} of {dset_prog[1]})"
            if dset_prog is not None
            else ""
        )
        print(f"|   dataset : {dataset.name}{dset_prog}")
        print(f"|     model : {self.args.model_name}")
        print(f"|   variant : {variant}")
        print(f'|  start at : {get_current_dt("%Y-%m-%d %H:%M:%S")}')
        print("-----------------------------")

        if self.args.tracker_cb is not None:
            # TODO:
            print("[DBG] applying tracker callback...")
            self.args.tracker_cb(self.raw_tracker)

        session_data = DataCTR()  # persistent data across videos

        for v_idx, video in enumerate(dataset):
            self.historical_mgr = HistoricalMgr(
                video,
                # mem_size=min(10, len(video) // 100),
            )
            self.processor_reg.reset_all()

            if len(self.args.video) != 0:
                if video.name not in self.args.video:
                    continue
            if len(sequences) != 0:
                if video.name not in sequences:
                    continue

            attr = self.test_attr_cls(
                **{
                    **{
                        "args": self.args,
                        "video": video,
                        "dataset": dataset,
                    },
                    **test_attr_params,
                }
            )
            attr.check_done_status()
            if attr.done_status is DoneStatus.Continue:
                continue
            if attr.done_status is DoneStatus.Break:
                break

            # trkpp initialization
            for grp in self.processor_reg.get_trkpp_groups():
                for trk_pp in self.processor_reg.get_trkpp_list(grp):
                    trk_pp.linked_historical_mgr = self.historical_mgr
            grp_results = _GroupResults()

            try:
                pbar = tqdm(
                    video,
                    desc=f"{video.name:>10} {v_idx+1}/{len(dataset)}",
                    ncols=60,
                )
                for idx, (img, gt_bbox) in enumerate(pbar):
                    img = img.copy()
                    if attr.is_vot_st:
                        if len(gt_bbox) == 4:
                            gt_bbox = [
                                gt_bbox[0],
                                gt_bbox[1],
                                gt_bbox[0],
                                gt_bbox[1] + gt_bbox[3] - 1,
                                gt_bbox[0] + gt_bbox[2] - 1,
                                gt_bbox[1] + gt_bbox[3] - 1,
                                gt_bbox[0] + gt_bbox[2] - 1,
                                gt_bbox[1],
                            ]
                    tic = cv2.getTickCount()
                    his_frame = HistoricalFrame(
                        bbox=gt_bbox,
                        bbox_pred=None,
                        img=img,
                        aligned_bbox=get_axis_aligned_bbox(np.array(gt_bbox)),
                    )
                    his_tracking = HistoricalTracking(
                        x_crop=None,
                        x_crop_gt=None,
                        z_crop=None,
                    )
                    self.historical_mgr.next(
                        frame=his_frame,
                        tracking=Some(his_tracking),
                    )

                    trkpp_visualizer = ProcessVisualizer(
                        self.args.visualize,
                        save_options=self.args.process_visualizer_save_options,
                    )
                    trkpp_visualizer.save_dir = os.path.join(
                        attr._results_root,
                        "vis_trkpp",
                        video.name,
                    )
                    trkpp_visualizer.fname = f"{idx:05d}.jpg"
                    self.raw_tracker.trk_pp_visualizer = trkpp_visualizer

                    pp_list = self.processor_reg.get_pp_list(
                        group="default",  # TODO:
                    )
                    if len(pp_list) > 0:
                        for processor in pp_list:
                            img = processor.forward(img, self.historical_mgr)
                        if isinstance(img, torch.Tensor):
                            img = tensor2mat(img)

                    if idx != 0:
                        his_tracking.x_crop_gt = self.raw_tracker.get_x_crop(
                            img
                        )[0]
                    if idx == attr.start_frame:
                        (
                            cx,
                            cy,
                            w,
                            h,
                        ) = self.historical_mgr.cur.frame.aligned_bbox
                        gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                        callback_param = TestCBParamFrameInit(
                            args=self.args,
                            raw_tracker=self.raw_tracker,
                            attributes=attr,
                            v_idx=v_idx,
                            video=video,
                            dataset=dataset,
                            session_data=session_data,
                            messenger=pbar.write,
                            idx=idx,
                            img=img,
                            historical=self.historical_mgr,
                            gt_bbox=gt_bbox,
                            cx=cx,
                            cy=cy,
                            w=w,
                            h=h,
                            gt_bbox_=gt_bbox_,
                            tic=tic,
                        )
                        self.callback_init(callback_param, None)
                        pred_bbox = gt_bbox_
                        if attr.is_vot_st:
                            attr.pred_bboxes.append(1)
                        elif idx == 0:
                            attr.scores.append(None)
                            if "VOT2018-LT" == dataset.name:
                                attr.pred_bboxes.append([1])
                            else:
                                attr.pred_bboxes.append(pred_bbox)
                    else:
                        callback_param = TestCBParamFrame(
                            args=self.args,
                            raw_tracker=self.raw_tracker,
                            attributes=attr,
                            v_idx=v_idx,
                            video=video,
                            dataset=dataset,
                            session_data=session_data,
                            messenger=pbar.write,
                            idx=idx,
                            img=img,
                            historical=self.historical_mgr,
                            gt_bbox=gt_bbox,
                        )
                        if idx > attr.start_frame:
                            results: Dict[str, TrkppGroupResults] = {}
                            trkpp_groups = (
                                self.processor_reg.get_trkpp_groups()
                            )

                            for grp in trkpp_groups:
                                trkpp_list = self.processor_reg.get_trkpp_list(
                                    group=grp,
                                )
                                # for keep_clean in (False, True): # REMOVE THIS: unsafe
                                # if keep_clean:
                                #     grp = f'{grp}_clean_unsafe' # REMOVE THIS:
                                # print("trkpp_list", trkpp_list)
                                self.raw_tracker.trk_pp_list = trkpp_list
                                self.raw_tracker.trk_pp_visualizer.suffix = grp
                                self.raw_tracker.trk_pp_visualizer.batch_mode = (
                                    True
                                )
                                self.raw_tracker.trk_pp_visualizer._idx = idx
                                self.raw_tracker.trk_pp_visualizer._video = (
                                    video
                                )

                                # TODO: 2023/03/27
                                dummy = self.raw_tracker.create_dummy()
                                dummy.size = self.raw_tracker.size
                                dummy.center_pos = self.raw_tracker.center_pos
                                dummy.channel_average = (
                                    self.raw_tracker.channel_average
                                )

                                # TODO: unsafe
                                extra_options = {}
                                # extra_options = { 'keep_clean': keep_clean }
                                #
                                outputs = self.callback_track(
                                    callback_param, dummy, extra_options
                                )
                                for key in extra_options.keys():
                                    raise NotImplementedError(
                                        f"Unable to handle extra option {key}"
                                    )

                                overlap_ratio: float
                                if attr.is_vot_st:
                                    overlap_ratio = vot_overlap(
                                        outputs["bbox"],
                                        gt_bbox,
                                        (img.shape[1], img.shape[0]),
                                    )
                                else:
                                    overlap_ratio = iou_overlap(
                                        outputs["bbox"], gt_bbox
                                    )
                                self.raw_tracker.trk_pp_visualizer.add_info(
                                    f"overlap: {overlap_ratio:.6f}", name=None
                                )

                                if callback_param.args.visualize:

                                    def calculate_bbox_in_crop(
                                        center_pos,
                                        bbox,
                                        inp_size,
                                        scale,
                                        is_center=False,
                                    ):
                                        center_x, center_y = center_pos
                                        bbox_cx, bbox_cy, bbox_w, bbox_h = bbox

                                        crop_x = center_x - inp_size // 2
                                        crop_y = center_y - inp_size // 2

                                        bbox_cx_crop = bbox_cx - crop_x
                                        bbox_cy_crop = bbox_cy - crop_y

                                        bbox_w_crop = bbox_w * scale
                                        bbox_h_crop = bbox_h * scale

                                        tl_x = bbox_cx_crop - bbox_w_crop / 2
                                        tl_y = bbox_cy_crop - bbox_h_crop / 2

                                        tl_x = max(0, min(tl_x, inp_size))
                                        tl_y = max(0, min(tl_y, inp_size))

                                        return [
                                            tl_x,
                                            tl_y,
                                            bbox_w_crop,
                                            bbox_h_crop,
                                        ]

                                    scale = outputs["scale"]  # type: ignore

                                    gt_bbox = (
                                        self.historical_mgr.cur.frame.aligned_bbox
                                    )
                                    self.raw_tracker.trk_pp_visualizer.add_bbox_to_last(
                                        calculate_bbox_in_crop(
                                            # dummy.center_pos,
                                            self.raw_tracker.center_pos,
                                            gt_bbox,
                                            CFG.track.instance_size,
                                            scale,
                                        ),
                                        color=(86, 243, 113),
                                    )

                                    x1, y1, w, h = outputs["bbox"]
                                    cx = x1 + w // 2
                                    cy = y1 + h // 2
                                    if grp == "nodef_pysot_pt":
                                        b_color = (244, 112, 103)
                                    else:
                                        b_color = (55, 148, 255)
                                    self.raw_tracker.trk_pp_visualizer.add_bbox_to_last(
                                        calculate_bbox_in_crop(
                                            # dummy.center_pos,
                                            self.raw_tracker.center_pos,
                                            # outputs["bbox"],
                                            [cx, cy, w, h],
                                            CFG.track.instance_size,
                                            scale,
                                        ),
                                        color=b_color,
                                    )

                                results[grp] = TrkppGroupResults(
                                    outputs=outputs,
                                    # collected=self.raw_tracker.collect_saved(),
                                    overlap_ratio=overlap_ratio,
                                    dummy=dummy,
                                )
                            else:
                                if len(trkpp_groups) == 0:
                                    self.raw_tracker.trk_pp_list = []
                                    outputs = self.callback_track(
                                        callback_param, None, None
                                    )
                                else:
                                    for grp, res in results.items():
                                        grp_results.add_iou_result(
                                            grp,
                                            idx,
                                            res.overlap_ratio,
                                        )

                                    # TODO: better selection policy: current is best iou
                                    sorted_groups = sorted(
                                        results.items(),
                                        key=lambda x: x[1].overlap_ratio,
                                        reverse=True,
                                    )
                                    best = sorted_groups[0][1]
                                    outputs = best.outputs
                                    self.raw_tracker.load_attrs_from_dummy(
                                        best.dummy
                                    )

                                    # highlighting trk_pp_visualizer
                                    if (
                                        self.raw_tracker.trk_pp_visualizer.save_options.highlight_best
                                    ):
                                        for (
                                            n,
                                            vis_im,
                                        ) in (
                                            self.raw_tracker.trk_pp_visualizer.vis_saved.items()
                                        ):
                                            # n is "{trk_pp.name}_{grp}"
                                            if n.endswith(sorted_groups[0][0]):
                                                self.raw_tracker.trk_pp_visualizer.vis_saved[
                                                    n
                                                ] = add_border(
                                                    vis_im,
                                                    width=10,
                                                    color=self.raw_tracker.trk_pp_visualizer.save_options.highlight_best_border_color,
                                                    inner=True,
                                                )

                                    self.raw_tracker.trk_pp_visualizer.save(
                                        sort_fn=extra_attr.get(
                                            "groups_sort_fn", None
                                        ),
                                        force=True,
                                    )

                                    # self.raw_tracker._saved = (
                                    #     best.collected
                                    # )  # for feature extraction

                            if attr.is_vot_st:
                                if self.args.config.mask.mask:
                                    pred_bbox = outputs["polygon"]
                                else:
                                    pred_bbox = outputs["bbox"]
                                    his_frame.bbox_pred = (
                                        pred_bbox  # video_historical
                                    )
                                overlap: float = vot_overlap(
                                    pred_bbox,
                                    gt_bbox,
                                    (img.shape[1], img.shape[0]),
                                )
                                if overlap > 0:
                                    attr.pred_bboxes.append(pred_bbox)  # type: ignore
                                else:
                                    attr.pred_bboxes.append(2)
                                    attr.start_frame = idx + skip_frames
                                    attr.lost_number += 1
                                    # lost_here = True
                            else:
                                pred_bbox = outputs["bbox"]
                                his_frame.bbox_pred = (
                                    pred_bbox  # video_historical
                                )
                                attr.pred_bboxes.append(pred_bbox)
                                attr.scores.append(outputs["best_score"])
                        else:
                            # vot_st only
                            self.callback_skip(callback_param)
                            attr.pred_bboxes.append(0)
                    attr.toc += cv2.getTickCount() - tic
                    if not attr.is_vot_st:
                        attr.track_times.append(
                            (cv2.getTickCount() - tic) / cv2.getTickFrequency()
                        )
                    callback_param = TestCBParamFrameDone(
                        args=self.args,
                        raw_tracker=self.raw_tracker,
                        attributes=attr,
                        v_idx=v_idx,
                        video=video,
                        dataset=dataset,
                        session_data=session_data,
                        messenger=pbar.write,
                        idx=idx,
                        img=img,
                        historical=self.historical_mgr,
                        gt_bbox=gt_bbox,
                        pred_bbox=pred_bbox,  # type: ignore
                    )

                    # data collection
                    data = self.raw_tracker.collect_saved()
                    his_tracking.z_crop = data.get("z_crop", None)
                    his_tracking.x_crop = data.get("x_crop", None)
                    his_tracking.x_crop_trkpp_free = data.get(
                        "x_crop_trkpp_free", None
                    )

                    # dump trkpp group results
                    grp_results.save(
                        os.path.join(
                            attr._results_root,
                            "iou_analysis",
                            video.name,
                        )
                    )

                    done_status = self.callback_frame_done(callback_param)
                    if done_status is DoneStatus.Break:
                        break

                pbar.update(1)
                pbar.refresh()

                attr.toc /= cv2.getTickFrequency()
                callback_param = TestCBParamVideo(
                    args=self.args,
                    raw_tracker=self.raw_tracker,
                    attributes=attr,
                    v_idx=v_idx,
                    video=video,
                    dataset=dataset,
                    session_data=session_data,
                    messenger=pbar.write,
                )
                self.callback_video_done(callback_param)
            except KeyboardInterrupt:
                try:
                    print("\ninterrupted by user")
                    r = input("remove result? [Y/n]")
                    a = input("skip or exit? [s/E]")
                    if r.lower() != "n":
                        rm = attr.result_file_delete(video.name)
                        if rm is not None:
                            print(f"removed corrupt result:\n\t{rm}")
                    if a.lower() != "s":
                        exit(130)
                except KeyboardInterrupt:
                    rm = attr.result_file_delete(video.name)
                    if rm is not None:
                        print(f"removed corrupt result:\n\t{rm}")
                        exit(130)

            except Exception as e:
                rm = attr.result_file_delete(video.name)
                if rm is not None:
                    print(f"removed corrupt result:\n\t{rm}")
                raise e

            if not session_data.has("attack_latency"):
                session_data.set_data("attack_latency", [], mutable=True)
            latency_list = session_data.get_data("attack_latency").append(
                np.mean(attr.atk_latency)
            )

            # release
            del attr
            self.historical_mgr.flush()

        delattr(self, "historical_mgr")

        print("-----------------------------")
        print(f"|  `Test.exec` is done")
        if session_data.has("attack_latency"):
            latency = session_data.get_data("attack_latency")
            print(f"| average attack latency: {np.mean(latency) * 1000:.3f}ms")
        print(f'|  done at  : {get_current_dt("%Y-%m-%d %H:%M:%S")}')
        print("-----------------------------")
