import cv2
import numpy as np
import os
import pickle
import torch
from attackers.spark.SPARK.attacker.attacker_builder import (
    ValidAttacker,
    build_attacker,
)
from attackers.spark.SPARK.attacker.oim_atk_cfgmod import OIMAttacker
from attackers.spark.SPARK.cfg import CFG, SPARK_Config
from attackers.spark.SPARK.tracker.siamrpn_tracker import SiamRPNTracker_SPARK
from attackers.spark.SPARK.types import AtkType, NormType, RegType
from pysot.models.model_builder_cfgmod import ModelBuilder
from pyotp.tools.test import (
    DoneStatus,
    TestAttributes as _TestAttributes,
    TestCBParamFrame,
    TestCBParamFrameDone,
    TestCBParamFrameInit,
    TestCBParamVideo,
    TrackingOutputs,
    args_parse,
    initial_testing_args as _initial_testing_args,
    Test,
    TestArgs as _TestArgs,
    TestCliArgs as _TestCliArgs,
    V,
    save_track_tracking_result_default,
)
from pysot.utils.bbox import get_axis_aligned_bbox
from toolkit.datasets.dataset import Dataset
from toolkit.datasets.video import Video
from typing import (
    Any,
    List,
    Optional,
    Tuple,
    TypeVar,
)


class TestCliArgs(_TestCliArgs):
    accframes: int = 30
    apts: bool = False
    """whether attacking apts"""
    apts_num: int = 2
    attack_method: ValidAttacker = "OIM"
    """select different kinds of attacker"""
    attack_type: AtkType = AtkType.UA
    """UA or TA"""
    enable_same_pert: bool = False
    """whether tell the same objective"""
    eplison: float = 3e-1
    interval: int = -1
    """attack interval"""
    name_suffix: str = ""
    """L_inf, L_1 or L_2"""  # ?
    norm_type: NormType = NormType.L_inf
    max_num: int = 10
    """max iteration"""
    opt_flow: bool = False
    """whether using optical flow"""
    reg_type: RegType = RegType.L21


T = TypeVar("T", bound=SiamRPNTracker_SPARK)  # TODO:


class TestArgs(_TestArgs[SiamRPNTracker_SPARK, SPARK_Config]):
    accframes: int
    apts: bool
    apts_num: int
    atk_method: ValidAttacker
    atk_type: AtkType
    enable_same_pert: bool
    eplison: float
    interval: int
    name_suffix: str
    norm_type: NormType
    max_num: int
    opt_flow: bool
    reg_type: RegType
    spark_visualize: bool


class TestAttributes(_TestAttributes[TestArgs, V]):
    attacker: OIMAttacker
    attack_times: List[Any]
    iter_nums: List[int]
    track_times: List[float]
    #
    target_traj: Any
    # pert_degs: List[List[np.ndarray]]
    pert_degs: List[List]
    attacked_root: str
    attack_toc: Any
    out: Optional[Any]
    """AtkType.TA only"""
    #
    result_path: str
    pert_path: str
    attack_time_path: str
    track_time_path: str
    iternum_path: str

    # using during frame iter
    over_iter_prev_perts: Optional[torch.Tensor] = None
    over_iter_weights: Optional[torch.Tensor] = None
    over_iter_APTS: bool = False
    over_iter_OPTICAL_FLOW: bool = False
    over_iter_ADAPT: bool = False

    def __init__(
        self,
        args: TestArgs,
        video: V,
        dataset: Dataset[V],
        attacker: OIMAttacker,  # TODO:
        attacked_root: str,
        atk_type: AtkType,
    ) -> None:
        super().__init__(
            args,
            video,
            dataset,
            # is_vot_st=False, # FIXME: wtf is this
        )
        self.attacker = attacker
        self.attack_times = []
        self.iter_nums = []
        self.track_times = []
        self.attacked_root = attacked_root

        args = self._args
        video = self._video

        if args.config.attacker.save_video:
            raise NotImplementedError

        if atk_type is AtkType.TA:
            traj_path = os.path.join(
                self.attacked_root,
                self._dataset.name,
                f"{video.name}_ta_traj.pkl",
            )
            if os.path.exists(traj_path):
                with open(traj_path, "rb") as f:
                    target_traj = pickle.load(f)
                    attacker.target_traj = target_traj
            else:
                if not os.path.exists(os.path.dirname(traj_path)):
                    os.makedirs(os.path.dirname(traj_path))
                with open(traj_path, "wb") as f:
                    try:
                        init_rect = video._gt_bboxes[0]  # h5
                        v_len = len(video)
                    except AttributeError:
                        init_rect = video.init_rect
                        v_len = len(video.img_names)
                    target_traj = attacker.target_traj_gen(
                        init_rect,
                        video.height,
                        video.width,
                        v_len,
                    )
                    pickle.dump(target_traj, f)
            self.target_traj = target_traj

        if args.config.attacker.eval:
            self.pert_degs = []

        if not os.path.isdir(self._seq_result_root):
            os.makedirs(self._seq_result_root, exist_ok=True)

        self.result_path = os.path.join(
            self._seq_result_root,
            f"{video.name}.txt",
        )
        self.pert_path = os.path.join(
            self._seq_result_root,
            f"{video.name}_pert.txt",
        )
        self.attack_time_path = os.path.join(
            self._seq_result_root,
            f"{video.name}_attack_time.txt",
        )
        self.track_time_path = os.path.join(
            self._seq_result_root,
            f"{video.name}_track_time.txt",
        )
        self.iternum_path = os.path.join(
            self._seq_result_root,
            f"{video.name}_iternum.txt",
        )

        def _done_check_fn() -> Optional[DoneStatus]:
            TIMEOUT_THLD = 60 * 60 * 24  # TODO: make this configurable
            if args.config.attacker.check_exist:
                if os.path.exists(self.result_path):
                    status = self._result_file_detect(
                        self.result_path,
                        TIMEOUT_THLD,
                        args.force,
                    )
                    if status.is_some():
                        return status.unwrap()
                    elif (
                        # TODO:
                        os.path.exists(self.result_path)
                        and os.path.exists(self.pert_path)
                        and os.path.exists(self.attack_time_path)
                        and os.path.exists(self.track_time_path)
                    ):
                        return DoneStatus.Continue
                else:
                    self._result_file_create(self.result_path)
            else:
                ...

        self._done_status_check_fn = _done_check_fn


def initial_testing_args(
    cliargs: TestCliArgs,
    argv: List[str],
    args: TestArgs,
) -> None:
    _initial_testing_args(cliargs, argv, args, config=CFG)
    args.accframes = cliargs.accframes
    args.apts = cliargs.apts
    args.apts_num = cliargs.apts_num
    args.atk_method = cliargs.attack_method
    args.atk_type = cliargs.attack_type
    args.enable_same_pert = cliargs.enable_same_pert
    args.eplison = cliargs.eplison
    args.interval = cliargs.interval
    args.name_suffix = cliargs.name_suffix
    args.norm_type = cliargs.norm_type
    args.max_num = cliargs.max_num
    args.opt_flow = cliargs.opt_flow
    args.reg_type = cliargs.reg_type
    args.spark_visualize = False  # keep this away from PYOTP's visualization
    #
    save_name_split = [
        args.model_name,
        args.atk_method,
        args.atk_type.name,
        "FLOW" if args.opt_flow else "noFLOW",
        "ADAPT" if args.interval == -1 else "noADAPT" + str(args.interval),
        "APTS" + str(args.apts_num) if args.apts else "noAPTS",
        "SPERT" if args.enable_same_pert else "noSPERT",
        "REG" + args.reg_type.name,
        "NORM" + args.norm_type.name,
    ]
    args.model_name = "_".join(save_name_split)


TestAlias = Test[SiamRPNTracker_SPARK, TestArgs, V, TestAttributes]


def _initial_spark(test: TestAlias) -> OIMAttacker:
    raw_tracker = test.initial_tracker()

    def get_attacker(test: TestAlias):
        args = test.args
        attacker: OIMAttacker = build_attacker(
            args.atk_method,
            args.atk_type,
            args.max_num,
            args.apts_num,
            reg_type=args.reg_type,
            norm_type=args.norm_type,
            eplison=args.eplison,
            accframes=args.accframes,
        )
        return attacker

    return get_attacker(test)


def initial_tester(
    args: TestArgs,
) -> Tuple[TestAlias, OIMAttacker]:
    test = Test(
        args,
        SiamRPNTracker_SPARK,
        TestAttributes,
    )
    attacker = _initial_spark(test)

    def spark_trk_init(
        params: TestCBParamFrameInit[
            TestArgs, SiamRPNTracker_SPARK, Video, TestAttributes
        ],
        dummy: Optional[T],
    ):
        args = params.args
        tracker = params.raw_tracker
        attr = params.attributes
        img = params.img
        gt_bbox_ = params.gt_bbox_
        tic = params.tic
        #
        tracker.init(img, gt_bbox_, dummy)
        attr.attack_toc = tic
        if args.config.attacker.eval:
            attr.pert_degs.append([0])
            if args.atk_type is AtkType.UA:
                tpos = []
                tpos.append(params.cx)
                tpos.append(params.cy)
                attacker.target_traj.append(tpos)
            attr.attack_times.append([0])

    def spark_trk_track(
        params: TestCBParamFrame[
            TestArgs, SiamRPNTracker_SPARK, Video, TestAttributes
        ],
        dummy: Optional[T],
        extra_options: Optional[dict] = None,
    ):
        args = params.args
        tracker = params.raw_tracker
        img = params.img
        gt_bbox = params.gt_bbox
        idx = params.idx
        attr = params.attributes
        log = params.messenger

        if extra_options is not None:
            if extra_options.pop("keep_clean", False):
                return tracker.track(
                    img,
                    torch.tensor([]),
                    dummy,
                    is_perturbed=False,
                )

        # start attacking
        attack_tic = cv2.getTickCount()
        attacker.v_id = idx
        prev_perts = attr.over_iter_prev_perts
        weights = attr.over_iter_weights
        APTS = attr.over_iter_APTS
        OPTICAL_FLOW = attr.over_iter_OPTICAL_FLOW
        ADAPT = attr.over_iter_ADAPT
        if idx == 1:
            prev_perts, weights, APTS, OPTICAL_FLOW, ADAPT = (
                None,
                None,
                False,
                False,
                False,
            )
        # only attack the first frame
        elif args.interval == 1000:
            APTS, OPTICAL_FLOW, ADAPT = False, False, False
        elif args.interval > 0:
            if (idx - 1) % args.interval == 0:
                prev_perts, weights, APTS, OPTICAL_FLOW, ADAPT = (
                    None,
                    None,
                    args.apts,
                    args.opt_flow,
                    False,
                )
            else:
                APTS, OPTICAL_FLOW, ADAPT = args.apts, args.opt_flow, False
        # adaptive attack
        elif args.interval == -1:
            APTS, OPTICAL_FLOW, ADAPT = args.apts, args.opt_flow, True

        if prev_perts is None:
            t_prev_perts = prev_perts
        else:
            t_prev_perts = prev_perts.clone().detach()

        if weights is None:
            t_weights = weights
        else:
            t_weights = weights.clone().detach()

        import time

        t0 = time.time()

        x_crop, pert_true, prev_perts, weights, img_ = attacker.attack(
            tracker,
            img,
            t_prev_perts,
            t_weights,
            APTS,
            OPTICAL_FLOW,
            ADAPT,
            Enable_same_prev=args.enable_same_pert,
        )
        attr.over_iter_prev_perts = prev_perts
        attr.over_iter_weights = weights
        attr.over_iter_APTS = APTS
        attr.over_iter_OPTICAL_FLOW = OPTICAL_FLOW
        attr.over_iter_ADAPT = ADAPT

        attr.attack_toc = cv2.getTickCount()
        attr.attack_times.append(
            (attr.attack_toc - attack_tic) / cv2.getTickFrequency()
        )

        params.attributes.atk_latency.append(time.time() - t0)

        if args.config.attacker.eval:
            pert_deg = []
            pert_deg.append(torch.norm(pert_true, 1).data.cpu().numpy())
            pert_deg.append(torch.norm(pert_true, 2).data.cpu().numpy())
            pert_deg.append(
                torch.norm(pert_true, float("inf")).data.cpu().numpy()
            )
            attr.pert_degs.append(pert_deg)

        # start tracking
        torch.cuda.empty_cache()
        outputs = tracker.track(
            img,
            x_crop,
            immutable_dummy=dummy,
            is_perturbed=True,
        )
        # force the ground truth to be center
        # print('[debug] gt_bbox:', gt_bbox)
        # cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
        cx, cy, w, h = params.historical[-1].frame.aligned_bbox

        if np.isnan(cx) or np.isnan(cy) or np.isnan(w) or np.isnan(h):
            # log(f'[warn] skip at {idx}: gt_bbox = {params.historical[-1].frame.aligned_bbox}')
            ...
        elif w == 0 or h == 0:
            # SKIP: 2023-05022
            # this only happens on LaSOT dataset when object is blocked
            # this only happens due to SPARK reset tracker's size to gt
            # log(f'[warn] skip at {idx}: gt_bbox = {params.historical[-1].frame.aligned_bbox}')
            ...
        else:
            # log(f'[debug] skip: gt_bbox = {params.historical[-1].frame.aligned_bbox}')
            if dummy is not None:
                # dummy.center_pos = np.array([cx, cy]) # comment by original author
                dummy.size = np.array([w, h])
            else:
                # tracker.center_pos = np.array([cx, cy]) # comment by original author
                tracker.size = np.array([w, h])
        return outputs

    def spark_trk_frame_done(
        params: TestCBParamFrameDone[
            TestArgs, SiamRPNTracker_SPARK, Video, TestAttributes
        ]
    ):
        args = params.args
        attr = params.attributes
        img = params.img
        gt_bbox = params.gt_bbox
        video = params.video
        idx = params.idx
        pred_bbox = params.pred_bbox
        #
        attr.track_times.append(
            (cv2.getTickCount() - attr.attack_toc) / cv2.getTickFrequency()
        )
        attr.iter_nums.append(attacker.acc_iters)

        if (
            args.spark_visualize or args.config.attacker.save_video
        ) and idx > 0:
            gt_bbox = list(map(int, gt_bbox))
            pred_bbox = list(map(int, pred_bbox))
            cv2.rectangle(
                img,
                (gt_bbox[0], gt_bbox[1]),
                (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]),
                (0, 255, 0),
                3,
            )
            cv2.rectangle(
                img,
                (pred_bbox[0], pred_bbox[1]),
                (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]),
                (0, 255, 255),
                3,
            )
            if args.atk_type is AtkType.TA:
                cv2.rectangle(
                    img,
                    (
                        int(attr.target_traj[idx][0]) - 5,
                        int(attr.target_traj[idx][1]) - 5,
                    ),
                    (
                        int(attr.target_traj[idx][0]) + 5,
                        int(attr.target_traj[idx][1]) + 5,
                    ),
                    (255, 0, 0),
                    3,
                )
                pts = np.array(attr.target_traj, np.int32)
                pts = pts.reshape((-1, 1, 2))
                img = cv2.polylines(img, [pts], False, (255, 0, 0))
            cv2.putText(
                img,
                str(idx),
                (40, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )
            if args.config.attacker.save_video and attr.out is not None:
                attr.out.write(img)
            if args.spark_visualize:
                cv2.imshow(video.name, img)
                cv2.waitKey(1)

    def spark_trk_video_done(
        params: TestCBParamVideo[
            TestArgs, SiamRPNTracker_SPARK, Video, TestAttributes
        ]
    ):
        args = params.args
        attr = params.attributes
        v_idx = params.v_idx
        video = params.video
        #
        if args.config.attacker.save_video and attr.out is not None:
            attr.out.release()
        if args.config.attacker.eval and args.atk_type is AtkType.UA:
            # save random target_traj
            # traj_path = os.path.join(dataset_root, video.name + '_ua_traj.pkl')
            os.makedirs(
                os.path.join(
                    attr.attacked_root,
                    attr._dataset.name,
                ),
                exist_ok=True,
            )
            traj_path = os.path.join(
                attr.attacked_root,
                attr._dataset.name,
                video.name + "_ua_traj.pkl",
            )
            with open(traj_path, "wb") as f:
                pickle.dump(attacker.target_traj, f)

        with open(attr.pert_path, "w") as f:
            for x in attr.pert_degs:
                f.write(",".join([str(i) for i in x]) + "\n")
        with open(attr.iternum_path, "w") as f:
            for x in attr.iter_nums:
                f.write(str(x) + "\n")
        with open(attr.attack_time_path, "w") as f:
            f.write(",".join([str(i) for i in attr.attack_times]) + "\n")
        with open(attr.track_time_path, "w") as f:
            f.write(",".join([str(i) for i in attr.track_times]) + "\n")

        save_track_tracking_result_default(params)
        session_data = params.session_data
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
        else:
            Test.show_details(
                v_idx,
                video.name,
                attr.toc,
                len(video),
            )

    test.callback_init = spark_trk_init
    test.callback_track = spark_trk_track
    test.callback_frame_done = spark_trk_frame_done
    test.callback_video_done = spark_trk_video_done

    return test, attacker
