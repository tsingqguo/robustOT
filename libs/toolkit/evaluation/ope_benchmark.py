from typing import Optional, TypeVar, Union
import numpy as np
from toolkit.datasets.dataset import Dataset
from colorama import Style, Fore
from toolkit.datasets.lasot import LaSOTVideo
from toolkit.datasets.otb import OTBVideo
from toolkit.datasets.uav import UAVVideo

from toolkit.evaluation.benchmark import Benchmark

from ..utils import overlap_ratio, success_overlap, success_error


OPEV = TypeVar("OPEV", bound=Union[LaSOTVideo, OTBVideo, UAVVideo])


class OPEBenchmark(Benchmark[OPEV]):
    def __init__(self, dataset):
        self.dataset = dataset
        self.default_variants = "baseline"

    def convert_bb_to_center(self, bboxes):
        return np.array(
            [
                (bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2),
            ]
        ).T

    def convert_bb_to_norm_center(self, bboxes, gt_wh):
        return self.convert_bb_to_center(bboxes) / (gt_wh + 1e-16)

    def eval_success(
        self,
        eval_tracker: Optional[str] = None,
        eval_variant: Optional[str] = None,
    ):
        def get_results(tracker_name: str, variant: str):
            success_ret = {}
            for video in self.dataset:
                gt_traj = np.array(video.gt_traj)
                if tracker_name not in video.pred_trajs:
                    tracker_traj = video.load_tracker(
                        self.dataset.tracker_path, tracker_name, variant, False
                    )
                    tracker_traj = np.array(tracker_traj)
                else:
                    tracker_traj = np.array(video.pred_trajs[tracker_name])
                n_frame = len(gt_traj)
                if isinstance(video, LaSOTVideo):
                    gt_traj = gt_traj[video.absent == 1]
                    tracker_traj = tracker_traj[video.absent == 1]
                success_ret[video.name] = success_overlap(
                    gt_traj, tracker_traj, n_frame
                )
            return success_ret

        return self._eval(eval_tracker, eval_variant, get_results)

    def eval_precision(
        self,
        eval_tracker: Optional[str] = None,
        eval_variant: Optional[str] = None,
    ):
        def get_results(tracker_name: str, variant: str):
            precision_ret = {}
            for video in self.dataset:
                try:
                    gt_traj = np.array(video.gt_traj)
                    if tracker_name not in video.pred_trajs:
                        tracker_traj = video.load_tracker(
                            self.dataset.tracker_path, tracker_name, variant, False
                        )
                        tracker_traj = np.array(tracker_traj)
                    else:
                        tracker_traj = np.array(video.pred_trajs[tracker_name])
                    n_frame = len(gt_traj)
                    # if hasattr(video, "absent"):
                    if isinstance(video, LaSOTVideo):  # TODO: check this
                        gt_traj = gt_traj[video.absent == 1]
                        tracker_traj = tracker_traj[video.absent == 1]
                    gt_center = self.convert_bb_to_center(gt_traj)
                    tracker_center = self.convert_bb_to_center(tracker_traj)
                    thresholds = np.arange(0, 51, 1)
                    precision_ret[video.name] = success_error(
                        gt_center, tracker_center, thresholds, n_frame
                    )
                except Exception as e:
                    print(f'[ERR.Prec] err on video {video}')
            return precision_ret

        return self._eval(eval_tracker, eval_variant, get_results)

    def eval_norm_precision(
        self,
        eval_tracker: Optional[str] = None,
        eval_variant: Optional[str] = None,
    ):
        def get_results(tracker_name: str, variant: str):
            norm_precision_ret = {}
            for video in self.dataset:
                gt_traj = np.array(video.gt_traj)
                if tracker_name not in video.pred_trajs:
                    tracker_traj = video.load_tracker(
                        self.dataset.tracker_path, tracker_name, variant, False
                    )
                    tracker_traj = np.array(tracker_traj)
                else:
                    tracker_traj = np.array(video.pred_trajs[tracker_name])
                n_frame = len(gt_traj)
                # if hasattr(video, "absent"):
                if isinstance(video, LaSOTVideo):  # TODO: check this
                    gt_traj = gt_traj[video.absent == 1]
                    tracker_traj = tracker_traj[video.absent == 1]
                gt_center_norm = self.convert_bb_to_norm_center(
                    gt_traj, gt_traj[:, 2:4]
                )
                tracker_center_norm = self.convert_bb_to_norm_center(
                    tracker_traj, gt_traj[:, 2:4]
                )
                thresholds = np.arange(0, 51, 1) / 100
                norm_precision_ret[video.name] = success_error(
                    gt_center_norm, tracker_center_norm, thresholds, n_frame
                )
            return norm_precision_ret

        return self._eval(eval_tracker, eval_variant, get_results)

    def show_result(
        self,
        success_ret,
        precision_ret=None,
        norm_precision_ret=None,
        show_video_level: bool = False,
        highlight_threshold: float = 0.6,
    ):
        results = success_ret
        headers = ["Tracker Name", "Success", "Norm Precision", "Precision"]
        data = []

        """pretty print result
        Args:
            result: returned dict from function eval
        """
        # sort tracker
        tracker_auc = {}
        for tracker_name in success_ret.keys():
            auc = np.mean(list(success_ret[tracker_name].values()))
            tracker_auc[tracker_name] = auc
        tracker_auc_ = sorted(
            tracker_auc.items(), key=lambda x: x[1], reverse=True
        )[:20]
        tracker_names = [x[0] for x in tracker_auc_]

        print(success_ret.keys())
        tracker_name_len = max(
            (max([len(x) for x in success_ret.keys()]) + 2), 12
        )
        header = (
            "|{:^" + str(tracker_name_len) + "}|{:^9}|{:^16}|{:^11}|"
        ).format("Tracker name", "Success", "Norm Precision", "Precision")
        formatter = (
            "|{:^" + str(tracker_name_len) + "}|{:^9.3f}|{:^16.3f}|{:^11.3f}|"
        )
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for tracker_name in tracker_names:
            # success = np.mean(list(success_ret[tracker_name].values()))
            success = tracker_auc[tracker_name]
            if precision_ret is not None:
                precision = np.mean(
                    list(precision_ret[tracker_name].values()), axis=0
                )[20]
            else:
                precision = 0
            if norm_precision_ret is not None:
                norm_precision = np.mean(
                    list(norm_precision_ret[tracker_name].values()), axis=0
                )[20]
            else:
                norm_precision = 0
            print(
                formatter.format(
                    tracker_name, success, norm_precision, precision
                )
            )
        print("-" * len(header))

        if (
            show_video_level
            and len(success_ret) < 10
            and precision_ret is not None
            and len(precision_ret) < 10
        ):
            print("\n\n")
            header1 = "|{:^21}|".format("Tracker name")
            header2 = "|{:^21}|".format("Video name")
            for tracker_name in success_ret.keys():
                # col_len = max(20, len(tracker_name))
                header1 += ("{:^21}|").format(tracker_name)
                header2 += "{:^9}|{:^11}|".format("success", "precision")
            print("-" * len(header1))
            print(header1)
            print("-" * len(header1))
            print(header2)
            print("-" * len(header1))
            videos = list(success_ret[tracker_name].keys())
            for video in videos:
                row = "|{:^21}|".format(video)
                for tracker_name in success_ret.keys():
                    success = np.mean(success_ret[tracker_name][video])
                    precision = np.mean(precision_ret[tracker_name][video])
                    success_str = "{:^9.3f}".format(success)
                    if success < highlight_threshold:
                        row += f"{Fore.RED}{success_str}{Style.RESET_ALL}|"
                    else:
                        row += success_str + "|"
                    precision_str = "{:^11.3f}".format(precision)
                    if precision < highlight_threshold:
                        row += f"{Fore.RED}{precision_str}{Style.RESET_ALL}|"
                    else:
                        row += precision_str + "|"
                print(row)
            print("-" * len(header1))
