import itertools
import numpy as np
from pyotp.utils import silent_nanmean, to_green, to_red
from toolkit.datasets import VOTDataset
from toolkit.datasets.video import Video
from toolkit.datasets.dataset import Dataset
from toolkit.evaluation.benchmark import Benchmark
from toolkit.utils import calculate_failures, calculate_accuracy
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    TypedDict,
    Union,
)


ListNum = List[Union[int, float]]
VideoName = str


class BenchmarkResult(TypedDict):
    overlaps: Dict[VideoName, ListNum]
    failures: Dict[VideoName, ListNum]


class AccuracyRobustnessBenchmark(Benchmark):
    burnin: int
    # dataset: Dataset
    dataset: VOTDataset # TODO:

    def __init__(self, dataset: VOTDataset, burnin: int = 10):
        self.dataset = dataset
        self.default_variants = "baseline"
        self.burnin = burnin

    def eval(
        self,
        eval_tracker: Optional[str] = None,
        eval_variant: Optional[str] = None,
    ) -> Dict[str, BenchmarkResult]:
        def get_result(
            tracker_name: str,
            variant: str,
        ) -> BenchmarkResult:
            accuracy, failures = self._calculate_accuracy_robustness(
                tracker_name, variant
            )
            return {"overlaps": accuracy, "failures": failures}

        return self._eval(eval_tracker, eval_variant, get_result)

    def show_result(
        self,
        result: Dict[str, BenchmarkResult],
        eao_result=None,
        show_video_level: bool = False,
        highlight_threshold: float = 0.5,
    ) -> None:
        results = result
        headers = ["Tracker Name", "Accuracy", "Robustness", "Lost Number"]
        data = []
        for t_name, res in results.items():
            overlaps = list(itertools.chain(*res["overlaps"].values()))
            accuracy = silent_nanmean(overlaps)
            length = sum([len(x) for x in res["overlaps"].values()])
            failures = list(res["failures"].values())
            lost_num = np.mean(np.sum(failures, axis=0))
            robustness = (
                np.mean(np.sum(np.array(failures), axis=0) / length) * 100
            )
            data.append([t_name, accuracy, robustness, lost_num])
        if eao_result is not None:
            headers.append("EAO")
            for r_i, t_name in enumerate(results.keys()):
                print(f'{t_name}: {eao_result[t_name].keys()}')
                data[r_i].append(eao_result[t_name]["all"])

        self._print_result(
            headers, data, fmt={"Lost Number": lambda x: f"{float(x):.1f}"}
        )

        if show_video_level and len(result) < 10:
            v_headers = ["Video name"]
            v_data = []
            for _ in results.keys():
                v_headers = [*v_headers, "Acc", "LN"]
            videos = list(list(results.values())[0]["overlaps"].keys())
            for v_name in videos:
                v_row_data = [v_name]
                for t_name, res in results.items():
                    overlaps = res["overlaps"][v_name]
                    accuracy = silent_nanmean(overlaps)
                    failures = res["failures"][v_name]
                    lost_num = np.mean(failures)
                    v_row_data = [*v_row_data, accuracy, lost_num]
                v_data.append(v_row_data)

            def fmt_acc(x: float) -> str:
                num = f"{x:.3f}"
                if x >= highlight_threshold:
                    num = to_green(num)
                return num

            def fmt_lost_num(x: float) -> str:
                num = f"{x:0.0f}"
                if x > 0:
                    num = to_red(num)
                return num

            print("\n")
            self._print_result(
                v_headers,
                v_data,
                fmt={
                    "Acc": fmt_acc,
                    "LN": fmt_lost_num,
                },
                arbitrary_fmt=[],
                print_args={"alignment": "l" + "r" * (len(v_headers) - 1)},
            )

    def _calculate_accuracy_robustness(
        self, tracker_name: str, variant: str
    ) -> Tuple[Dict[VideoName, ListNum], Dict[VideoName, ListNum]]:
        overlaps: Dict[VideoName, ListNum] = {}
        failures: Dict[VideoName, ListNum] = {}
        for i in range(len(self.dataset)):
            video = self.dataset[i]
            gt_traj = video.gt_traj
            if tracker_name not in video.pred_trajs:
                tracker_trajs = video.load_tracker(
                    self.dataset.tracker_path,
                    tracker_name,
                    variant=variant,
                    store=False,
                )
            else:
                tracker_trajs = video.pred_trajs[tracker_name]
            overlaps_group = []
            num_failures_group = []
            for tracker_traj in tracker_trajs:
                num_failures = calculate_failures(tracker_traj)[0]
                overlaps_ = calculate_accuracy(
                    tracker_traj,
                    gt_traj,
                    burnin=10,
                    bound=(video.width, video.height),
                )[1]
                overlaps_group.append(overlaps_)
                num_failures_group.append(num_failures)

            overlaps[video.name] = silent_nanmean(
                overlaps_group, axis=0
            ).tolist()
            failures[video.name] = num_failures_group
        return overlaps, failures
