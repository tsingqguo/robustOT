import os
from glob import glob
from itertools import product
from multiprocessing import Pool
from pyotp import ENV
from toolkit.datasets import (
    OTBDataset,
    UAVDataset,
    LaSOTDataset,
    VOTDataset,
    NFSDataset,
    VOTLTDataset,
)
from toolkit.evaluation import (
    OPEBenchmark,
    AccuracyRobustnessBenchmark,
    EAOBenchmark,
    F1Benchmark,
)
from tqdm import tqdm
from typed_cap import Cap
from typing import List, Literal, Set, TypedDict


class EvalCliArgs:
    # @alias=d
    dataset: Literal[
        "VOT2018", "VOT2019", "OTB100", "UAV123", "LaSOT", "NFS30", "NFS240"
    ]
    """dataset name"""

    # @alias=n
    num_thread: int = 1
    """number of thread to eval"""

    # @alias=s
    show_video_level: bool = False

    # @alias=p
    tracker_path: str
    """tracker result path"""

    # @alias=t
    tracker_filter: str
    """filter for tackers' name"""

    # @alias=v
    variant_filter: str
    """filter for variant' name (e.g. baseline"""


def handle_cliargs() -> EvalCliArgs:
    cap = Cap(EvalCliArgs)
    return cap.parse().val


def main(args: EvalCliArgs):
    tracker_dir = os.path.join(args.tracker_path, args.dataset)
    trackers = glob(
        os.path.join(args.tracker_path, args.dataset, args.tracker_filter)
    )
    trackers = [os.path.basename(x) for x in trackers]
    _tracker_list = [f"\n\t- {t_name}" for t_name in trackers]
    print(f'filtered trackers:{"".join(_tracker_list)}')

    variants_set: Set[str] = set()
    for tracker in trackers:
        variants_set.update(
            [
                os.path.basename(v_path)
                for v_path in glob(
                    os.path.join(
                        args.tracker_path,
                        args.dataset,
                        tracker,
                        args.variant_filter,
                    )
                )
            ]
        )
    _variants_list = [f"\n\t- {v_name}" for v_name in variants_set]
    print(f"filtered variants:{''.join(_variants_list)}")
    variants: List[str] = list(variants_set)

    assert len(trackers) > 0
    args.num_thread = min(args.num_thread, len(trackers))

    dataset_roots = {
        "UAV": os.path.join(
            ENV.dset_root_testing,
            "UAV123",
            "data_seq",
            "UAV123",
        ),
        "UAV123": os.path.join(
            ENV.dset_root_testing,
            "UAV123",
            "data_seq",
            "UAV123",
        ),
        "LaSOT": os.path.join(
            ENV.dset_root_testing,
            "lasot",
            "LaSOTTesting",
        ),
        "NFS30": os.path.join(
            ENV.dset_root_testing,
            "NFS",
        ),
        "NFS240": os.path.join(
            ENV.dset_root_testing,
            "NFS",
        ),
    }  # TODO:

    dataset_root = dataset_roots.get(args.dataset, None)
    if dataset_root is None:
        dataset_root = os.path.join(ENV.dset_root_testing, args.dataset)

    if "OTB" in args.dataset:
        dataset = OTBDataset(args.dataset, dataset_root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num_thread) as pool:
            for ret in tqdm(
                pool.starmap(
                    benchmark.eval_success, product(trackers, variants)
                ),
                desc="eval success",
                total=len(trackers),
            ):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num_thread) as pool:
            for ret in tqdm(
                pool.starmap(
                    benchmark.eval_precision, product(trackers, variants)
                ),
                desc="eval precision",
                total=len(trackers),
            ):
                # print(f'[DBG.M] ret: {ret.keys()}')
                precision_ret.update(ret)
        # print(f'[DBG.M] precision_ret.keys(): {precision_ret.keys()}')
        benchmark.show_result(
            success_ret,
            precision_ret,
            show_video_level=args.show_video_level,
        )
    elif "UAV" in args.dataset:
        dataset = UAVDataset(args.dataset, dataset_root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num_thread) as pool:
            for ret in tqdm(
                # pool.imap_unordered(benchmark.eval_success, trackers),
                pool.starmap(
                    benchmark.eval_success, product(trackers, variants)
                ),
                desc="eval success",
                total=len(trackers),
            ):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num_thread) as pool:
            for ret in tqdm(
                # pool.imap_unordered(benchmark.eval_precision, trackers),
                pool.starmap(
                    benchmark.eval_precision, product(trackers, variants)
                ),
                desc="eval precision",
                total=len(trackers),
            ):
                precision_ret.update(ret)
        benchmark.show_result(
            success_ret,
            precision_ret,
            show_video_level=args.show_video_level,
        )
    elif "LaSOT" == args.dataset:
        dataset = LaSOTDataset(args.dataset, dataset_root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num_thread) as pool:
            for ret in tqdm(
                pool.starmap(
                    benchmark.eval_success, product(trackers, variants)
                ),
                desc="eval success",
                total=len(trackers),
            ):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num_thread) as pool:
            for ret in tqdm(
                pool.starmap(
                    benchmark.eval_precision, product(trackers, variants)
                ),
                desc="eval precision",
                total=len(trackers),
            ):
                precision_ret.update(ret)
        norm_precision_ret = {}
        with Pool(processes=args.num_thread) as pool:
            for ret in tqdm(
                pool.starmap(
                    benchmark.eval_norm_precision, product(trackers, variants)
                ),
                desc="eval norm precision",
                total=len(trackers),
            ):
                norm_precision_ret.update(ret)
        benchmark.show_result(
            success_ret,
            precision_ret,
            norm_precision_ret,
            show_video_level=args.show_video_level,
        )
    elif "NFS" in args.dataset:
        dataset = NFSDataset(args.dataset, dataset_root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num_thread) as pool:
            for ret in tqdm(
                # pool.imap_unordered(benchmark.eval_success, trackers),
                # desc="eval success",
                # total=len(trackers),
                pool.starmap(
                    benchmark.eval_success, product(trackers, variants)
                ),
                desc="eval success",
                total=len(trackers),
            ):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num_thread) as pool:
            for ret in tqdm(
                # pool.imap_unordered(benchmark.eval_precision, trackers),
                # desc="eval precision",
                # total=len(trackers),
                pool.starmap(
                    benchmark.eval_precision, product(trackers, variants)
                ),
                desc="eval precision",
                total=len(trackers),
            ):
                precision_ret.update(ret)
        benchmark.show_result(
            success_ret,
            precision_ret,
            show_video_level=args.show_video_level,
        )
    elif args.dataset in ["VOT2016", "VOT2017", "VOT2018", "VOT2019"]:
        dataset = VOTDataset(args.dataset, dataset_root)
        dataset.set_tracker(tracker_dir, trackers)
        ar_benchmark = AccuracyRobustnessBenchmark(dataset)
        ar_result = {}
        with Pool(processes=args.num_thread) as pool:
            for ret in tqdm(
                pool.starmap(ar_benchmark.eval, product(trackers, variants)),
                desc="eval ar",
                total=len(trackers),
            ):
                ar_result.update(ret)
        benchmark = EAOBenchmark(dataset)
        eao_result = {}
        with Pool(processes=args.num_thread) as pool:
            for ret in tqdm(
                pool.starmap(benchmark.eval, product(trackers, variants)),
                desc="eval eao",
                total=len(trackers),
            ):
                eao_result.update(ret)
        ar_benchmark.show_result(
            ar_result, eao_result, show_video_level=args.show_video_level
        )
    elif "VOT2018-LT" == args.dataset:
        dataset = VOTLTDataset(args.dataset, dataset_root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = F1Benchmark(dataset)
        f1_result = {}
        with Pool(processes=args.num_thread) as pool:
            for ret in tqdm(
                pool.imap_unordered(benchmark.eval, trackers),
                desc="eval f1",
                total=len(trackers),
            ):
                f1_result.update(ret)
        benchmark.show_result(
            f1_result, show_video_level=args.show_video_level
        )


if __name__ == "__main__":
    args = handle_cliargs()
    main(args)
