import os
import numpy as np
import torch

from attackers.iou.tracker import IoU_SiamRPNTracker
from attackers.iou.utils import (
    forward_perturbation,
    forward_perturbation_torch,
    get_diff,
    get_diff_torch,
    orthogonal_perturbation,
    orthogonal_perturbation_torch,
    overlap_ratio,
)
from pyotp.config.pysot import CFG
from pyotp.tools.test import (
    TestAttributes as _TestAttributes,
    TestCBParamFrame,
    TestCBParamFrameInit,
    args_parse,
    initial_testing_args as _initial_testing_args,
    Test,
    TestArgs as _TestArgs,
    TestCliArgs as _TestCliArgs,
    V,
)
from pyotp.utils import mat2tensor, tensor2mat
from pysot.models.model_builder_cfgmod import ModelBuilder
from pysot.tracker.siamrpn_tracker_cfgmod import SiamRPNTracker
from toolkit.datasets.video import Video
from typing import (
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
)


class TestCliArgs(_TestCliArgs):
    ...


T = TypeVar("T", bound=IoU_SiamRPNTracker)  # TODO:


class TestArgs(_TestArgs):
    ...


class TestAttributes(_TestAttributes[TestArgs, V]):
    iou_tracker: IoU_SiamRPNTracker
    perturb_max: int
    save_bbox: List
    l2_norms: List
    last_preturb: np.ndarray
    last_preturb_tensor: torch.Tensor

    def __init__(self, args, video, dataset, iou_tracker: IoU_SiamRPNTracker):
        super().__init__(args, video, dataset, is_vot_st=None)

        self.iou_tracker = iou_tracker
        self.perturb_max = 10000
        self.l2_norms = []
        # self.last_preturb = None # initialized in _init_tracker


def initial_testing_args(
    cliargs: TestCliArgs, argv: List[str], args: TestArgs
) -> None:
    _initial_testing_args(cliargs, argv, args)


def _initial_iou_tracker(test: Test) -> IoU_SiamRPNTracker:
    raw_tracker = test.initial_tracker()

    tracker = IoU_SiamRPNTracker(raw_tracker, CFG)  # TODO: check CFG here

    def iou_trk_init(
        params: TestCBParamFrameInit[
            TestArgs, SiamRPNTracker, Video, TestAttributes
        ],
        dummy: Optional[T],
    ):
        # copied from test.Test.__init__.trk_init
        tracker = params.raw_tracker
        img = params.img
        gt_bbox_ = params.gt_bbox_
        #
        tracker.init(img, gt_bbox_, dummy)
        # copied from test.Test.__init__.trk_init

        params.attributes.save_bbox = gt_bbox_
        params.attributes.last_preturb = np.zeros_like(img)
        params.attributes.last_preturb_tensor = torch.zeros(
            3, *img.shape[:2]
        ).to(tracker.model.parameters().__next__().device)

    def iou_trk_track(
        params: TestCBParamFrame[
            TestArgs, SiamRPNTracker, Video, TestAttributes
        ],
        dummy: Optional[T],
        extra_options: Optional[dict] = None,
    ):
        iou_tracker = params.attributes.iou_tracker
        perturb_max = params.attributes.perturb_max

        # black-box IoU attack
        image = params.img
        last_gt = params.attributes.save_bbox
        l2_normes = params.attributes.l2_norms
        last_preturb = params.attributes.last_preturb

        if extra_options is not None:
            if extra_options.pop("keep_clean", False):
                return iou_tracker.raw.track(image, dummy)

        heavy_noise = (
            np.random.randint(
                -1,
                2,
                (image.shape[0], image.shape[1], image.shape[2]),
            )
            * 128
        )
        image_noise = image + heavy_noise
        image_noise = np.clip(image_noise, 0, 255)

        noise_sample = image_noise - 128
        clean_sample_init = image.astype(float) - 128
        image_noise = image_noise.astype(np.uint8)
        # query
        outputs_orig = iou_tracker.track_fixed(image)
        outputs_target = iou_tracker.track_fixed(image_noise)
        target_score = overlap_ratio(
            np.array(outputs_orig["bbox"]),
            np.array(outputs_target["bbox"]),
        )
        adversarial_sample = image.astype(float) - 128

        import time

        t0 = time.time()

        if target_score < 0.8:
            # parameters
            n_steps = 0
            epsilon = 0.05
            delta = 0.05
            weight = 0.5
            para_rate = 0.9
            # Move a small step
            while True:
                # Initialize with previous perturbations
                clean_sample = clean_sample_init + weight * last_preturb
                trial_sample = clean_sample + forward_perturbation(
                    epsilon * get_diff(clean_sample, noise_sample),
                    adversarial_sample,
                    noise_sample,
                )
                trial_sample = np.clip(trial_sample, -128, 127)
                outputs_adv = iou_tracker.track_fixed(
                    (trial_sample + 128).astype(np.uint8)
                )
                # IoU score
                threshold_1 = overlap_ratio(
                    np.array(outputs_orig["bbox"]),
                    np.array(outputs_adv["bbox"]),
                )
                threshold_2 = overlap_ratio(
                    np.array(last_gt),
                    np.array(outputs_adv["bbox"]),
                )
                threshold = (
                    para_rate * threshold_1 + (1 - para_rate) * threshold_2
                )
                adversarial_sample = trial_sample
                break

            while True:
                # Tangential direction
                d_step = 0
                while True:
                    d_step += 1
                    trial_samples = []
                    score_sum = []
                    for i in np.arange(10):
                        trial_sample = (
                            adversarial_sample
                            + orthogonal_perturbation(
                                delta,
                                adversarial_sample,
                                noise_sample,
                            )
                        )
                        trial_sample = np.clip(trial_sample, -128, 127)
                        # query
                        outputs_adv = iou_tracker.track_fixed(
                            (trial_sample + 128).astype(np.uint8)
                        )
                        # IoU score
                        score_1 = overlap_ratio(
                            np.array(outputs_orig["bbox"]),
                            np.array(outputs_adv["bbox"]),
                        )
                        score_2 = overlap_ratio(
                            np.array(last_gt),
                            np.array(outputs_adv["bbox"]),
                        )
                        score = para_rate * score_1 + (1 - para_rate) * score_2
                        score_sum = np.hstack((score_sum, score))
                        trial_samples.append(trial_sample)
                    _t = time.time()
                    d_score = np.mean(score_sum <= threshold)
                    if d_score > 0.0:
                        if d_score < 0.3:
                            delta /= 0.9
                        elif d_score > 0.7:
                            delta *= 0.9
                        adversarial_sample = np.array(trial_samples)[
                            np.argmin(np.array(score_sum))
                        ]
                        threshold = score_sum[np.argmin(np.array(score_sum))]
                        break
                    elif d_step >= 5 or delta > 0.3:
                        break
                    else:
                        delta /= 0.9
                # Normal direction
                e_step = 0
                while True:
                    e_step += 1
                    trial_sample = adversarial_sample + forward_perturbation(
                        epsilon * get_diff(adversarial_sample, noise_sample),
                        adversarial_sample,
                        noise_sample,
                    )
                    trial_sample = np.clip(trial_sample, -128, 127)
                    # query
                    outputs_adv = iou_tracker.track_fixed(
                        (trial_sample + 128).astype(np.uint8)
                    )
                    l2_norm = np.mean(
                        get_diff(clean_sample_init, trial_sample)
                    )
                    # IoU score
                    threshold_1 = overlap_ratio(
                        np.array(outputs_orig["bbox"]),
                        np.array(outputs_adv["bbox"]),
                    )
                    threshold_2 = overlap_ratio(
                        np.array(last_gt),
                        np.array(outputs_adv["bbox"]),
                    )
                    threshold_sum = (
                        para_rate * threshold_1 + (1 - para_rate) * threshold_2
                    )

                    if threshold_sum <= threshold:
                        adversarial_sample = trial_sample
                        epsilon *= 0.9
                        threshold = threshold_sum
                        break
                    elif e_step >= 30 or l2_norm > perturb_max:
                        break
                    else:
                        epsilon /= 0.9
                n_steps += 1

                if threshold <= target_score or l2_norm > perturb_max:
                    adversarial_sample = np.clip(adversarial_sample, -128, 127)
                    l2_norm = np.mean(
                        get_diff(clean_sample_init, adversarial_sample)
                    )
                    l2_normes.append(l2_norm)
                    break

            params.attributes.last_preturb = adversarial_sample - clean_sample
            img = (adversarial_sample + 128).astype(np.uint8)
        else:
            adversarial_sample = image + last_preturb
            adversarial_sample = np.clip(adversarial_sample, 0, 255)
            img = adversarial_sample.astype(np.uint8)

        params.attributes.atk_latency.append(time.time() - t0)

        tracker = params.raw_tracker
        outputs = tracker.track(img, dummy)
        return outputs

    test.callback_init = iou_trk_init
    test.callback_track = iou_trk_track  # type: ignore

    return tracker


def initial_iou_tester(
    args: TestArgs,
) -> Tuple[
    Test[SiamRPNTracker, TestArgs, Video, TestAttributes], IoU_SiamRPNTracker
]:
    test = Test(
        args,
        test_attr_cls=TestAttributes,
    )
    tracker = _initial_iou_tracker(test)
    return test, tracker
