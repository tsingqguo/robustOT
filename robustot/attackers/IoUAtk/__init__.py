import time
from dataclasses import dataclass, field

from line_profiler import profile
import numpy as np
import numpy.typing as npt

from msot.tools.test.utils.process import (
    Allow,
    Input,
    ProcessTemplate,
    Processor,
    ProceesorAttrs,
    ProceesorConfig,
)
from msot.trackers.base import TrackResult  # TODO:
from msot.utils.dataship import DataCTR as DC

from .utils import forward_perturbation, get_diff, orthogonal_perturbation


@dataclass
class IoUAtkConfig(ProceesorConfig):
    # timer_type = TimerType.CUDA
    allow_access = Allow.TARGET | Allow.TRACKER | Allow.HISTORICAL
    perturb_max: int = 10_000


@dataclass
class IoUAtkAttrs(ProceesorAttrs):
    l2_norms: DC[list[float]] = field(
        default_factory=lambda: DC(
            is_shared=False, is_mutable=True, allow_unbound=True
        )
    )
    last_perturb: DC[np.ndarray] = field(
        default_factory=lambda: DC(
            is_shared=False, is_mutable=True, allow_unbound=True
        )
    )

    @property
    def valid_names(self) -> set[str]:
        return super().valid_names | {"l2_norms", "last_perturb"}


class IoUAtkProcessor(Processor[IoUAtkAttrs, IoUAtkConfig]):
    def __init__(
        self,
        name: str,
        config: IoUAtkConfig,
    ) -> None:
        super().__init__(name, config, IoUAtkAttrs)

    def fork_track(self, image: npt.NDArray[np.uint8]) -> TrackResult:
        _crop, result = self.tracker.fork().track(image)
        return result

    @profile
    def process(self, input: Input) -> Input | None:
        if not isinstance(input, np.ndarray):
            raise RuntimeError("IoUAtkProcessor only accepts ndarray input")

        if type(self.process_target) is ProcessTemplate:
            return None

        # TODO:
        last_pred = self.historical.last.unwrap().result.pred.get()

        heavy_noise = (
            np.random.randint(
                -1,
                2,
                (input.shape[0], input.shape[1], input.shape[2]),
            )
            * 128
        )
        input_noise = input + heavy_noise
        input_noise = np.clip(input_noise, 0, 255)

        noise_sample = input_noise - 128
        clean_sample_init = input.astype(float) - 128
        input_noise = input_noise.astype(np.uint8)

        # query
        outputs_orig = self.fork_track(input).output
        output_target = self.fork_track(input_noise).output
        target_score = outputs_orig.get_overlap_ratio(output_target)
        adversarial_sample = input.astype(float) - 128

        if target_score < 0.8:
            # parameters
            n_steps = 0
            epsilon = 0.05
            delta = 0.05
            weight = 0.5
            para_rate = 0.9
            # Move a small step
            while True:
                # print(f'[debug] exec loop A ({time.time() - t0:.2f}s after)')
                # Initialize with previous perturbations
                clean_sample = (
                    clean_sample_init
                    + weight
                    * self.attrs.last_perturb.get(default=np.zeros_like(input))
                )
                trial_sample = clean_sample + forward_perturbation(
                    epsilon * get_diff(clean_sample, noise_sample),
                    adversarial_sample,
                    noise_sample,
                )
                trial_sample = np.clip(trial_sample, -128, 127)
                outputs_adv = self.fork_track(
                    (trial_sample + 128).astype(np.uint8)
                ).output

                # IoU score
                threshold_1 = outputs_orig.get_overlap_ratio(outputs_adv)
                threshold_2 = last_pred.get_overlap_ratio(outputs_adv)
                threshold = (
                    para_rate * threshold_1 + (1 - para_rate) * threshold_2
                )
                adversarial_sample = trial_sample
                break

            while True:
                # print(f'\t\texec loop B ({time.time() - t0:.2f}s after)')
                # Tangential direction
                d_step = 0
                while True:
                    # print(f'\t\t\texec loop C ({time.time() - t0:.2f}s after)')
                    d_step += 1
                    # print("\t#{}".format(d_step))
                    trial_samples = []
                    score_sum = []
                    for i in np.arange(10):
                        # print(f'\t\t\t\tC.mini_loop[{i}] ({time.time() - t0:.2f}s after)')
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
                        outputs_adv = self.fork_track(
                            (trial_sample + 128).astype(np.uint8)
                        ).output
                        # IoU score
                        score_1 = outputs_orig.get_overlap_ratio(outputs_adv)
                        score_2 = last_pred.get_overlap_ratio(outputs_adv)
                        score = para_rate * score_1 + (1 - para_rate) * score_2
                        score_sum = np.hstack((score_sum, score))
                        trial_samples.append(trial_sample)
                    # print('\t\t\t\tdo following shit takes', end=None)
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
                    # print(f'\t\t\texec loop D ({time.time() - t0:.2f}s after)')
                    e_step += 1  # FIXME: this is original not exist
                    trial_sample = adversarial_sample + forward_perturbation(
                        epsilon * get_diff(adversarial_sample, noise_sample),
                        adversarial_sample,
                        noise_sample,
                    )
                    trial_sample = np.clip(trial_sample, -128, 127)
                    # query
                    outputs_adv = self.fork_track(
                        (trial_sample + 128).astype(np.uint8)
                    ).output
                    l2_norm = np.mean(
                        get_diff(clean_sample_init, trial_sample)
                    )
                    # IoU score
                    threshold_1 = outputs_orig.get_overlap_ratio(outputs_adv)
                    threshold_2 = last_pred.get_overlap_ratio(outputs_adv)
                    threshold_sum = (
                        para_rate * threshold_1 + (1 - para_rate) * threshold_2
                    )

                    if threshold_sum <= threshold:
                        adversarial_sample = trial_sample
                        epsilon *= 0.9
                        threshold = threshold_sum
                        break
                    elif e_step >= 30 or l2_norm > self.config.perturb_max:
                        break
                    else:
                        epsilon /= 0.9
                n_steps += 1

                if (
                    threshold <= target_score
                    or l2_norm > self.config.perturb_max
                ):
                    adversarial_sample = np.clip(adversarial_sample, -128, 127)
                    l2_norm = np.mean(
                        get_diff(clean_sample_init, adversarial_sample)
                    )
                    self.attrs.l2_norms.get(default=[]).append(float(l2_norm))
                    break

            self.attrs.last_perturb.update(adversarial_sample - clean_sample)
            img = (adversarial_sample + 128).astype(np.uint8)
        else:
            adversarial_sample = input + self.attrs.last_perturb.get(
                default=np.zeros_like(input)
            )
            adversarial_sample = np.clip(adversarial_sample, 0, 255)
            img = adversarial_sample.astype(np.uint8)

        return img
