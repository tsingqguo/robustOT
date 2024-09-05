import numpy as np
import torch
from attackers.rtaa.utils import where
from pyotp.config.pysot import CFG
from pyotp.tools.test import (
    TestAttributes as _TestAttributes,
    TestCBParamFrame,
    args_parse,
    initial_testing_args as _initial_testing_args,
    Test,
    TestArgs as _TestArgs,
    TestCliArgs as _TestCliArgs,
    V,
)
from pyotp.utils import mat2tensor, tensor2mat
from pysot.datasets.anchor_target_cfgmod import AnchorTarget
from pysot.models.model_builder_cfgmod import ModelBuilder
from pysot.tracker.siamrpn_tracker_cfgmod import SiamRPNTracker
from pysot.utils.bbox import Corner, center2corner
from toolkit.datasets.dataset import Dataset
from toolkit.datasets.video import Video
from typing import (
    Optional,
    Tuple,
    TypeVar,
)


class TestCliArgs(_TestCliArgs):
    iteration: int = 10
    eps: int = 10


T = TypeVar("T", bound=SiamRPNTracker)  # TODO:


class TestArgs(_TestArgs):
    iteration: int
    eps: int


class TestAttributes(_TestAttributes[TestArgs, V]):
    eps: int
    alpha: int
    iteration: int
    prev_atk: Optional[np.ndarray]

    anchor_target: AnchorTarget
    _size_overwrite: int
    """only for alexnet (buggy)"""

    def __init__(self, args: TestArgs, video: V, dataset: Dataset[V]) -> None:
        super().__init__(args, video, dataset)

        self.eps = args.eps
        self.alpha = 1
        self.iteration = args.iteration

        if CFG.track.instance_size == 287:
            self._size_overwrite = 21
        else:
            self._size_overwrite = CFG.train.output_size

        self.anchor_target = AnchorTarget(
            im_c_overwrite=CFG.track.instance_size,
            size_overwrite=self._size_overwrite,
        )
        self.prev_atk = None


def initial_testing_args(
    cliargs: TestCliArgs, argv: list[str], args: TestArgs
) -> None:
    _initial_testing_args(cliargs, argv, args)
    args.iteration = cliargs.iteration
    args.eps = cliargs.eps
    args.model_name = "_".join([args.model_name, f"iter={args.iteration}"])


def _initial_rtaa_tracker(
    test: Test[SiamRPNTracker, TestArgs, V, TestAttributes]
):
    raw_tracker = test.initial_tracker()

    tracker = SiamRPNTracker(raw_tracker.model)

    def trk_track(
        params: TestCBParamFrame[
            TestArgs, SiamRPNTracker, Video, TestAttributes
        ],
        dummy: Optional[T],
        extra_options: Optional[dict] = None,
    ):
        attr = params.attributes
        tracker = params.raw_tracker
        img = params.img
        his = params.historical
        idx = params.idx

        x_crop = tracker.get_x_crop(img, dummy)

        x_crop, scale = tracker.get_x_crop(img, dummy)

        if idx % 30 == 0:
            attr.prev_atk = None

        import time

        t0 = time.time()
        # attack start
        prev_atk = attr.prev_atk
        eps = attr.eps
        anchor_target = attr.anchor_target
        if prev_atk is not None:
            if prev_atk.ndim == 3:
                prev_atk = np.resize(prev_atk, (1, *x_crop.shape))
            else:
                prev_atk = np.resize(prev_atk, x_crop.shape)
            prev_atk = torch.from_numpy(prev_atk).to(x_crop.device)
            x_crop_init = x_crop + prev_atk
        else:
            x_crop_init = x_crop
        x_crop_init = torch.clamp(x_crop_init, 0, 255)

        x = x_crop.detach().clone().requires_grad_(True)
        x_adv = x_crop_init.detach().clone().requires_grad_(True)

        if his.last.is_some():
            last = his.last.unwrap()
            bbox = last.frame.bbox_pred
        else:
            bbox = None

        if bbox is not None:
            x1, y1, w, h = bbox
            corner = Corner(x1, y1, x1 + w, y1 + h)
        else:
            x1, y1, x2, y2 = center2corner(his.cur.frame.aligned_bbox)
            corner = Corner(x1, y1, x2, y2)

        gt_cls, gt_delta, gt_delta_weight, _ = anchor_target(
            corner,
            attr._size_overwrite,
            neg=False,
        )
        alpha = eps * 1.0 / attr.iteration
        # print("")
        for i in range(attr.iteration):
            out = tracker.model.forward(
                {
                    "template": his[0].tracking.unwrap().z_crop,
                    "search": x_adv,
                    "label_cls": torch.from_numpy(gt_cls).unsqueeze(0),
                    "label_loc": torch.from_numpy(gt_delta).unsqueeze(0),
                    "label_loc_weight": torch.from_numpy(
                        gt_delta_weight
                    ).unsqueeze(0),
                }
            )
            tracker.model.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)

            tot_loss = out["total_loss"]

            tot_loss.backward(retain_graph=True)

            adv_grad = where(
                (x_adv.grad > 0) | (x_adv.grad < 0),
                x_adv.grad,
                torch.tensor(0),
            )
            adv_grad = torch.sign(adv_grad)
            x_adv = x_adv - alpha * adv_grad

            x_adv = where(x_adv > x + eps, x + eps, x_adv)
            x_adv = where(x_adv < x - eps, x - eps, x_adv)
            x_adv = torch.clamp(x_adv, 0, 255)
            x_adv = x_adv.detach().clone().requires_grad_(True)

        attr.prev_atk = (x_adv - x_crop).detach().cpu().numpy()

        params.attributes.atk_latency.append(time.time() - t0)

        return tracker.get_res_from_x_crop(img, x_adv, scale, dummy)

    test.callback_track = trk_track  # type: ignore

    return test, tracker


def initial_rtaa_tester(
    args: TestArgs,
) -> Tuple[
    Test[SiamRPNTracker, TestArgs, Video, TestAttributes], SiamRPNTracker
]:
    test = Test(
        args,
        test_attr_cls=TestAttributes,
    )
    test, tracker = _initial_rtaa_tracker(test)
    return test, tracker
