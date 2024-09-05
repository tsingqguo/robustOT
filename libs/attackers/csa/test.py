import os

from .utils import AtkOn, AtkOnFull, AtkType

from attackers.csa.CSA.base_model import Base_L2_500
from attackers.csa.CSA.tracker.siamrpn_tracker_cfgmod import CSA_SiamRPNTracker

# from pix2pix.models.base_model_csa import BaseModel_CSA
from pysot.models.model_builder_cfgmod import ModelBuilder
from pyotp.tools.test import (
    TestAttributes,
    TestCBParamFrame,
    TestCBParamFrameInit,
    args_parse,
    initial_testing_args as _initial_testing_args,
    Test,
    TestArgs as _TestArgs,
    TestCliArgs as _TestCliArgs,
)
from pysot.tracker.siamrpn_tracker_cfgmod import SiamRPNTracker
from toolkit.datasets.video import Video
from typing import (
    List,
    Literal,
    Optional,
    Tuple,
    Type,
)


class TestCliArgs(_TestCliArgs):
    attack_on: AtkOn
    """
    attack on search only("0S"), template only("T0"), template&search("TS")
    """

    # @alias=a
    attack_type: AtkType
    """
    attack type should either co(cooling-only) or cs(cooling-shrinking)
    """

    save_adv_result: bool = False
    """
    save clean/adv inputs to ...
    """


class TestArgs(_TestArgs):
    attack_on: AtkOnFull
    GAN: Base_L2_500  # TODO: for _500_ only
    model_builder: Type[ModelBuilder]
    save_adv_result: bool


# A = TypeVar("A", bound=TestArgs)
# T = TypeVar("T", bound=SiamRPNTracker)  # TODO:
# V = TypeVar("V", bound=Video)


def initial_testing_args(
    cliargs: TestCliArgs, argv: List[str], args: TestArgs
) -> None:
    _initial_testing_args(cliargs, argv, args)
    if cliargs.attack_on == "0S":
        args.attack_on = "search_only"
    elif cliargs.attack_on == "T0":
        args.attack_on = "template_only"
    elif cliargs.attack_on == "TS":
        args.attack_on = "search_and_template"
    else:
        raise ValueError
    #
    if cliargs.attack_type == "co":
        suffix = ""
        if args.attack_on == "search_only":
            from attackers.csa.CSA.GAN_utils_search_co import GAN, opt
        elif args.attack_on == "template_only":
            from attackers.csa.CSA.GAN_utils_template_co import GAN, opt
        elif args.attack_on == "search_and_template":
            from attackers.csa.CSA.GAN_utils_search_co import GAN, opt

            suffix = "_TS"
        else:
            raise ValueError()
        args.GAN = GAN
        args.model_name = opt.model + suffix
    elif cliargs.attack_type == "cs":
        suffix = ""
        if args.attack_on == "search_only":
            from attackers.csa.CSA.GAN_utils_search_cs import GAN, opt
        elif args.attack_on == "template_only":
            from attackers.csa.CSA.GAN_utils_template_cs import GAN, opt
        elif args.attack_on == "search_and_template":
            from attackers.csa.CSA.GAN_utils_search_cs import GAN, opt

            suffix = "_TS"
        else:
            raise ValueError()
        args.GAN = GAN
        args.model_name = opt.model + suffix
    # FIXME:
    # args.variant_name += cliargs.variant_suffix

    args.save_adv_result = cliargs.save_adv_result


def _initial_csa_tracker(test: Test) -> CSA_SiamRPNTracker:
    raw_tracker = test.initial_tracker()

    tracker = CSA_SiamRPNTracker(raw_tracker)

    def get_adv_result_saving_path(video: Video) -> Optional[str]:
        args = test.args
        if args.save_adv_result:
            return os.path.join(
                ".",
                "adv_results",
                args.dataset,
                args.model_name,
                args.variant_name + args.variant_suffix,
                video.name,
            )
        else:
            return None

    def csa_trk_init(
        params: TestCBParamFrameInit[
            TestArgs, SiamRPNTracker, Video, TestAttributes
        ],
        dummy: Optional[SiamRPNTracker],
    ):
        args = params.args
        gt_bbox_ = params.gt_bbox_
        img = params.img
        video = params.video
        #
        if args.attack_on == "search_only":
            tracker.raw.init(img, gt_bbox_, dummy)
        elif args.attack_on == "template_only":
            """GAN"""
            tracker.init_adv(
                img,
                gt_bbox_,
                args.GAN,  # ignore this without type: ignore
                dummy,
                save_path=get_adv_result_saving_path(video),
                name=video.name,
            )
        elif args.attack_on == "search_and_template":
            """GAN"""
            tracker.init_adv_S(
                img,
                gt_bbox_,
                args.GAN,  # ignore this without type: ignore
                dummy,
                save_path=get_adv_result_saving_path(video),
                name=video.name,
            )  # attack template with search model
        else:
            raise ValueError

    def csa_trk_tracker(
        params: TestCBParamFrame[
            TestArgs, SiamRPNTracker, Video, TestAttributes
        ],
        dummy: Optional[SiamRPNTracker],
        extra_options: Optional[dict] = None,
    ):
        args = params.args
        img = params.img
        idx = params.idx
        video = params.video
        if extra_options is not None:
            if extra_options.pop('keep_clean', False):
                return tracker.raw.track(img, dummy)
            
        import time
        t0 = time.time()

        if args.attack_on in [
            "search_only",
            "search_and_template",
        ]:
            """GAN"""
            outputs, ta = tracker.track_adv(
                img,
                args.GAN,  # ignore this without type: ignore
                dummy,
                save_path=get_adv_result_saving_path(video),
                frame_id=idx,
            )
        elif args.attack_on in ["template_only"]:
            outputs = tracker.raw.track(img, dummy)
            ta = 0
        else:
            raise ValueError
        
        # params.attributes.atk_latency.append(ta)
        params.attributes.atk_latency.append(time.time() - t0)

        return outputs

    test.callback_init = csa_trk_init
    test.callback_track = csa_trk_tracker  # type: ignore
    # test.callback_skip = csa_trk_skip

    return tracker


def initial_csa_tester(
    args: TestArgs,
) -> Tuple[
    Test[SiamRPNTracker, TestArgs, Video, TestAttributes], CSA_SiamRPNTracker
]:  # TODO:
    test = Test(args)
    tracker = _initial_csa_tracker(test)
    return test, tracker
