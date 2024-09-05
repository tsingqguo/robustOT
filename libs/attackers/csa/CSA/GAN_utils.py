import os
from argparse import Namespace
from attackers.csa.CSA.options.base_options import CSA_ATK_ON, CSA_ATK_TYPE, CSA_BaseOptions
from attackers.csa.CSA.options.test_options import TestOptions
from pix2pix.models import create_model
from pix2pix.models.base_model_csa import BaseModel_CSA
from typing import Callable, List, Literal, Optional, Tuple


def cvt_atk_type_str(atk_type: Literal["CO", "CS"]) -> CSA_ATK_TYPE:
    if atk_type == "CO":
        return "co"
    elif atk_type == "CS":
        return "cs"
    else:
        raise ValueError


def cvt_atk_on_str(atk_on: Literal["0S", "T0", "TS"]) -> CSA_ATK_ON:
    if atk_on == "0S":
        return "search_only"
    elif atk_on == "T0":
        return "template_only"
    elif atk_on == "TS":
        return "search_and_template"
    else:
        raise ValueError


def init_GAN(
    atk_type: CSA_ATK_TYPE,
    atk_on: CSA_ATK_ON,
    args_overwrite: List[str] = [],
    custom_opt: Optional[Callable[[Namespace], Namespace]] = None,
) -> Tuple[BaseModel_CSA, Namespace]:
    opt = TestOptions(atk_type).parse(args_overwrite)
    saved_name: str

    if atk_type == "co":
        #
        if atk_on == "search_only":
            saved_name = "1_net_G.pth"
            opt.netG = "unet_256"
            opt.model = "G_search_L2_500"
        #
        elif atk_on == "template_only":
            saved_name = "latest_net_G.pth"
            opt.netG = "unet_128"
            opt.model = "G_template_L2_500"
        #
        elif atk_on == "search_and_template":
            saved_name = "1_net_G.pth"
            opt.netG = "unet_256"
            opt.model = "G_search_L2_500"
    #
    elif atk_type == "cs":
        #
        if atk_on == "search_only":
            saved_name = "1_net_G.pth"
            opt.netG = "unet_256"
            opt.model = "G_search_L2_500_regress"
        #
        elif atk_on == "template_only":
            saved_name = "latest_net_G.pth"
            opt.netG = "unet_128"
            opt.model = "G_template_L2_500_regress"
        #
        elif atk_on == "search_and_template":
            saved_name = "1_net_G.pth"
            opt.netG = "unet_256"
            opt.model = "G_search_L2_500"
    #
    else:
        raise NotImplemented(f"atk_type: {atk_type}")

    if custom_opt is not None:
        opt = custom_opt(opt)

    GAN: BaseModel_CSA = create_model(opt)  # type: ignore
    GAN.load_path_overwrite = os.path.join(
        os.environ["PYOTP_PATH"],
        "experiments",
        "CSA",
        "checkpoints",
        opt.model,
        saved_name,
    )
    GAN.setup(opt)  # regular setup: load and print networks; create schedulers
    GAN.eval()
    return (GAN, opt)
