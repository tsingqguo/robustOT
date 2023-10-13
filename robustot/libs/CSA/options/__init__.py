from argparse import Namespace

from ..types import AttackOn, AttackType
from .test import TestOptions


def get_test_options(
    attack_on: AttackOn,
    attack_type: AttackType,
) -> Namespace:
    opt = TestOptions(attack_type).parse([])
    if attack_type is AttackType.COOLING_ONLY:
        if AttackOn.SEARCH in attack_on:
            opt.model = "G_search_L2_500"
            opt.netG = "unet_256"
            opt.epoch = "1"
        elif AttackOn.TEMPLATE in attack_on:
            opt.model = "G_template_L2_500"
            opt.netG = "unet_128"
            opt.epoch = "latest"
        else:
            raise RuntimeError("Unknown attack on")
    elif attack_type is AttackType.COOLING_SHRINKING:
        if AttackOn.SEARCH in attack_on:
            opt.model = "G_search_L2_500_regress"
            opt.netG = "unet_256"
            opt.epoch = "1"
        elif AttackOn.TEMPLATE in attack_on:
            opt.model = "G_template_L2_500_regress"
            opt.netG = "unet_128"
            opt.epoch = "latest"
        else:
            raise RuntimeError("Unknown attack on")
    else:
        raise RuntimeError("Unknown attack type")
    return opt
