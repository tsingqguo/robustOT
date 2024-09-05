from argparse import Namespace
from attackers.csa.CSA.base_model import Base_L2_500
from enum import Enum
from typing import Callable, Literal, Tuple, Type

AtkOn = Literal["0S", "T0", "TS"]
AtkOnFull = Literal["search_only", "template_only", "search_and_template"]
AtkType = Literal["co", "cs"]

class AtkOnEnum(Enum):
    S = "search_only"
    T = "template_only"
    TS = "search_and_template"

def cvt_atkon_to_atkonfull(atkon: AtkOn) -> AtkOnFull:
    if atkon == "0S":
        return "search_only"
    elif atkon == "T0":
        return "template_only"
    elif atkon == "TS":
        return "search_and_template"
    else:
        raise ValueError

def cvt_atkon_to_atkenum(atkon: AtkOn) -> AtkOnEnum:
    if atkon == "0S":
        return AtkOnEnum.S
    elif atkon == "T0":
        return AtkOnEnum.T
    elif atkon == "TS":
        return AtkOnEnum.TS
    else:
        raise ValueError

def get_CSA_GAN(atk: AtkOnEnum, atk_type: AtkType) -> Tuple[Base_L2_500, Namespace, str]:
    suffix: str = ''
    GAN: Base_L2_500

    if atk_type == 'co':
        if atk is AtkOnEnum.S:
            from attackers.csa.CSA.GAN_utils_search_co import GAN, opt
        elif atk is AtkOnEnum.T:
            from attackers.csa.CSA.GAN_utils_template_co import GAN, opt
        elif atk is AtkOnEnum.TS:
            from attackers.csa.CSA.GAN_utils_search_co import GAN, opt
            suffix = '_TS'
        else:
            raise ValueError(f'bad atk: {atk}')

    elif atk_type == 'cs':
        suffix = ''
        if atk is AtkOnEnum.S:
            from attackers.csa.CSA.GAN_utils_search_cs import GAN, opt
        elif atk is AtkOnEnum.T:
            from attackers.csa.CSA.GAN_utils_template_cs import GAN, opt
        elif atk is AtkOnEnum.TS:
            from attackers.csa.CSA.GAN_utils_search_cs import GAN, opt
            suffix = '_TS'
        else:
            raise ValueError(f'bad atk: {atk}')

    else:
        raise ValueError(f'bad atk_type: {atk_type}')

    return GAN, opt, suffix

def get_attack_fn(atk: AtkOnEnum) -> Callable:
    ...