from enum import Enum
from typing import Literal

AttackerMethod = Literal["BIM", "FGSM"]


class NormType(Enum):
    L_inf = 0
    L_1 = 1
    L_2 = 2
    Momen = 3


class RegType(Enum):
    L21 = 0
    L2 = 1
    L_inf = 2
    Weighted = 3
    NONE = 4


class AtkType(Enum):
    UA = 0
    TA = 1
