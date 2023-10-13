from enum import Enum, Flag, auto


class AttackOn(Flag):
    SEARCH = 1 << 0
    TEMPLATE = 1 << 1


class AttackType(Enum):
    COOLING_ONLY = auto()
    COOLING_SHRINKING = auto()
