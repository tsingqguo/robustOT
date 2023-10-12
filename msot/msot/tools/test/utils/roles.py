from enum import IntEnum, auto


class TDRoles(IntEnum):
    NONE = 0
    TRACKER = auto()
    DEFENDER = auto()
    ATTACKER = auto()
    TEST = auto()
    ANALYSIS = auto()
    DEBUG = auto()
