from pysot.core.config import cfg
from yacs.config import CfgNode as CN

__C = cfg

# ------------------------------------------------------------------------ #
# ATTACKER OPTIONS
# ------------------------------------------------------------------------ #
__C.ATTACKER = CN()

__C.ATTACKER.METHOD = "BIM"  #'BIM' # 'FGSM'

__C.ATTACKER.SHOW = False

__C.ATTACKER.ATTACK_TYPE = "UA"

__C.ATTACKER.GEN_ADV_IMG = False

__C.ATTACKER.SAVE_VIDEO = False

__C.ATTACKER.APTS = True

__C.ATTACKER.OPTICAL_FLOW = False

__C.ATTACKER.EVAL = True

__C.ATTACKER.CHECK_EXIST = True

__C.ATTACKER.SAVE_META = False
