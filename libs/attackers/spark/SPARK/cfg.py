from pyotp.config.pysot import PYSOT_Config
from pyotp.utils.config import Config
from attackers.spark.SPARK.types import AtkType, AttackerMethod


class _Cfg_Attacker(Config):
    method: AttackerMethod = "BIM"

    show: bool = False

    # attack_type: AtkType = AtkType.UA

    gen_adv_img: bool = False

    save_video: bool = False

    apts: bool = False

    eval: bool = True

    check_exist: bool = True

    save_meta: bool = False


class SPARK_Config(PYSOT_Config):

    attacker: _Cfg_Attacker = _Cfg_Attacker()


CFG = SPARK_Config()
