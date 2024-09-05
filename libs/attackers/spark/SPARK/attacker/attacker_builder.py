# from attackers.spark.SPARK.attacker.fgsm_attacker import FGSMAttacker
# from attackers.spark.SPARK.attacker.bim_attacker import BIMAttacker
# from attackers.spark.SPARK.attacker.oim_attacker import OIMAttacker
from attackers.spark.SPARK.attacker.oim_atk_cfgmod import OIMAttacker
from attackers.spark.SPARK.types import AtkType, NormType, RegType
from typing import Any, Dict, Literal

# from attackers.spark.SPARK.attacker.oim_attacker_eco import OIMAttackerECO
# from attackers.spark.SPARK.attacker.oim_attacker_siamdw import OIMAttackerSiamDW
# from attackers.spark.SPARK.attacker.mifgsm_attacker import MIFGSMAttacker
# from attackers.spark.SPARK.attacker.cw_attacker import CWAttacker
# from attackers.spark.SPARK.attacker.sap_attacker import SAPAttacker
# from attackers.spark.SPARK.attacker.curl_attacker import CURLAttacker

ValidAttacker = Literal["OIM"]

ATTACKERS: Dict[ValidAttacker, Any] = {
    #   'FGSM': FGSMAttacker,
    #   'BIM': BIMAttacker,
    "OIM": OIMAttacker,
    #   'OIMECO':OIMAttackerECO,
    #   'OIMSIAMDW': OIMAttackerSiamDW,
    #   'MIFGSM': MIFGSMAttacker,
    #   'CW-L2':CWAttacker,
    #   'SAP':SAPAttacker,
    #   'CURL':CURLAttacker
}


def build_attacker(
    attacker_method: ValidAttacker,
    type: AtkType,
    max_num: int = 10,
    apts_num: int = 2,
    inta: int = 10,
    reg_type: RegType = RegType.L21,
    norm_type: NormType = NormType.L_inf,
    eplison: float = 1,
    accframes: int = 30,
):
    return ATTACKERS[attacker_method](
        type,
        max_num,
        inta=inta,
        apts_num=apts_num,
        reg_type=reg_type,
        norm_type=norm_type,
        eplison=eplison,
        accframes=accframes,
    )
