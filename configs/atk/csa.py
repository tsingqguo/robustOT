def setup():
    from robustot.attackers.CSA import (
        CSAConfig,
        CSAProcessor,
        AttackOn,
        AttackType,
    )

    csa_config = CSAConfig(
        AttackOn.TEMPLATE | AttackOn.SEARCH,
        AttackType.COOLING_SHRINKING,
        "pretrained_models/CSA/checkpoints",  # TODO: ENV impl
    )

    return CSAProcessor("csa_atk", csa_config)
