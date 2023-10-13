def setup():
    from robustot.attackers.RTAA import RTAAConfig, RTAAProcessor

    config = RTAAConfig(
        eps=5,
        iteration=5,
    )

    return RTAAProcessor(f"rtaa_atk_e{config.eps}i{config.iteration}", config)
