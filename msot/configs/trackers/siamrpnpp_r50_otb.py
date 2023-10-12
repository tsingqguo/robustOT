def configure():
    import os
    from msot.env import ENV
    from msot.models.config import ModelConfig
    from msot.trackers.config import TrackerConfig
    from msot.trackers.siamese.siamrpnpp import TrackConfig as SiamRPNPPConfig
    from msot.utils.config.helper import from_file

    cfg = TrackerConfig(SiamRPNPPConfig())

    cfg.name = "siamrpnpp_r50_otb"

    cfg.track.penalty_k = 0.24
    cfg.track.window_influence = 0.5
    cfg.track.lr = 0.25
    cfg.track.exemplar_size = 127
    cfg.track.instance_size = 255
    cfg.track.base_size = 8
    cfg.track.context_amount = 0.5

    cfg.model = from_file(
        os.path.join(ENV.msot_root, "configs/trackers/models/r50.py"),
        ModelConfig,
    )
    cfg.model.pretrained = os.path.join(
        ENV.msot_root,
        "pretrained_models/pysot/siamrpn_r50_l234_dwxcorr_otb.pth",
    )

    return cfg
