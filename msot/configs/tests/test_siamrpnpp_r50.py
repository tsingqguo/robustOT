def configure():
    import os
    from msot.env import ENV
    from msot.tools.test.config import TestConfig
    from msot.trackers.config import TrackerConfig
    from msot.trackers.siamese.siamrpnpp import TrackConfig as SiamRPNPPConfig
    from msot.trackers.types import Backends
    from msot.utils.config.helper import from_file

    tracker = TrackerConfig.unsafe_load(
        os.path.join(ENV.msot_root, "configs/trackers/siamrpnpp_r50.py"),
    )

    cfg = TestConfig(tracker)

    cfg.tracker.backend = Backends.CUDA

    return cfg
