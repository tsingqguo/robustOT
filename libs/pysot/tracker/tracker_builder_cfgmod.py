from pyotp.config.pysot import CFG
from pysot.models import ModelBuilder
from pysot.tracker.siamrpn_tracker_cfgmod import SiamRPNTracker
# from pysot.tracker.siammask_tracker import SiamMaskTracker
# from pysot.tracker.siamrpnlt_tracker import SiamRPNLTTracker
from pysot.types import VALID_TRACKER  # TODO:
from typing import Type


def build_tracker(model: ModelBuilder) -> SiamRPNTracker:
    TRACKERS: dict[VALID_TRACKER, Type[SiamRPNTracker]] = {
        "SiamRPNTracker": SiamRPNTracker,
        # "SiamMaskTracker": SiamMaskTracker, # TODO: `Config` compatible
        # "SiamRPNLTTracker": SiamRPNLTTracker, # TODO: `Config` compatible
    }
    return TRACKERS[CFG.track.type](model)
