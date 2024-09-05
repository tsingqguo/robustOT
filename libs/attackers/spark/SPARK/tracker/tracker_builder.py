from pysot.tracker.siamrpn_tracker import SiamRPNTracker
from pysot.tracker.tracker_builder import TRACKS
from attackers.spark.SPARK.config.attack_cfg import cfg
from attackers.spark.SPARK.tracker.siamrpn_tracker import SiamRPNTracker_SPARK

TRACKS = {
    **TRACKS,
    "SiamRPNTracker_SPARK": SiamRPNTracker_SPARK,
}


# TODO: generic tracker
def build_tracker(model) -> SiamRPNTracker:
    return TRACKS[cfg.TRACK.TYPE](model)
