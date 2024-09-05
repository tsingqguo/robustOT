# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Literal, Type, TypedDict

from pysot.core.config import cfg
from pysot.tracker.siamrpn_tracker import SiamRPNTracker
from pysot.tracker.siammask_tracker import SiamMaskTracker
from pysot.tracker.siamrpnlt_tracker import SiamRPNLTTracker
from pysot.types import VALID_TRACKER


TRACKS: dict[VALID_TRACKER, Type[SiamRPNTracker]] = {
    "SiamRPNTracker": SiamRPNTracker,
    "SiamMaskTracker": SiamMaskTracker,
    "SiamRPNLTTracker": SiamRPNLTTracker,
}

# TODO: generic tracker
def build_tracker(model) -> SiamRPNTracker:
    return TRACKS[cfg.TRACK.TYPE](model)
