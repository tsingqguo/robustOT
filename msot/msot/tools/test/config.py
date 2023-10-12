from dataclasses import dataclass
from typing import Generic, TypeVar

from msot.trackers.base.config import TrackConfig
from msot.trackers.config import TrackerConfig
from msot.utils.config import Config

C = TypeVar("C", bound=TrackConfig)


@dataclass
class TestConfig(Config, Generic[C]):
    tracker: TrackerConfig[C]
    """config for tracker (tracker + model)"""

    force: bool = False
