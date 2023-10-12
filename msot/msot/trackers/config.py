from dataclasses import dataclass, field
from typing import Generic, TypeVar

from msot.models.config import ModelConfig
from msot.utils.config import Config

from .base.config import TrackConfig
from .types import Backends

C = TypeVar("C", bound=TrackConfig)


@dataclass
class TrackerConfig(Config, Generic[C]):
    track: C
    """config for tracker instance"""

    model: ModelConfig = field(default_factory=ModelConfig)
    """config for tracker's model"""

    name: str = "arbitrary_tracker"
    """name of tracker"""

    backend: Backends = Backends.CPU
