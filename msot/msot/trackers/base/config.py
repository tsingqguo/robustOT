from dataclasses import dataclass

from msot.utils.config import Config


@dataclass
class TrackConfig(Config):
    exemplar_size: int
    """Size of examplar"""

    instance_size: int
    """Size of instance"""
