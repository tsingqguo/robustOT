from dataclasses import dataclass
from msot.data.datasets import Dataset

from .config import TestConfig
from .utils.process import Processor

@dataclass
class Args:
    config: TestConfig
    """config loaded from file"""

    datasets: list[Dataset]

    processors: list[Processor] | None

    force: bool
    """overwrite existing results"""

    output_dir: str
    """root directory for results"""

    result_timeout_thld: int
    """timeout threshold for started runs"""

    variant_name: str | None

    variant_suffix: str
