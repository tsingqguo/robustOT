from msot.data.datasets import VALID_DATASET_NAMES

from .config import TestConfig


class Args:
    config: TestConfig
    """config loaded from file"""

    dataset_names: list[VALID_DATASET_NAMES]

    force: bool
    """overwrite existing results"""

    output_dir: str
    """root directory for results"""

    result_timeout_thld: int
    """timeout threshold for started runs"""

    variant_name: str | None

    variant_suffix: str
