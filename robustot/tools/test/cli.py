import sys
from typing import Optional, Type, TypeVar

from typed_cap import Cap

from msot.data.datasets import VALID_DATASET_NAMES
from msot.utils import unwrap_or

from .args import Args
from .config import TestConfig

T = TypeVar("T")


class CliArgs:
    # @alias=c
    config: str
    """path to config file"""

    # @alias=d
    datasets: Optional[list[VALID_DATASET_NAMES]]
    """dataset names to test"""

    # @alias=f
    force: Optional[bool]
    """overwrite existing results"""

    # @alias=o
    output_dir: str = "results"
    """path to output directory"""

    tracker_name: Optional[str]
    """override tracker name"""

    # @alias=t
    timeout: int = 60 * 60 * 24
    """timeout for each run"""

    suffix: Optional[str]
    """suffix for variant name"""

    # @alias=g
    gpu_id: Optional[list[int]]
    """gpu id to use"""


def args_parse(
    cliargs: Type[T], argv: list[str] = sys.argv[1:]
) -> tuple[list[str], T]:
    cap = Cap(cliargs)
    parsed = cap.parse(argv)
    return parsed.args, parsed.val


def initial_testing_args(
    cliargs: CliArgs,
    argv: list[str],
    args: Args,
):
    args.config = TestConfig.unsafe_load(cliargs.config)
    args.dataset_names = unwrap_or(cliargs.datasets, [])
    args.force = unwrap_or(cliargs.force, args.config.force)
    args.output_dir = cliargs.output_dir
    args.result_timeout_thld = cliargs.timeout
    args.variant_name = None
    args.variant_suffix = unwrap_or(cliargs.suffix, "")

    if cliargs.gpu_id is not None:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cliargs.gpu_id))