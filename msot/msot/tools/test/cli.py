import sys
from typing import Type, TypeVar, Union

from typed_cap import Cap

from msot.data.datasets import VALID_DATASET_NAMES, get_dataset
from msot.utils import unwrap_or
from msot.utils.log import LogLevel, get_logger, set_level

from .args import Args
from .config import TestConfig
from .utils.process import Processor

log = get_logger(__name__)

A = TypeVar("A", bound=Args)
T = TypeVar("T")


class CliArgs:
    # @alias=c
    config: str
    """path to config file"""

    # @alias=d
    datasets: Union[list[VALID_DATASET_NAMES], None]  # TODO:
    """dataset names to test"""

    # @alias=S
    sequences: Union[list[str], None]  # TODO:

    # @alias=p @none_delimiter
    processors: Union[list[str], None]  # TODO: typed_cap uniontype issue

    # @alias=f
    force: bool = False
    """overwrite existing results"""

    # @alias=o
    output_dir: str = "results"
    """path to output directory"""

    tracker_name: str | None
    """override tracker name"""

    # @alias=t
    timeout: int = 60 * 60 * 24
    """timeout for each run"""

    suffix: str | None
    """suffix for variant name"""

    # @alias=g
    gpu_id: list[int] | None
    """gpu id to use"""

    # @alias=v
    verbose: bool = False


def args_parse(
    cliargs: Type[T], argv: list[str] = sys.argv[1:]
) -> tuple[list[str], T]:
    cap = Cap(cliargs)
    parsed = cap.parse(argv)
    v_cnt = parsed.count("verbose")
    log_level: LogLevel
    if v_cnt == 0:
        log_level = LogLevel.ERROR
    elif v_cnt == 1:
        log_level = LogLevel.WARNING
    elif v_cnt == 2:
        log_level = LogLevel.INFO
    else:
        log_level = LogLevel.DEBUG
    set_level(log_level)
    log.debug(f"log level has been set to {log_level.value}")
    return parsed.argv, parsed.args


def test_args_init(
    cliargs: CliArgs,
    cliargs_argv: list[str],
    args_cls: Type[A],
) -> A:
    config = TestConfig.unsafe_load(cliargs.config)
    datasets = []
    for d_name in unwrap_or(cliargs.datasets, []):
        datasets.append(get_dataset(d_name))
    sequences = cliargs.sequences
    if cliargs.processors is not None:
        processors = []
        for p_fp in cliargs.processors:
            processors.append(Processor.from_file(p_fp))
    else:
        processors = None

    force = unwrap_or(cliargs.force, config.force)
    output_dir = cliargs.output_dir
    result_timeout_thld = cliargs.timeout
    variant_name = None
    variant_suffix = unwrap_or(cliargs.suffix, "")

    if cliargs.gpu_id is not None:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cliargs.gpu_id))

    return args_cls(
        config=config,
        datasets=datasets,
        sequences=sequences,
        processors=processors,
        force=force,
        output_dir=output_dir,
        result_timeout_thld=result_timeout_thld,
        variant_name=variant_name,
        variant_suffix=variant_suffix,
    )
