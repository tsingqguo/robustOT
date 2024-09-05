from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    NoReturn,
    Optional,
    TypeVar,
    Union,
)
from pyotp.utils import print_table, TableArgs
from toolkit.datasets.dataset import Dataset, VariantNotFound
from toolkit.datasets.video import Video


R = TypeVar("R")
V = TypeVar('V', bound=Video)


class Benchmark(Generic[V]):
    dataset: Dataset[V]
    default_variants: str

    def _eval(
        self,
        eval_tracker: Optional[str],
        eval_variant: Optional[str],
        get_result: Callable[[str, str], R],
    ) -> Dict[str, R]:
        trackers: List[str]
        variants: List[str]

        if eval_tracker == None:
            # raise KeyError('eval_tracker undefined')
            trackers = self.dataset.tracker_names
        if isinstance(eval_tracker, str):
            trackers = [eval_tracker]
        else:
            raise ValueError(
                f"expect eval_tracker be str but got {type(eval_tracker)}"
            )

        if eval_variant == None:
            variants = [self.default_variants]
        if isinstance(eval_variant, str):
            variants = [eval_variant]
        else:
            raise ValueError(
                f"expect eval_variant be str but got {type(eval_variant)}"
            )

        res = {}

        for tracker_name in trackers:
            for variant in variants:
                try:
                    results = get_result(tracker_name, variant)
                    res[f"{tracker_name}.{variant}"] = results
                except VariantNotFound:
                    # print(f"{tracker_name}.{variant} not found")
                    continue
                except Exception as e:
                    print(f'[ERR] failed on tracker: {tracker_name}.{variant}')
                    print(e)
        return res

    def _print_result(
        self,
        headers: List[str],
        data: Iterable[Iterable],
        fmt: Dict[str, Callable[[Any], str]] = {},
        arbitrary_fmt: List[Callable[[Any], Union[str, NoReturn]]] = [
            lambda x: f"{x:.3f}"
        ],
        print_args: TableArgs = {},
    ) -> None:
        return print_table(
            headers,
            data,
            fmt,
            arbitrary_fmt=arbitrary_fmt,
            print_args={
                "alignment": "l" + "c" * (len(headers) - 1),
                **print_args,
            },  # type: ignore
        )
