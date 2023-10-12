from typing import TypeVar

T = TypeVar("T")


# FIXME: remove this
def unwrap_or(x: T | None, default: T) -> T:
    if x is None:
        return default
    return x
