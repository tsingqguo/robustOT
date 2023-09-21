from typing import (
    Any,
    Generic,
    Optional,
    TypeVar,
)


T = TypeVar("T")


class _NoneErr(Exception):
    pass


class Option(Generic[T]):
    _v: Optional[T]

    def __init__(self, val: Optional[T] = None) -> None:
        self._v = val

    @property
    def _val(self) -> T:
        if self._v is not None:
            return self._v
        else:
            raise _NoneErr

    def is_none(self) -> bool:
        return self._v is None

    def is_some(self) -> bool:
        return not self.is_none()

    def unwrap(self) -> T:
        try:
            return self._val
        except _NoneErr:
            raise ValueError("called `Option::unwrap()` on a `None` value")

    def unwrap_or(self, default: T) -> T:
        try:
            return self._val
        except _NoneErr:
            return default

    def __str__(self) -> str:
        try:
            return f"Option::Some({self._val})"
        except _NoneErr:
            return f"Option::None"

    def __repr__(self) -> str:
        return self.__str__()

def Some(value: T) -> Option[T]:
    return Option(value)


NONE: Option[Any] = Option()
