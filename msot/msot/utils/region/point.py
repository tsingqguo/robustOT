from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar

ValidType: TypeAlias = float | int

T = TypeVar("T", bound=ValidType)


@dataclass
class Point(Generic[T]):
    x: T
    y: T

    def __tuple__(self):
        return (self.x, self.y)
