from collections import OrderedDict
from typing import TypeVar

U = TypeVar("U")
V = TypeVar("V")


class FIFODict(OrderedDict[U, V]):
    capacity: int

    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity

    def __setitem__(self, key: U, value: V):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.capacity:
            self.popitem(last=False)
