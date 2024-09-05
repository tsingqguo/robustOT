from __future__ import annotations

import json
from typing import (
    Any,
    Dict,
    List,
    Tuple,
)


class Config:
    def __init__(self) -> None:
        self.__setattr__(
            "_hidden_attributes",
            [
                "_hidden_attributes",
                "_attrs_iter",
            ],
        )

    def __iter__(self):
        keys = []
        for k in dir(self):
            if k[:2] == k[-2:] and k[:2] == "__":
                continue
            if k in self.__getattribute__("_hidden_attributes"):
                continue
            keys.append(k)
        try:
            anno = list(self.__class__.__annotations__.keys())
        except AttributeError:
            anno = []
        ko: List[str] = []
        for a in anno:
            try:
                ko.append(keys.pop(keys.index(a)))
            except ValueError:
                pass
        keys = [*ko, *keys]
        keys.reverse()
        self.__setattr__("_attrs_iter", keys)
        return self

    def __next__(self) -> Tuple[str, Any]:
        try:
            k: str = self.__getattribute__("_attrs_iter").pop()
            return k, self.__getattribute__(k)
        except IndexError:
            raise StopIteration

    def __str__(self) -> str:
        def dump(d: Dict[str, Any]) -> str:
            return json.dumps(d, indent=4)

        def load(s: str) -> Dict[str, Any]:
            return json.loads(s)

        def cvt(d: Any):
            if type(d) in [str, int, float, bool]:
                _d = d
            elif type(d) is dict:
                _d = {}
                for k, v in d.items():
                    if k not in [str, int]:
                        k = str(k)
                    _d[k] = cvt(v)

            elif type(d) in [list, tuple]:
                _d = [cvt(x) for x in d]
                if type(d) is tuple:
                    _d = tuple(_d)
            elif isinstance(d, Config):
                _d = load(str(d))
            else:
                _d = str(d)
            return _d

        res = cvt(dict(self))
        return dump(res)
