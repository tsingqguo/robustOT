from dataclasses import dataclass
from typing_extensions import Self


import json

from .helper import from_file, from_file_unsafe


@dataclass
class Config:
    def __str__(self) -> str:
        def serialize(obj):
            if hasattr(obj, "__str__"):
                return str(obj)
            elif hasattr(obj, "__dict__"):
                return obj.__dict__

        return json.dumps(
            self.__dict__,
            indent=4,
            default=serialize,
        )

    @classmethod
    def unsafe_load(cls, config_fp: str) -> Self:
        cfg = from_file_unsafe(config_fp)
        if not isinstance(cfg, cls):
            raise TypeError(f"Invalid config for test, got {type(cfg)}")
        return cfg
