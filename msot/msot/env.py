import os
from dataclasses import dataclass


@dataclass
class MSOTEnv:
    msot_root: str
    dataset_testing_root: str


ENV = MSOTEnv(
    msot_root=os.environ.get("MSOT_ROOT", ""),
    dataset_testing_root=os.environ.get("MSOT_TESTING", ""),
)
