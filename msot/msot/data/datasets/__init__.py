import os
from typing import Literal

from msot.libs.potp_test.libs.dataset import Dataset

VALID_DATASET_NAMES = Literal[
    "VOT2018",
    "VOT2019",
    "OTB100",
    "LaSOT",
    "NFS30",
    "NFS240",
]


def get_dataset(
    name: VALID_DATASET_NAMES, dataset_root: str | None = None
) -> Dataset:
    if dataset_root is None:
        from msot.env import ENV

        dataset_root = ENV.dataset_testing_root

    fp = os.path.join(dataset_root, f"{name.lower()}_rawjpeg.h5")

    if name == "VOT2019":
        from msot.libs.potp_test.libs.datasets.vot import VotDataset

        db = VotDataset(fp)
    elif name == "OTB100":
        from msot.libs.potp_test.libs.datasets.otb import OTB100Dataset

        db = OTB100Dataset(fp)
    elif name == "UAV123":
        from msot.libs.potp_test.libs.datasets.uav import UAVDataset

        db = UAVDataset(fp)
    elif name == "LaSOT":
        from msot.libs.potp_test.libs.datasets.lasot import LaSOTDataset

        db = LaSOTDataset(fp)
    elif name == "NFS30":
        from msot.libs.potp_test.libs.datasets.nfs import NFS30Dataset

        db = NFS30Dataset(fp)
    elif name == "NFS240":
        from msot.libs.potp_test.libs.datasets.nfs import NFS240Dataset

        db = NFS240Dataset(fp)
    else:
        raise NotImplementedError
    return db
