from .got10k import GOT10kDataset

from .lasot import LaSOTDataset, ValidDatasetNames as ValidDatasetNames_LaSOT
from .nfs import NFSDataset, ValidDatasetNames as ValidDatasetNames_NFS
from .otb import OTBDataset, ValidDatasetNames as ValidDatasetNames_OTB
from .trackingnet import TrackingNetDataset
from .uav import UAVDataset, ValidDatasetNames as ValidDatasetNames_UAV
from .vot import (
    VOTDataset,
    VOTLTDataset,
    ValidDatasetNames as ValidDatasetNames_VOT,
    ValidDatasetNames as ValidDatasetNames_VOTLT,
)
from typing import Union

ValidDatasetNames = Union[
    ValidDatasetNames_LaSOT,
    ValidDatasetNames_NFS,
    ValidDatasetNames_OTB,
    ValidDatasetNames_UAV,
    ValidDatasetNames_VOT,
    ValidDatasetNames_VOTLT,
]


class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):
        """
        Args:
            name: dataset name 'OTB2015', 'LaSOT', 'UAV123', 'NFS240', 'NFS30',
                'VOT2018', 'VOT2016', 'VOT2018-LT'
            dataset_root: dataset root
            load_img: wether to load image
        Return:
            dataset
        """
        assert "name" in kwargs, "should provide dataset name"
        name = kwargs["name"]
        if "OTB" in name:
            dataset = OTBDataset(**kwargs)
        elif "LaSOT" == name:
            dataset = LaSOTDataset(**kwargs)
        elif "UAV" in name:
            dataset = UAVDataset(**kwargs)
        elif "NFS" in name:
            dataset = NFSDataset(**kwargs)
        elif "VOT2018" == name or "VOT2016" == name or "VOT2019" == name:
            dataset = VOTDataset(**kwargs)
        elif "VOT2018-LT" == name:
            dataset = VOTLTDataset(**kwargs)
        elif "TrackingNet" == name:
            dataset = TrackingNetDataset(**kwargs)
        elif "GOT-10k" == name:
            dataset = GOT10kDataset(**kwargs)
        else:
            raise Exception("unknow dataset {}".format(kwargs["name"]))
        return dataset
