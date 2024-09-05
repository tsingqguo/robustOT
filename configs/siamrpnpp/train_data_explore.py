import os
from pyotp.env import ENV
from pyotp.config.pysot import PYSOT_Config

base_config = os.path.join(
    ENV.project_path,
    "configs",
    "siamrpnpp",
    "bases",
    "r50_dummy.py",
)


def config_assign(cfg: PYSOT_Config):
    cfg.cuda = True

    cfg.dataset.names = [
        # "YOUTUBEBB",
        # "VID",
        # "COCO",
        # "DET",
    ]
    cfg.dataset.search.shift = 0
    cfg.dataset.search.scale = 0.0

    cfg.dataset.neg = 0
    cfg.dataset.videos_per_epoch = -1

    cfg.train.epoch = 1
    cfg.train.print_freq = 50
    # cfg.train.batch_size = 16
    cfg.train.batch_size = 8
