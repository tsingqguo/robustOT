import os
from pyotp.config.pysot import PYSOT_Config
from pyotp.env import ENV

base_config = os.path.join(
    ENV.project_path,
    "configs",
    "siamrpnpp",
    "bases",
    "r50_base.py",
)


def config_assign(cfg: PYSOT_Config):
    cfg.train.epoch = 20
    cfg.train.batch_size = 64

    cfg.train.partial.backbone = True
    cfg.train.partial.neck = True
    cfg.train.partial.rpn_head = True
    cfg.train.partial.mask_head = True
    cfg.train.partial.refine_head = True

    cfg.dataset.names = ["COCO", "DET"]
