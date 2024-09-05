import os
from pyotp.config.pysot import PYSOT_Config
from pyotp.env import ENV

base_config = os.path.join(
    ENV.project_path,
    "configs",
    "siamrpnpp",
    "bases",
    "r50_train_base.py",
)


def config_assign(cfg: PYSOT_Config):
    import os
    from pyotp.env import ENV
    from pyotp.utils.option import Option, Some, NONE

    # PT = "FGSM"
    PT = "PGD"
    # PT = "CSA"

    IS_OTB = False
    # IS_OTB = True

    if IS_OTB:
        otb_str = "_otb"
    else:
        otb_str = ""

    cfg.dataset.names = [
        "COCO",
        "DET",
        "VID",
        "YOUTUBEBB",
    ]

    cfg.dataset.VID.use_hdf5 = False  # use h5_adv instead
    cfg.dataset.YOUTUBEBB.use_hdf5 = False  # use h5_adv instead

    cfg.dataset.COCO.num_use = 50000
    cfg.dataset.DET.num_use = -1
    cfg.dataset.VID.num_use = 100000
    cfg.dataset.YOUTUBEBB.num_use = 100000

    cfg.dataset.videos_per_epoch = 100000

    cfg.train.base_lr = 1e-3

    cfg.train.epoch = 5
    cfg.backbone.train_epoch = 0
    cfg.train.lr.warmup.warmup = False

    data_size = cfg.dataset.videos_per_epoch * cfg.train.epoch
    data_size = f"{data_size // 1000}k"

    refine_backbone_only = False

    cfg.train.partial.backbone = True
    if refine_backbone_only:
        cfg.train.partial.neck = False
        cfg.train.partial.rpn_head = False
        cfg.train.partial.mask_head = False
        cfg.train.partial.refine_head = False
        cfg.meta_arc = f"siamrpnpp_r50_bpt_pt={PT}_{data_size}_{otb_str}_iclr"
    else:
        cfg.train.partial.neck = True
        cfg.train.partial.rpn_head = True
        cfg.train.partial.mask_head = True
        cfg.train.partial.refine_head = True
        cfg.meta_arc = f"siamrpnpp_r50_mpt_pt={PT}_{data_size}_{otb_str}_iclr"

    cfg.train.snapshot_dir = os.path.join(
        ENV.experiments_path, "refine_iclr", cfg.meta_arc
    )

    if IS_OTB:
        cfg.train.pretrained = Some(
            os.path.join(
                ENV.experiments_path,
                "pysot",
                "pretrained",
                "siamrpn_r50_l234_dwxcorr_otb",
                "model.pth",
            )
        )
        cfg.track.penalty_k = 0.24
        cfg.track.window_influence = 0.5
        cfg.track.lr = 0.25
    else:
        cfg.train.pretrained = Some(
            os.path.join(
                ENV.experiments_path,
                "pysot",
                "pretrained",
                "siamrpn_r50_l234_dwxcorr",
                "model.pth",
            )
        )
