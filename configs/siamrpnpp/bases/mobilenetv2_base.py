from pyotp.config.pysot import PYSOT_Config


def config_assign(cfg: PYSOT_Config):
    cfg.cuda = True

    cfg.meta_arc = "do_not_use_this_directly"

    cfg.backbone.type = "mobilenetv2"
    cfg.backbone.kwargs = {
        "used_layers": [3, 5, 7],
        "width_mult": 1.4,
    }

    cfg.adjust.type = "AdjustAllLayer"
    cfg.adjust.kwargs = {
        "in_channels": [44, 134, 448],
        "out_channels": [256, 256, 256],
    }

    cfg.rpn.kwargs = {
        "anchor_num": 5,
        "in_channels": [256, 256, 256],
        "weighted": False,
    }

    cfg.mask.mask = False

    cfg.track.penalty_k = 0.04
    cfg.track.window_influence = 0.4
    cfg.track.lr = 0.5
    cfg.track.exemplar_size = 127
    cfg.track.instance_size = 255
    cfg.track.base_size = 8
    cfg.track.context_amount = 0.5
