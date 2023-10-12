from msot.models.config import ModelConfig


def configure(cfg: ModelConfig):
    cfg.backbone.type = "resnet50"
    cfg.backbone.kwargs = {
        "used_layers": [2, 3, 4],
    }

    cfg.adjust.kwargs = {
        "in_channels": [512, 1024, 2048],
        "out_channels": [256, 256, 256],
    }

    cfg.rpn.kwargs = {
        "anchor_num": 5,
        "in_channels": [256, 256, 256],
        "weighted": True,
    }
