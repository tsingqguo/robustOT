from typing import Literal

VALID_TRACKER = Literal[
    "SiamRPNTracker",
    "SiamMaskTracker",
    "SiamRPNLTTracker",
]

VALID_BACKBONE = Literal[
    "alexnet",
    "alexnetlegacy",
    "mobilenetv2",
    "resnet18",
    "resnet34",
    "resnet50",
]

VALID_ADJUST = Literal[
    "AdjustLayer",
    "AdjustAllLayer",
]

VALID_MASK = Literal[
    "MaskCorr",
]

VALID_REFINE = Literal[
    "Refine",
]

VALID_RPN = Literal[
    "UPChannelRPN",
    "DepthwiseRPN",
    "MultiRPN",
]

VALID_TRAINING_DSET = Literal[
    "VID",
    "COCO",
    "DET",
    "YOUTUBEBB",
]
