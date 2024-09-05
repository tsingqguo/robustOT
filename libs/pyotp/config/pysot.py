from pysot.types import (
    VALID_ADJUST,
    VALID_BACKBONE,
    VALID_MASK,
    VALID_REFINE,
    VALID_RPN,
    VALID_TRACKER,
    VALID_TRAINING_DSET,
)
from pyotp.utils.config import Config
from pyotp.utils.option import Option, Some, NONE

from typing import (
    Any,
    Dict,
    List,
    Literal,
)


class _Cfg_Arbitrary_Kwargs(Config):
    pass


_Cfg_LrType = Literal[
    "log",  # pysot.utils.lr_scheduler.LogScheduler
    "step",  # pysot.utils.lr_scheduler.StepScheduler
    "multi-step",  # pysot.utils.lr_scheduler.MultiStepScheduler
    "linear",  # pysot.utils.lr_scheduler.LinearStepScheduler
    "cos",  # pysot.utils.lr_scheduler.CosStepScheduler
]
"""pysot.utils.lr_scheduler.LRs"""


class _Cfg_Train_Lr_Warmup(Config):
    warmup: bool = True
    type: _Cfg_LrType = "step"
    epoch: int = 5

    # kwargs = _Cfg_Arbitrary_Kwargs()
    kwargs: Dict[str, Any] = {}


class _Cfg_Train_Lr(Config):
    type: _Cfg_LrType = "log"
    # TODO:
    # kwargs = _Cfg_Arbitrary_Kwargs()
    kwargs: Dict[str, Any] = {}
    warmup = _Cfg_Train_Lr_Warmup()


class _Cfg_Train_Partial(Config):
    backbone: bool = True
    neck: bool = True
    rpn_head: bool = True
    mask_head: bool = True
    refine_head: bool = True


class _Cfg_Train(Config):
    thr_high: float = 0.6
    """Positive anchor threshold"""

    thr_low: float = 0.3
    """Negative anchor threshold"""

    neg_num: int = 16
    """Number of negative"""

    pos_num: int = 16
    """Number of positive"""

    total_num: int = 64
    """Number of anchors per images"""

    exemplar_size: int = 127

    search_size: int = 255

    base_size = 8

    output_size = 25

    # RESUME = ""
    resume: Option[str] = NONE

    # PRETRAINED = ""
    pretrained: Option[str] = NONE

    # LOG_DIR = "./logs"
    log_dir: Option[str] = NONE

    snapshot_dir = "./snapshot"

    epoch: int = 20

    start_epoch: Option[int] = NONE

    batch_size: int = 32

    num_workers: int = 1

    momentum: float = 0.9

    weight_decay: float = 1e-4

    cls_weight: float = 1.2

    loc_weight: float = 1.0

    mask_weight: float = 1.0

    print_freq: int = 20

    log_grads: bool = False

    grad_clip: float = 10.0

    base_lr: float = 5e-3

    lr = _Cfg_Train_Lr()

    partial = _Cfg_Train_Partial()


class _Cfg_Dataset_Augmentation(Config):
    shift: int = 0
    """
    Random shift see [SiamPRN++](https://arxiv.org/pdf/1812.11703) for detail discussion
    """

    scale: float = 0.0

    blur: float = 0.0

    flip: float = 0.0

    color: float = 1.0

    gn_mean: float = 0.0

    gn_var: float = 0.0


class _Cfg_Dataset_Aug_Template(_Cfg_Dataset_Augmentation):
    shift = 4
    scale = 0.05


class _Cfg_Dataset_Aug_Search(_Cfg_Dataset_Augmentation):
    shift = 64
    scale = 0.18


class _Cfg_DatasetCfg(Config):
    root: Option[str] = NONE
    """path of dataset"""

    anno: Option[str] = NONE
    """path of train.json"""

    frame_range: int = 1

    num_use: int = -1
    """
    repeat until reach `num_use`;
    `-1` for use all not repeat
    """

    use_hdf5: bool = False


#
DATASET_VID = _Cfg_DatasetCfg()
DATASET_VID.root = Some("$TRAIN_DSET_PATH/vid/crop511")
DATASET_VID.anno = Some("$TRAIN_DSET_PATH/vid/train.json")
DATASET_VID.frame_range = 100
DATASET_VID.num_use = 100000
DATASET_VID.use_hdf5 = True
#
DATASET_COCO = _Cfg_DatasetCfg()
DATASET_COCO.root = Some("$TRAIN_DSET_PATH/coco/crop511")
DATASET_COCO.anno = Some("$TRAIN_DSET_PATH/coco/train2017.json")
DATASET_COCO.frame_range = 1
DATASET_COCO.num_use = -1
#
DATASET_DET = _Cfg_DatasetCfg()
DATASET_DET.root = Some("$TRAIN_DSET_PATH/det/crop511")
DATASET_DET.anno = Some("$TRAIN_DSET_PATH/det/train.json")
DATASET_DET.frame_range = 1
DATASET_DET.num_use = -1
#
DATASET_YOUTUBEBB = _Cfg_DatasetCfg()
DATASET_YOUTUBEBB.root = Some("$TRAIN_DSET_PATH/yt_bb/crop511")
DATASET_YOUTUBEBB.anno = Some("$TRAIN_DSET_PATH/yt_bb/train.json")
DATASET_YOUTUBEBB.frame_range = 3
DATASET_YOUTUBEBB.num_use = -1
DATASET_YOUTUBEBB.use_hdf5 = True


class _Cfg_Dataset(Config):
    template: _Cfg_Dataset_Augmentation = _Cfg_Dataset_Aug_Template()

    search: _Cfg_Dataset_Augmentation = _Cfg_Dataset_Aug_Search()

    neg: float = 0.2
    """
    Sample Negative pair see [DaSiamRPN](https://arxiv.org/pdf/1808.06048) for detail discussion
    """

    gray: float = 0.0
    """improve tracking performance for otb100"""

    names: List[VALID_TRAINING_DSET] = ["VID", "COCO", "DET", "YOUTUBEBB"]

    VID: _Cfg_DatasetCfg = DATASET_VID

    COCO: _Cfg_DatasetCfg = DATASET_COCO

    DET: _Cfg_DatasetCfg = DATASET_DET

    YOUTUBEBB: _Cfg_DatasetCfg = DATASET_YOUTUBEBB

    videos_per_epoch: int = 600000


class _Cfg_Backbone(Config):
    # TODO:
    type: VALID_BACKBONE = "resnet50"
    """
    Backbone type, current only support resnet18,34,50;alexnet;mobilenet
    """

    # kwargs = _Cfg_Arbitrary_Kwargs()
    kwargs: Dict[str, Any] = {}

    pretrained: Option[str] = NONE
    """Pretrained backbone weights"""

    train_layers: List[str] = ["layer2", "layer3", "layer4"]
    """Train layers"""

    layers_lr: float = 0.1
    """Layer LR"""

    train_epoch: int = 10
    """Switch to train layer"""


CFG_BACKBONE = _Cfg_Backbone()


class _Cfg_Adjust(Config):
    adjust: bool = True

    # kwargs = _Cfg_Arbitrary_Kwargs()
    kwargs: Dict[str, Any] = {}

    type: VALID_ADJUST = "AdjustAllLayer"
    """Adjust layer type"""


CFG_ADJUST = _Cfg_Adjust()


class _Cfg_Rpn(Config):
    type: VALID_RPN = "MultiRPN"

    # kwargs = _Cfg_Arbitrary_Kwargs()
    kwargs: Dict[str, Any] = {}


CFG_RPN = _Cfg_Rpn()


class _Cfg_Mask(Config):
    mask: bool = False
    """Whether to use mask generate segmentation"""

    type: VALID_MASK = "MaskCorr"

    # kwargs = _Cfg_Arbitrary_Kwargs()
    kwargs: Dict[str, Any] = {}


CFG_MASK = _Cfg_Mask()


class _Cfg_Mask_Refine(Config):
    refine: bool = False
    """Mask refine"""

    type: VALID_REFINE = "Refine"
    """Refine type"""


CFG_MASK_REFINE = _Cfg_Mask_Refine()


class _Cfg_Anchor(Config):
    stride: int = 8
    """Anchor stride"""

    ratios: List[float] = [0.33, 0.5, 1, 2, 3]
    """Anchor ratios"""

    scales: List[int] = [8]
    """Anchor scales"""

    @property
    def anchor_num(self) -> int:
        """Anchor number"""
        return len(self.ratios) * len(self.scales)


CFG_ANCHOR = _Cfg_Anchor()
# CFG_ANCHOR.anchor_num = len(CFG_ANCHOR.ratios) * len(CFG_ANCHOR.sacles)


class _Cfg_Track(Config):
    # FIXME:
    # type: Literal["SiamRPNTracker"] = "SiamRPNTracker"
    type: VALID_TRACKER = "SiamRPNTracker"

    penalty_k: float = 0.04
    """Scale penalty"""

    window_influence: float = 0.44
    """Window influence"""

    lr: float = 0.4
    """Interpolation learning rate"""

    exemplar_size: int = 127
    """Exemplar size; default `127`"""

    instance_size: int = 255
    """Instance size; default `255`"""

    base_size: int = 8
    """Base size; default 8"""

    context_amount: float = 0.5
    """Context amount; default 0.5"""

    lost_instance_size: int = 831
    """Long term lost search size"""

    confidence_low: float = 0.85
    """Long term confidence low"""

    confidence_high: float = 0.998
    """Long term confidence high"""

    mask_threshold: float = 0.3
    """Mask threshold"""

    mask_output_size: int = 127
    """Mask output size"""


CFG_TRACK = _Cfg_Track()


class PYSOT_Config(Config):
    meta_arc: str = "siamrpn_r50_l234_dwxcorr"

    cuda: bool = True

    train: _Cfg_Train = _Cfg_Train()

    dataset: _Cfg_Dataset = _Cfg_Dataset()

    # --------------------
    # | Backbone options |
    # --------------------
    backbone: _Cfg_Backbone = CFG_BACKBONE

    # ------------------------
    # | Adjust layer options |
    # ------------------------
    adjust: _Cfg_Adjust = CFG_ADJUST

    # ---------------
    # | RPN options |
    # ---------------
    rpn: _Cfg_Rpn = CFG_RPN

    # ----------------
    # | Mask options |
    # ----------------
    mask: _Cfg_Mask = CFG_MASK

    refine: _Cfg_Mask_Refine = CFG_MASK_REFINE

    # ------------------
    # | Anchor options |
    # ------------------
    anchor: _Cfg_Anchor = CFG_ANCHOR

    # -------------------
    # | Tracker options |
    # -------------------
    track: _Cfg_Track = CFG_TRACK


CFG = PYSOT_Config()
