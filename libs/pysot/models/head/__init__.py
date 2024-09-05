from pysot.models.head.mask import MaskCorr, Refine
from pysot.models.head.rpn import RPN, UPChannelRPN, DepthwiseRPN, MultiRPN
from pysot.types import VALID_MASK, VALID_REFINE, VALID_RPN  # TODO:


def get_rpn_head(name: VALID_RPN, **kwargs) -> RPN:
    RPNS = {
        "UPChannelRPN": UPChannelRPN,
        "DepthwiseRPN": DepthwiseRPN,
        "MultiRPN": MultiRPN,
    }
    return RPNS[name](**kwargs)


def get_mask_head(name: VALID_MASK, **kwargs) -> MaskCorr:
    MASKS = {
        "MaskCorr": MaskCorr,
    }
    return MASKS[name](**kwargs)


def get_refine_head(name: VALID_REFINE) -> Refine:
    REFINE = {
        "Refine": Refine,
    }
    return REFINE[name]()
