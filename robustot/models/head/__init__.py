from .config import MaskCfg, MaskRefineCfg, RPNCfg
from .mask import MaskCorr, Refine
from .rpn import RPN, UPChannelRPN, DepthwiseRPN, MultiRPN


def build_rpn_head(cfg: RPNCfg) -> RPN:
    return {
        "UPChannelRPN": UPChannelRPN,
        "DepthwiseRPN": DepthwiseRPN,
        "MultiRPN": MultiRPN,
    }[cfg.type](**cfg.kwargs)


def build_mask_head(cfg: MaskCfg) -> MaskCorr:
    return {
        "MaskCorr": MaskCorr,
    }[cfg.type](**cfg.kwargs)


def build_refine_head(cfg: MaskRefineCfg) -> Refine:
    return {
        "Refine": Refine,
    }[cfg.type]()
