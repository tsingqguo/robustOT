import torch.nn as nn
from torch import Tensor
from pysot.models.head.rpn import RPN
from typing import Any, Optional, TypedDict, Union


class _MB_Track_R(TypedDict):
    cls: Tensor
    loc: Tensor
    mask: Optional[Any]  # TODO:


class MB_M_Input(TypedDict):
    template: Tensor
    search: Tensor
    label_cls: Tensor
    label_loc: Tensor
    label_loc_weight: Tensor


class MB_M_Output(TypedDict):
    total_loss: Tensor
    cls_loss: Union[Tensor, float]  # TODO:
    loc_loss: Tensor
    mask_loss: Optional[Tensor]
    _cls: Tensor
    _loc: Tensor


class ModelBuilder(nn.Module):
    backbone: nn.Module
    neck: nn.Module
    rpn_head: RPN
    zf: Tensor
    """
    saved feature of the template image
    """

    def template(self, z: Tensor) -> None:
        raise NotImplementedError

    def track(self, x: Tensor) -> _MB_Track_R:
        raise NotImplementedError

    def mask_refine(self, pos):
        raise NotImplementedError

    def log_softmax(self, cls: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, data: MB_M_Input) -> MB_M_Output:
        raise NotImplementedError
