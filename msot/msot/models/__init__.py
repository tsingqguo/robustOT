from __future__ import annotations
from typing import NamedTuple
from typing_extensions import Self

import torch
import torch.nn as nn

from .backbone import build_backbone
from .config import ModelConfig
from .neck import build_neck
from .head import (
    build_rpn_head,
    build_mask_head,
    build_refine_head,
)


class TModelResult(NamedTuple):
    cls: torch.Tensor
    loc: torch.Tensor
    mask: torch.Tensor | None


class TModel(nn.Module):
    neck: nn.Module | None
    """only available when adjust is enabled"""

    _config: ModelConfig
    """feature of z-crop"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self._config = config
        self.backbone = build_backbone(self.config.backbone)

        if self.config.adjust.adjust:
            self.neck = build_neck(self.config.adjust)
        else:
            self.neck = None

        self.rpn_head = build_rpn_head(self.config.rpn)

        if self.config.mask.mask:
            raise NotImplementedError
            self.mask_head = build_mask_head(self.config.mask)

            if self.config.refine.refine:
                self.refine_head = build_refine_head(self.config.refine)
            else:
                self.refine_head = None
        else:
            self.mask_head = None

    @classmethod
    def load_from_config(
        cls, config: ModelConfig, device: torch.device
    ) -> Self:
        model = cls(config)
        model = model.load_pretrained(model, device)
        return model

    @property
    def config(self) -> ModelConfig:
        return self._config

    # TODO: abs
    @staticmethod
    def load_pretrained(
        model: TModel,
        device: torch.device,
        pretrained: str | None = None,
    ):
        from msot.libs.pysot.utils.model_load import load_pretrain

        if pretrained is None:
            pretrained = model.config.pretrained

        if pretrained is None:
            raise ValueError("no pretrained model provided")

        model = load_pretrain(model, pretrained)
        model = model.to(device)
        return model

    def get_template_feature(self, z: torch.Tensor) -> torch.Tensor:
        f = self.backbone(z)
        if self.mask_head is not None:
            f = f[-1]
        if self.neck is not None:
            f = self.neck(f)
        return f

    def track(self, z_f: torch.Tensor, x: torch.Tensor) -> TModelResult:
        x_f = self.backbone(x)
        if self.mask_head is not None:
            raise NotImplementedError
        if self.neck is not None:
            x_f = self.neck(x_f)
        cls, loc = self.rpn_head(z_f, x_f)
        if self.mask_head is not None:
            raise NotImplementedError
        else:
            mask = None
        return TModelResult(cls, loc, mask)
