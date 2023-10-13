from dataclasses import dataclass, field

import numpy as np
import torch

from msot.tools.test.utils.process import (
    Allow,
    Input,
    ProcessTemplate,
    Processor,
    ProceesorAttrs,
    ProceesorConfig,
)
from msot.trackers.types import ScaledCrop
from msot.utils.dataship import DataCTR as DC

from robustot.libs.RTAA.perturb import perturb_forward
from robustot.libs.RTAA.utils import where


@dataclass
class RTAAConfig(ProceesorConfig):
    eps: int = 10
    iteration: int = 10
    og_attack: bool = False
    allow_access = Allow.HISTORICAL | Allow.TRACKER | Allow.TARGET


@dataclass
class RTAAAttrs(ProceesorAttrs):
    counter: DC[int] = field(
        default_factory=lambda: DC(0, is_shared=False, is_mutable=True)
    )
    prev_atk: DC[torch.Tensor] = field(
        default_factory=lambda: DC(
            is_shared=False, is_mutable=True, allow_unbound=True
        )
    )

    @property
    def valid_names(self) -> set[str]:
        return super().valid_names | {"counter", "prev_atk"}


class RTAAProcessor(Processor[RTAAAttrs, RTAAConfig]):
    def __init__(
        self,
        name: str,
        config: RTAAConfig,
    ) -> None:
        super().__init__(name, config, RTAAAttrs)

    def process(self, input: Input) -> Input | None:
        if isinstance(input, np.ndarray):
            raise RuntimeError("RTAAProcessor only accepts scaled-crop input")

        if type(self.process_target) is ProcessTemplate:
            return None

        if self.attrs.counter.get() % 30 == 0:
            self.attrs.prev_atk.unbind()

        if self.attrs.prev_atk.is_unbound():
            x_crop_init = input.crop
        else:
            x_crop_init = input.crop + self.attrs.prev_atk.get()
        x_crop_init = torch.clamp(x_crop_init, 0, 1)

        x = input.crop.detach().clone().requires_grad_(True)
        x_adv = x_crop_init.detach().clone().requires_grad_(True)

        bbox = self.historical.last.unwrap().result.pred.get()

        if self.config.og_attack:
            raise NotImplementedError

        alpha = self.config.eps * 1.0 / self.config.iteration

        for _ in range(self.config.iteration):
            if self.config.og_attack:
                raise NotImplementedError
            else:
                out = perturb_forward(
                    self.tracker,
                    x_adv,
                    input.size.scale,
                    bbox,
                )

            # reset gradients
            self.tracker.model.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            out["total_loss"].backward(retain_graph=True)

            if x_adv.grad is None:
                raise RuntimeError

            # # -- ++ -- ATTACK -- ++ --
            adv_grad = where(
                (x_adv.grad > 0) | (x_adv.grad < 0),
                x_adv.grad,
                torch.tensor(0),
            )
            adv_grad = torch.sign(adv_grad)
            x_adv = x_adv - alpha * adv_grad
            # # -- ++ -- ++ -- ++ -- ++ --

            x_adv = torch.clamp(
                x_adv, x - self.config.eps, x + self.config.eps
            )
            x_adv = torch.clamp(x_adv, 0, 255)
            x_adv = x_adv.detach().clone().requires_grad_(True)

        self.attrs.prev_atk.update((x_adv - input.crop).detach())

        return ScaledCrop(x_adv, input.size)
