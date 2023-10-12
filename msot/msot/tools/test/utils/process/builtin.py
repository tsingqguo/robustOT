from dataclasses import dataclass
from enum import Flag, auto

from .ip import (
    Allow,
    Input,
    ProcessSearch,
    ProcessTemplate,
    Processor,
    ProceesorAttrs,
    ProceesorConfig,
)


class NoProcess(Processor[ProceesorAttrs, ProceesorConfig]):
    def __init__(
        self,
        name: str | None,
    ) -> None:
        super().__init__(name or "noproc", ProceesorConfig(), ProceesorAttrs)

    def process(self, input: Input) -> Input:
        return input


@dataclass
class DefaultCropAttrs(ProceesorAttrs):
    allow_access = Allow.TRACKER | Allow.TARGET


class DefaultCropOn(Flag):
    TEMPLATE = auto()
    SEARCH = auto()


@dataclass
class DefaultCropConfig(ProceesorConfig):
    allow_access = Allow.TRACKER | Allow.TARGET
    crop_on = DefaultCropOn.TEMPLATE | DefaultCropOn.SEARCH


class DefaultCrop(Processor[DefaultCropAttrs, DefaultCropConfig]):
    def __init__(
        self,
        name: str | None,
        config: DefaultCropConfig,
    ) -> None:
        super().__init__(name or "default_crop", config, DefaultCropAttrs)

    def process(self, input: Input) -> Input | None:
        import numpy as np

        if isinstance(input, np.ndarray):
            if type(self.process_target) is ProcessTemplate:
                if not DefaultCropOn.TEMPLATE in self.config.crop_on:
                    return None

                return self.tracker.get_template(
                    self.tracker.config,
                    self.tracker.state,
                    input,
                    self.process_target.bbox,
                    self.tracker.device,
                )
            elif type(self.process_target) is ProcessSearch:
                if not DefaultCropOn.SEARCH in self.config.crop_on:
                    return None

                return self.tracker.get_search(
                    self.tracker.config,
                    self.tracker.state,
                    input,
                    self.tracker.device,
                )
            else:
                raise RuntimeError("invalid process target type")
        else:
            raise RuntimeError("DefaultCrop only accepts uncropped input")
