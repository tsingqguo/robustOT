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
    ScaledCrop,
    ValidOutput,
)


@dataclass
class _NoProcessConfig(ProceesorConfig):
    valid_output = ValidOutput.FRAME_IMAGE | ValidOutput.SCALED_CROP


class NoProcess(Processor[ProceesorAttrs, _NoProcessConfig]):
    def __init__(
        self,
        name: str | None,
    ) -> None:
        super().__init__(name or "noproc", _NoProcessConfig(), ProceesorAttrs)

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
    valid_output = ValidOutput.NONE | ValidOutput.SCALED_CROP


class DefaultCrop(Processor[DefaultCropAttrs, DefaultCropConfig]):
    def __init__(
        self,
        name: str | None,
        config: DefaultCropConfig,
    ) -> None:
        super().__init__(name or "default_crop", config, DefaultCropAttrs)

    def process(self, input: Input) -> ScaledCrop | None:
        if not isinstance(input, ScaledCrop):
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
            if type(self.process_target) is ProcessTemplate:
                if not DefaultCropOn.TEMPLATE in self.config.crop_on:
                    return None
            elif type(self.process_target) is ProcessSearch:
                if not DefaultCropOn.SEARCH in self.config.crop_on:
                    return None

            raise RuntimeError(
                "DefaultCrop does not accept scaled crop as input; try to config `crop_on`"
            )
