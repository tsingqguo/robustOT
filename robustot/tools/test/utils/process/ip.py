from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Generic, NamedTuple, Type, TypeVar

import numpy as np
import numpy.typing as npt
import torch

from msot.trackers.base import BaseTracker
from msot.trackers.types import ScaledCrop
from msot.utils.boxes import Bbox
from msot.utils.dataship import DataCTR as DC, DataShip as DS, VertDCAC

from ..historical import Historical
from ..roles import TDRoles

Input = ScaledCrop | npt.NDArray[np.uint8]

_ImgVert = VertDCAC[npt.NDArray[np.uint8], TDRoles, str | None]
_ScaledCropVert = VertDCAC[ScaledCrop, TDRoles, str | None]

A = TypeVar("A", bound="ProceesorAttrs")
C = TypeVar("C", bound="ProceesorConfig")


class Allow(IntEnum):
    """allow input process access to input data"""

    NONE = 0
    TRACKER = 1 << 0
    HISTORICAL = 1 << 1
    TARGET = 1 << 2


@dataclass
class ProceesorAttrs(DS):
    # allow_access: DC[int] = field(
    #     default_factory=lambda: DC(Allow.NONE, is_shared=True)
    # )

    # @property
    # def valid_names(self) -> set[str]:
    #     return super().valid_names | {"allow_access"}

    ...


@dataclass
class ProceesorConfig:
    allow_access = Allow.NONE


@dataclass
class _ProcessTarget:
    crop_size: int
    """size of output crop"""


@dataclass
class ProcessTemplate(_ProcessTarget):
    bbox: Bbox


@dataclass
class ProcessSearch(_ProcessTarget):
    ...


class Processor(Generic[A, C]):
    name: str
    attrs: A
    config: C
    _attrs_cls: Type[A]
    _tracker: BaseTracker | None
    _historical: Historical | None
    _process_target: _ProcessTarget | None

    def __init__(
        self,
        name: str,
        config: C,
        attrs_cls: Type[A],
    ) -> None:
        self.name = name
        self.config = config
        self._attrs_cls = attrs_cls
        # self.attrs =  # unbound
        # self._tracker =  # unbound
        # self._historical =  # unbound

    def init(
        self,
        attrs: A | None,
        tracker: BaseTracker,
        historical: Historical,
        process_target: _ProcessTarget | None = None,
    ) -> None:
        if attrs is None:
            self.attrs = self._attrs_cls()
        else:
            # TODO: fork dilemma
            self.attrs = attrs  # IMPORTANT: deepcopy before Test::action_track

        if Allow.TRACKER & self.config.allow_access:
            self._tracker = tracker
        else:
            self._tracker = None

        if Allow.HISTORICAL & self.config.allow_access:
            self._historical = historical
        else:
            self._historical = None

        if Allow.TARGET & self.config.allow_access:
            self._process_target = process_target
        else:
            self._process_target = None

    @property
    def tracker(self) -> BaseTracker:
        if self._tracker is None:
            raise ValueError("tracker is not allowed to access")
        return self._tracker

    @property
    def historical(self) -> Historical:
        if self._historical is None:
            raise ValueError("historical is not allowed to access")
        return self._historical

    @property
    def process_target(self) -> _ProcessTarget:
        if self._process_target is None:
            raise ValueError("process_target is not allowed to access")
        return self._process_target

    def process(self, input: Input) -> Input | None:
        raise NotImplementedError

    def reset(self) -> None:
        if hasattr(self, "attrs"):
            del self.attrs
        if hasattr(self, "_tracker"):
            del self._tracker
        if hasattr(self, "_historical"):
            del self._historical

    # def summary(self):
    #     raise NotImplementedError


class _ProcessOutput(NamedTuple):
    input: Input
    img_vert: _ImgVert
    scaled_crop_vert: _ScaledCropVert
    processor_attrs: dict[str, ProceesorAttrs]


class InputProcess:
    processors: list[Processor]

    def __init__(self) -> None:
        self.processors = []

    @staticmethod
    def _append_to_vert(
        img_vert: _ImgVert,
        scaled_crop_vert: _ScaledCropVert,
        input: Input,
        p_name: str | None,
    ) -> None:
        if isinstance(input, np.ndarray):
            img_vert.append(input, p_name)
        elif isinstance(input, ScaledCrop):
            scaled_crop_vert.append(input, p_name)
        else:
            raise NotImplementedError

    def add(self, *processor: Processor) -> None:
        for p in self.processors:
            if hasattr(p, "_tracker"):
                raise ValueError("add after processors prepared")
        for p in processor:
            self.processors.append(p)
            p.reset()

    def prepare(
        self,
        attributes: dict[str, ProceesorAttrs],
        tracker: BaseTracker,
        historical: Historical,
        process_target: _ProcessTarget,
    ) -> None:
        for p in self.processors:
            p.init(
                deepcopy(attributes.get(p.name, None)),
                tracker,
                historical,
                process_target,
            )

    def execute(self, input: Input) -> _ProcessOutput:
        img_vert = VertDCAC(
            TDRoles,
            lambda role: role >= TDRoles.TRACKER,
            lambda role: role >= TDRoles.DEBUG,
        )
        scaled_crop_vert = VertDCAC(
            TDRoles,
            lambda role: role >= TDRoles.TRACKER,
            lambda role: role >= TDRoles.DEBUG,
        )

        self._append_to_vert(img_vert, scaled_crop_vert, input, None)
        for p in self.processors:
            out = p.process(input)
            if out is None:
                # skip
                ...
            else:
                input = out
                self._append_to_vert(img_vert, scaled_crop_vert, input, p.name)

        attrs = {}
        for p in self.processors:
            attrs[p.name] = p.attrs

        self.reset()

        return _ProcessOutput(
            input=input,
            img_vert=img_vert,
            scaled_crop_vert=scaled_crop_vert,
            processor_attrs=attrs,
        )

    def reset(self) -> None:
        for p in self.processors:
            p.reset()
