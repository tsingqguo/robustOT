from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntFlag
from typing import Generic, NamedTuple, Type, TypeAlias, TypeVar, TYPE_CHECKING

import numpy as np

from msot.trackers.base import BaseTracker
from msot.trackers.types import FrameImage, ScaledCrop
from msot.utils.color import Colors, fg
from msot.utils.dataship import DataCTR as DC, DataShip as DS, VertDCAC
from msot.utils.log import get_logger
from msot.utils.region import Region
from msot.utils.timer import Timer, TimerType

from ..roles import TDRoles

if TYPE_CHECKING:
    from ..historical import Historical

log = get_logger(__name__)

Input: TypeAlias = ScaledCrop | FrameImage

_PInfo = tuple[int, str | None]
"""`(exec_order, name | None)`"""

FrameImageVert: TypeAlias = VertDCAC[FrameImage, TDRoles, _PInfo]
ScaledCropVert: TypeAlias = VertDCAC[ScaledCrop, TDRoles, _PInfo]

A = TypeVar("A", bound="ProceesorAttrs")
C = TypeVar("C", bound="ProceesorConfig")


class Allow(IntFlag):
    """allow input process access to input data"""

    NONE = 0
    TRACKER = 1 << 0
    HISTORICAL = 1 << 1
    TARGET = 1 << 2


class ValidInput(IntFlag):
    FRAME_IMAGE = 1 << 0
    SCALED_CROP = 1 << 1


class ValidOutput(IntFlag):
    NONE = 0
    FRAME_IMAGE = 1 << 0
    SCALED_CROP = 1 << 1


@dataclass
class ProceesorAttrs(DS):
    cost: DC[float] = field(
        default_factory=lambda: DC(is_shared=False, allow_unbound=True)
    )

    @property
    def valid_names(self) -> set[str]:
        return super().valid_names | {"cost"}

    ...


@dataclass(kw_only=True, slots=True)
class ProceesorConfig:
    allow_access: Allow = field(default=Allow.NONE)
    timer: TimerType | None = None
    valid_input: ValidInput = field(
        default=ValidInput.FRAME_IMAGE | ValidInput.SCALED_CROP
    )
    valid_output: ValidOutput = field(
        default=ValidOutput.NONE
        | ValidOutput.FRAME_IMAGE
        | ValidOutput.SCALED_CROP
    )



@dataclass
class _ProcessTarget:
    crop_size: int
    """size of output crop"""


@dataclass
class ProcessTemplate(_ProcessTarget):
    gt: Region


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

        if Allow.TRACKER in self.config.allow_access:
            self._tracker = tracker
        else:
            self._tracker = None

        if Allow.HISTORICAL in self.config.allow_access:
            self._historical = historical
        else:
            self._historical = None

        if Allow.TARGET in self.config.allow_access:
            self._process_target = process_target
        else:
            self._process_target = None

    def __str__(self) -> str:
        return "{}: {} {}".format(
            self.__class__.__name__,
            self.name,
            self.config.__repr__(),
        )

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

    def forward(self, input: Input) -> Input | None:
        if isinstance(input, np.ndarray):
            if not ValidInput.FRAME_IMAGE in self.config.valid_input:
                raise RuntimeError(
                    f"{self.__class__.__name__} does not accept frame image as input"
                )
        elif isinstance(input, ScaledCrop):
            if not ValidInput.SCALED_CROP in self.config.valid_input:
                raise RuntimeError(
                    f"{self.__class__.__name__} does not accept scaled crop as input"
                )

        if self.config.timer is not None:
            timer = Timer(self.config.timer)
        else:
            timer = None

        out = self.process(input)

        if isinstance(out, np.ndarray):
            if not ValidOutput.FRAME_IMAGE in self.config.valid_output:
                raise RuntimeError(
                    f"{self.__class__.__name__} does not allow frame image as output"
                )
        elif isinstance(out, ScaledCrop):
            if not ValidOutput.SCALED_CROP in self.config.valid_output:
                raise RuntimeError(
                    f"{self.__class__.__name__} does not allow scaled crop as output"
                )
            # if out.crop.requires_grad and not out.crop.is_leaf: # TODO: TBD
            if out.crop.requires_grad:
                log.warning(
                    "processor {}'s output scaled crop has requires_grad==True".format(
                        self.__class__.__name__
                    )
                )

        elif out is None:
            if not ValidOutput.NONE in self.config.valid_output:
                raise RuntimeError(
                    f"{self.__class__.__name__} does not allow `None` as output"
                )

        if out is not None:
            if timer is not None:
                self.attrs.cost.unbind()  # FIXME: allow unbind-copy for DC
                self.attrs.cost.update(timer.elapsed)
        return out

    def reset(self) -> None:
        if hasattr(self, "attrs"):
            del self.attrs
        if hasattr(self, "_tracker"):
            del self._tracker
        if hasattr(self, "_historical"):
            del self._historical

    # def summary(self):
    #     raise NotImplementedError

    @staticmethod
    def from_file(fp: str) -> Processor:
        import pathlib
        from os import path
        from typing import Callable

        from msot.utils.config.helper import _get_sources_from_namespace

        if not path.exists(fp):
            raise FileNotFoundError(f"processor file {fp} not found")

        ext = pathlib.Path(fp).suffix[1:]
        if ext != "py":
            raise NotImplementedError(
                f"unsupported processor file extension: {ext}"
            )
        else:
            ns = {}
            exec(open(fp).read(), {}, ns)
            # TODO: sources support
            setup: Callable[[], None] | None = ns.get("setup")

        if setup is None:
            raise ValueError("No setup function found in namespace")
        else:
            proc = setup()

        if not isinstance(proc, Processor):
            raise TypeError("setup function must return `Processor`")

        log.info(
            "loading processor from file: {}\n{}".format(
                fp, fg(proc, Colors.Black).__str__()
            )
        )
        return proc


class _ProcessOutput(NamedTuple):
    input: Input
    img_vert: FrameImageVert
    scaled_crop_vert: ScaledCropVert
    processor_attrs: dict[str, ProceesorAttrs]


class InputProcess:
    processors: list[Processor]

    def __init__(self) -> None:
        self.processors = []

    @staticmethod
    def _append_to_vert(
        img_vert: FrameImageVert,
        scaled_crop_vert: ScaledCropVert,
        input: Input,
        p_info: _PInfo,
    ) -> None:
        if isinstance(input, np.ndarray):
            img_vert.append(input, p_info)
        elif isinstance(input, ScaledCrop):
            scaled_crop_vert.append(input, p_info)
        else:
            raise NotImplementedError

    def add(self, *processor: Processor) -> None:
        exist = set()
        for p in self.processors:
            exist.add(p.name)
            if hasattr(p, "_tracker"):
                raise ValueError("add after processors prepared")
        for p in processor:
            if p.name in exist:
                raise ValueError(f"processor name {p.name} already exists")

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
            attrs = attributes.get(p.name, None)
            if attrs is not None:
                attrs = attrs.smart_clone()
            p.init(
                attrs,
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

        self._append_to_vert(img_vert, scaled_crop_vert, input, (0, None))
        for ei, p in enumerate(self.processors):
            out = p.forward(input)
            if out is None:
                # skip
                ...
            else:
                input = out
                self._append_to_vert(
                    img_vert, scaled_crop_vert, input, p_info=(ei + 1, p.name)
                )

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
