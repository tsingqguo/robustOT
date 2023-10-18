from __future__ import annotations
from typing import Generic, NamedTuple, Type, TypeVar
from typing_extensions import Self

import numpy as np
import numpy.typing as npt
import torch

from msot.models import TModel
from msot.utils.dataship import DataCTR as DC, DataShip as DS
from msot.utils.region import Bbox, Point

from ..types import ScaledCrop
from .config import TrackConfig


class TrackerState(DS):
    # immutable
    z_feat: DC[torch.Tensor]
    """template feature"""

    size: DC[npt.NDArray[np.float_]]
    """size in format of (width, height)"""

    center: DC[Point[float]]
    """center postition"""

    @property
    def valid_names(self) -> set[str]:
        return super().valid_names | {"z_feat", "size", "center"}

    def __init__(self) -> None:
        self.z_feat = DC(
            is_shared=True,
            is_mutable=False,
            allow_unbound=True,
        )
        #
        self.size = DC(
            is_shared=False,
            is_mutable=True,
            allow_unbound=True,
        )
        self.center = DC(
            is_shared=False,
            is_mutable=True,
            allow_unbound=True,
        )

    def reset(self) -> None:
        self.z_feat.unbind()
        self.size.unbind()
        self.center.unbind()


class TrackResult(NamedTuple):
    output: Bbox
    """output in bbox or polygon format"""
    best_score: float | None


C = TypeVar("C", bound=TrackConfig)
S = TypeVar("S", bound=TrackerState)
R = TypeVar("R", bound=TrackResult)
T = TypeVar("T", bound="BaseTracker")


class BaseTracker(Generic[C, S, R]):
    _model: TModel
    _config: C
    _state: S
    _state_cls: Type[S]

    def __init__(
        self,
        model: TModel,
        config: C,
        state_cls: Type[S],
        state: S | None = None,
    ) -> None:
        self._model = model
        self._model.eval()
        self._config = config
        self._state_cls = state_cls
        if state is None:
            state = self._state_cls()
        self._state = state

    @property
    def model(self) -> TModel:
        return self._model

    @property
    def config(self) -> C:
        return self._config

    @property
    def state(self) -> S:
        if self._state is None:
            raise RuntimeError("state is not allowed to access")
        return self._state

    @property
    def is_cuda(self) -> bool:
        return next(self.model.parameters()).is_cuda

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def absorb(self, tracker) -> None:
        if not isinstance(tracker, self.__class__):
            raise RuntimeError("tracker type mismatch")
        if self.config != tracker.config or self.model != tracker.model:
            raise RuntimeError("tracker props mismatch")
        if not isinstance(tracker.state, self._state_cls):
            raise RuntimeError("tracker state type mismatch")
        self._state = tracker.state

        del tracker

    def state_reset(self) -> None:
        """reset state by calling `state.reset()`"""
        self._state.reset()

    def state_hard_reset(self) -> None:
        """reset state by re-assigning a new state instance"""
        self._state = self._state_cls()

    def fork(self) -> Self:
        state = self.state.smart_clone()
        return self.__class__(self.model, self.config, self._state_cls, state)

    def spawn(self, state: S) -> Self:
        """this method won't clone the state"""
        return self.__class__(self.model, self.config, self._state_cls, state)

    @classmethod
    def get_template(
        cls,
        cfg: TrackConfig,
        st: TrackerState,
        img: npt.NDArray[np.uint8],
        bbox: Bbox,
        device: torch.device,
    ) -> ScaledCrop:
        raise NotImplementedError

    @classmethod
    def get_search(
        cls,
        cfg: TrackConfig,
        st: TrackerState,
        img: npt.NDArray[np.uint8],
        device: torch.device,
    ) -> ScaledCrop:
        raise NotImplementedError

    def init_with_scaled_template(self, z_crop: torch.Tensor) -> None:
        z_feat = self.model.get_template_feature(z_crop)
        self.state.z_feat.update(z_feat)

    def init(
        self, img: npt.NDArray[np.uint8], bbox: Bbox
    ) -> tuple[ScaledCrop, None]:
        raise NotImplementedError

    def track_with_scaled_search(
        self, x_crop: torch.Tensor, scale: float, frame_size: tuple[int, ...]
    ) -> R:
        raise NotImplementedError

    def track(self, img: npt.NDArray[np.uint8]) -> tuple[ScaledCrop, R]:
        raise NotImplementedError
