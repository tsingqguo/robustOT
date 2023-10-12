from __future__ import annotations
import copy
from enum import Enum
from typing import Callable, Generic, Type, TypeVar
from typing_extensions import Self

import numpy as np
import torch

R = TypeVar("R", bound=Enum)
T = TypeVar("T")
U = TypeVar("U")
X = TypeVar("X")
Y = TypeVar("Y")


class DataUnbound:
    ...


class DataCTR(Generic[T]):
    _val: T | DataUnbound
    _is_shared: bool
    _is_mutable: bool
    _allow_unbound: bool = False

    def __init__(
        self,
        val: T | DataUnbound = DataUnbound(),
        is_shared: bool = False,
        is_mutable: bool = False,
        allow_unbound: bool = False,
    ) -> None:
        self._val = val
        self._is_shared = is_shared
        self._is_mutable = is_mutable
        self._allow_unbound = allow_unbound
        if isinstance(val, DataUnbound) and not allow_unbound:
            raise RuntimeError("Unbound init data")

    def _get_value(self) -> T:
        if isinstance(self._val, DataUnbound):
            raise RuntimeError("Unbound data")
        return self._val

    @property
    def val(self) -> T:
        return self._get_value()

    @property
    def value(self) -> T:
        return self.val

    def get(self, *, default: T | U | DataUnbound = DataUnbound()) -> T | U:
        if isinstance(self._val, DataUnbound):
            if isinstance(default, DataUnbound):
                raise RuntimeError("Unbound data")
            else:
                return default
        else:
            return self._val

    @property
    def is_shared(self) -> bool:
        return self._is_shared

    def is_unbound(self) -> bool:
        return isinstance(self._val, DataUnbound)

    def _val_as_ref(self) -> T | DataUnbound:
        return self._val

    def _val_clone(self) -> T | DataUnbound:
        if isinstance(self._val, torch.Tensor):
            v = self._val.clone()
        elif isinstance(self._val, np.ndarray):
            v = self._val.copy()
        else:
            v = copy.deepcopy(self._val)
        return v

    def as_ref(self) -> Self:
        return self.__class__(
            self._val_as_ref(),
            self.is_shared,
            is_mutable=False,
            allow_unbound=self._allow_unbound,
        )

    def clone(self) -> Self:
        return self.__class__(
            self._val_clone(),
            self._is_shared,
            self._is_mutable,
            self._allow_unbound,
        )

    def update(self, val: T):
        if isinstance(self._val, DataUnbound):
            self._val = val
        elif self._is_mutable:
            self._val = val
        else:
            raise RuntimeError("Immutable")

    def unbind(self):
        if self._allow_unbound:
            self._val = DataUnbound()
        else:
            raise RuntimeError("Unbound is not allowed")

    def smart_clone(self) -> Self:
        if self._is_shared:
            return self.as_ref()
        else:
            return self.clone()


class DataCTRAC(DataCTR[T], Generic[T, R]):
    _roles: Type[R]
    _access_validator: Callable[[R], bool]

    def __init__(
        self,
        role_cls: Type[R],
        validator: Callable[[R], bool],
        val: T | DataUnbound = DataUnbound(),
        is_shared: bool = False,
        is_mutable: bool = False,
        allow_unbound: bool = False,
    ) -> None:
        super().__init__(
            val,
            is_shared=is_shared,
            is_mutable=is_mutable,
            allow_unbound=allow_unbound,
        )
        self._roles = role_cls
        self._access_validator = validator

    @property
    def val(self):
        raise RuntimeError("Locked DataCTR, using `get(role: R)` instead")

    def get(
        self, role: R, default: T | U | DataUnbound = DataUnbound()
    ) -> T | U:
        if self._access_validator(role):
            return super().get(default=default)
        else:
            raise RuntimeError(f"Access denied for role {role}")

    def as_ref(self) -> Self:
        return self.__class__(
            self._roles,
            self._access_validator,
            self._val_as_ref(),
            self.is_shared,
            is_mutable=False,
            allow_unbound=self._allow_unbound,
        )

    def clone(self) -> Self:
        return self.__class__(
            self._roles,
            self._access_validator,
            self._val_clone(),
            self._is_shared,
            self._is_mutable,
            self._allow_unbound,
        )


class VertDCAC(Generic[X, R, Y]):
    _vert: list[tuple[DataCTRAC[X, R], Y]]

    _role: Type[R]
    _validator: Callable[[R], bool]
    _archived_validator: Callable[[R], bool]

    def __init__(
        self,
        role: Type[R],
        validator: Callable[[R], bool],
        archived_validator: Callable[[R], bool],
    ) -> None:
        self._vert = []
        self._role = role
        self._validator = validator
        self._archived_validator = archived_validator

    def _create_dcac(self, item: X) -> DataCTRAC[X, R]:
        return DataCTRAC(
            self._role,
            self._validator,
            item,
            is_shared=True,
            is_mutable=False,
            allow_unbound=False,
        )

    def append(self, item: X, extra: Y):
        self._vert.append((self._create_dcac(item), extra))
        if len(self) > 1:
            self._vert[-2][0]._access_validator = self._archived_validator

    def _append(self, item: DataCTRAC[X, R], extra: Y):
        self._vert.append((item, extra))
        item._access_validator = self._validator
        if len(self) > 1:
            self._vert[-2][0]._access_validator = self._archived_validator

    def smart_clone(self) -> Self:
        vd = self.__class__(
            self._role,
            self._validator,
            self._archived_validator,
        )
        for v, e in self._vert:
            vd._append(v.smart_clone(), e)  # FIXME: always shallow copy Y
        return vd

    def __len__(self) -> int:
        return len(self._vert)


class DataShip:
    @property
    def valid_names(self) -> set[str]:
        return set()

    def smart_clone(self) -> Self:
        ds = self.__class__()
        for k in self.valid_names:
            v = getattr(self, k)
            if isinstance(v, DataCTR):
                v = v.smart_clone()
                setattr(ds, k, v)
            elif isinstance(v, VertDCAC):
                v = v.smart_clone()
                setattr(ds, k, v)
            elif isinstance(v, DataShip):
                v = v.smart_clone()
                setattr(ds, k, v)
        return ds


if __name__ == "__main__":
    import psutil
    import os

    is_shared = True

    def mem_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss

    def mem_usage_mb():
        return mem_usage() / (1024**2)

    def test_clone(n):
        tensor = torch.randn(1000, 1000)
        data = DataCTR(tensor, is_shared=is_shared)
        print(f"Memory usage before cloning: {mem_usage_mb():.2f} MB")

        _ = [data.smart_clone() for _ in range(n)]
        print(f"Memory usage after cloning {n} times: {mem_usage_mb():.2f} MB")

    test_clone(100)
