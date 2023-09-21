from msot.utils.dataship import DataCTR as DC, DataCTRAC as DCAC

from .utils.roles import TDRoles


class SequenceInfo:
    _name: str | None
    attributes: DCAC[dict, TDRoles]
    size: DC[tuple[int, ...]]

    def __init__(
        self,
        name: str | None,
        attributes: dict | None,
    ):
        self._name = name
        self.attributes = DCAC(
            TDRoles,
            lambda role: role >= TDRoles.ANALYSIS,
            attributes or {},
            is_shared=True,
            is_mutable=False,
            allow_unbound=False,
        )
        self.size = DC(is_shared=True, is_mutable=False, allow_unbound=True)

    @property
    def name(self) -> str:
        return self._name or "unknown_seq"

    def check_size(self, size: tuple[int, ...]) -> None:
        if self.size.is_unbound():
            raise RuntimeError("Sequence size is unknown")
        if self.size.val != size:
            raise RuntimeError("Sequence size mismatch")
