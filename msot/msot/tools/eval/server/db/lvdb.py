import asyncio
from typing_extensions import Self

import plyvel

from .utils import FIFODict


class DB:
    db: plyvel.DB
    _locks: FIFODict[str, asyncio.Lock] | None = None

    @classmethod
    def from_path(cls, db_path: str) -> Self:
        db = cls.__new__(cls)
        db.db = plyvel.DB(db_path, create_if_missing=True)
        return db

    @classmethod
    def from_db(cls, db: plyvel.DB, split) -> Self:
        sub_db = cls.__new__(cls)
        if isinstance(split, str):
            if not split.endswith("/"):
                split += "/"
            split = split.encode()
        sub_db.db = db.prefixed_db(split)
        return sub_db

    @staticmethod
    def _encode(v: str) -> bytes:
        return v.encode()

    @staticmethod
    def _decode(v: bytes) -> str:
        return v.decode()

    @property
    def locks(self) -> FIFODict[str, asyncio.Lock]:
        if self._locks is None:
            self._locks = FIFODict(100)
        return self._locks

    async def get(self, key: str):
        return self.db.get(self._encode(key))

    async def put(self, key: str, value: bytes):
        async with self.locks.setdefault(key, asyncio.Lock()):
            return self.db.put(self._encode, value)

    async def delete(self, key: str):
        async with self.locks.setdefault(key, asyncio.Lock()):
            return self.db.delete(key.encode())
