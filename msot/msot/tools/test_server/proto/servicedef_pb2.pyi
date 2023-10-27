from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class FeedFrameRequest(_message.Message):
    __slots__ = ["id", "frame", "region"]
    ID_FIELD_NUMBER: _ClassVar[int]
    FRAME_FIELD_NUMBER: _ClassVar[int]
    REGION_FIELD_NUMBER: _ClassVar[int]
    id: str
    frame: bytes
    region: bytes
    def __init__(self, id: _Optional[str] = ..., frame: _Optional[bytes] = ..., region: _Optional[bytes] = ...) -> None: ...

class FeedFrameResponse(_message.Message):
    __slots__ = ["predict"]
    PREDICT_FIELD_NUMBER: _ClassVar[int]
    predict: bytes
    def __init__(self, predict: _Optional[bytes] = ...) -> None: ...

class FinishRequest(_message.Message):
    __slots__ = ["id"]
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...

class FinishResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...
