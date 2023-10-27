import pickle
from typing import TypeAlias

import grpc.aio

from msot.utils.log import get_logger

from ..test import Test as _Test, TestAttributes, TestServer
from ..test.args import Args

from .proto.servicedef_pb2 import (
    FeedFrameRequest,
    FeedFrameResponse,
    FinishRequest,
    FinishResponse,
)
from .proto import servicedef_pb2_grpc

log = get_logger(__name__)

Test: TypeAlias = _Test[Args, TestAttributes]


class TestServicer(servicedef_pb2_grpc.MSOTTestServiceServicer):
    server: TestServer[Args, TestAttributes]

    def __init__(self, test: Test) -> None:
        super().__init__()
        self.server = TestServer[Args, TestAttributes](test)
        assert self.server.test.args.output_dir is None

    async def init(
        self, request: FeedFrameRequest, context: grpc.aio.ServicerContext
    ) -> FeedFrameResponse:
        assert self.server.init()
        return FeedFrameResponse(predict=request.region)

    async def track(
        self, request: FeedFrameRequest, context: grpc.aio.ServicerContext
    ) -> FeedFrameResponse:
        frame = pickle.loads(request.frame)
        gt = pickle.loads(request.region)
        self.server.next(frame, gt)
        assert self.server._attrs is not None
        return FeedFrameResponse(
            predict=pickle.dumps(
                self.server._attrs.historical.cur.result.unwrap().pred.get(),
                protocol=5,
            )
        )

    async def finish(
        self, request: FinishRequest, context: grpc.aio.ServicerContext
    ) -> FinishResponse:
        self.server.finish()
        return FinishResponse()


async def serve(test: Test, port: int):
    server = grpc.aio.server()
    servicedef_pb2_grpc.add_MSOTTestServiceServicer_to_server(
        TestServicer(test), server
    )
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    log.info("listening on %s", listen_addr)
    await server.start()
    await server.wait_for_termination()
