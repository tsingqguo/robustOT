import torch.nn.functional as F
from torch import Tensor


def xcorr_fast(x: Tensor, kernel: Tensor) -> Tensor:
    """group conv2d to calculate cross correlation, fast version"""
    batch: int = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    po = F.conv2d(px, pk, groups=batch)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    return po


def xcorr_depthwise(x: Tensor, kernel: Tensor) -> Tensor:
    """depthwise cross correlation"""
    batch: int = kernel.size(0)
    channel: int = kernel.size(1)
    x = x.contiguous()
    x = x.view(1, batch * channel, x.size(2), x.size(3))
    kernel = kernel.contiguous()
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch * channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out
