# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Protocol


def xcorr_slow(x: Tensor, kernel: Tensor) -> Tensor:
    """for loop to calculate cross correlation, slow version"""
    batch: int = x.size()[0]
    out: List[Tensor] = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, -1, px.size()[1], px.size()[2])
        pk = pk.view(1, -1, pk.size()[1], pk.size()[2])
        po = F.conv2d(px, pk)
        out.append(po)
    return torch.cat(out, 0)


def xcorr_slow_alt(x: Tensor, kernel: Tensor) -> Tensor:
    # source: CSA.pysot, SPARK.pysot
    batch: int = x.size()[0]
    out: List[Tensor] = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, px.size()[0], px.size()[1], px.size()[2])
        pk = pk.view(-1, px.size()[1], pk.size()[1], pk.size()[2])
        po = F.conv2d(px, pk)
        out.append(po)
    return torch.cat(out, 0)


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
    # _compare_debug_only(x, kernel) # TODO: remove this
    batch: int = kernel.size(0)
    channel: int = kernel.size(1)
    x = x.contiguous()
    x = x.view(1, batch * channel, x.size(2), x.size(3))
    kernel = kernel.contiguous()
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch * channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


def xcorr_depthwise_alt(x: Tensor, kernel: Tensor) -> Tensor:
    # source: CSA.pysot
    batch: int = x.size(0)
    channel: int = x.size(1)
    # x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


G = 0
WRAPPING = False


def _verify(x: Tensor, kernel: Tensor) -> Tensor:
    time.sleep(0.314)
    return torch.randn([8, 16, 32])


class _Alt_xcorr(Protocol):
    def __call__(self, x: Tensor, kernel: Tensor) -> Tensor:
        raise NotImplementedError


def _compare_debug_only(x: Tensor, kernel: Tensor, max_G: int = 10):
    # change these
    # def_method: _Alt_xcorr = xcorr_slow
    # alt_method: _Alt_xcorr = xcorr_slow_alt
    def_method: _Alt_xcorr = xcorr_depthwise
    alt_method: _Alt_xcorr = xcorr_depthwise_alt
    # def_method: _Alt_xcorr = _verify
    # alt_method: _Alt_xcorr = _verify
    global G, WRAPPING
    if WRAPPING:
        return
    else:
        WRAPPING = True
    G += 1
    if G > max_G:
        print("halt")
        exit(0)
    print(f"\n[!] called: compare_debug ({G})")
    try:
        t0 = time.time()
        res_old = def_method(x, kernel)
        t1 = time.time() - t0
        res_old = torch.flatten(res_old)
        #
        t0 = time.time()
        res_new = alt_method(x, kernel)
        t2 = time.time() - t0
        res_new = torch.flatten(res_new)
        #
        print(f"shape: {res_old.shape}; {t1:.4f} <-> {t2:.4f}")
        diff = []
        for i in range(res_old.shape[0]):
            if res_old[i] != res_new[i]:
                diff.append(i)
        print(f"\thas {len(diff)} different(s)")
    except Exception as err:
        print(f"\t some error ocurred")
        pass
    WRAPPING = False
