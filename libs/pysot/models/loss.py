# Copyright (c) SenseTime. All Rights Reserved.

import torch
import torch.nn.functional as F
from torch import Tensor


def get_cls_loss(pred: Tensor, label: Tensor, select: Tensor):
    if len(select.size()) == 0 or select.size() == torch.Size([0]):
        return torch.tensor(0.0).to(pred.device)

    pred = torch.index_select(pred, 0, select)
    # should be all 0 for loss_bg
    #        or all 1 for loss_fg
    label = torch.index_select(label, 0, select)

    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred: Tensor, label: Tensor):
    """
    calculate cross entropy loss for RPN output's cls
    pred:  `[B, A, 25, 25, 2]` (-inf, 0) - log softmax
    label: `[B, A, 25, 25]` (-1 | 0 | 1) score map
    """
    pred = pred.view(-1, 2)
    label = label.view(-1)

    # 1d tensor
    fg_idx = label.data.eq(1).nonzero().squeeze().to(pred.device)
    bg_idx = label.data.eq(0).nonzero().squeeze().to(pred.device)

    loss_fg = get_cls_loss(pred, label, fg_idx)
    loss_bg = get_cls_loss(pred, label, bg_idx)

    return loss_fg * 0.5 + loss_bg * 0.5


def weight_l1_loss(pred: Tensor, label: Tensor, loss_weight: Tensor):
    """
    calculate weighted l1 loss for RPN output's loc
    pred:        `[B, 4*A, 25, 25]`
    label:       `[B, 4, 5, 25, 25]`
    loss_weight: `[B, 4*A, 25, 25]`
    """
    B, _, H, W = pred.shape
    pred = pred.view(B, 4, -1, H, W)

    diff = (pred - label).abs()
    diff = diff.sum(dim=1).view(B, -1, H, W)

    loss = diff * loss_weight

    return loss.sum().div(B)
