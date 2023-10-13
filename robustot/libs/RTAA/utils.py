import numpy as np
import torch
import torch.nn.functional as F


def rpn_smoothL1(
    input: torch.Tensor, target: np.ndarray, label: torch.Tensor
) -> torch.Tensor:
    input = torch.transpose(input, 0, 1)
    pos_index = np.where(label.cpu() == 1)  # changed

    t = torch.from_numpy(target).cuda().float()
    loss = F.smooth_l1_loss(input[pos_index], t[pos_index], reduction="sum")

    return loss


def where(cond: torch.Tensor, x, y) -> torch.Tensor:
    cond = cond.float()
    return (cond * x) + ((1 - cond) * y)
