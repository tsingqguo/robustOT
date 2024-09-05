import torch


def where(cond: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    cond = cond.float()
    return (cond * x) + ((1 - cond) * y)
