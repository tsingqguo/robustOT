from typing import Literal

import torch
import torch.nn.functional as F


class InputPadder:
    """Pads images such that dimensions are divisible by 8"""

    def __init__(self, dims, mode="sintel"):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == "sintel":
            self._pad = [
                pad_wd // 2,
                pad_wd - pad_wd // 2,
                pad_ht // 2,
                pad_ht - pad_ht // 2,
            ]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs: torch.Tensor):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x: torch.Tensor):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


def make_coord(
    shape,
    ranges=None,
    grid_indexing: Literal["ij", "xy"] = "ij",
    no_flatten: bool = False,
) -> torch.Tensor:
    """
    Make coordinates at grid centers.
    ---
    Comment: can be replaced with `F.affine_grid()`
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(
        torch.meshgrid(*coord_seqs, indexing=grid_indexing),
        dim=-1,
    )
    if not no_flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret
