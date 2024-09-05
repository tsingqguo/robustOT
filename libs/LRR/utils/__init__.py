import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
from torch.optim import SGD, Adam
from torch.optim.optimizer import Optimizer
from typing import Iterable, Generic, Literal, Type, TypeVar


class Averager:
    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


M = TypeVar("M", bound=nn.Module)


class DataParallel(nn.DataParallel, Generic[M]):
    module: M

    def __init__(
        self,
        module: M,
        device_ids=None,
        output_device=None,
        dim: int = 0,
    ) -> None:
        super().__init__(module, device_ids, output_device, dim)


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


def make_data_parallel(model: M) -> DataParallel[M]:
    m = nn.parallel.DataParallel(model)
    return m  # type: ignore


class Timer:
    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return "{:.1f}h".format(t / 3600)
    elif t >= 60:
        return "{:.1f}m".format(t / 60)
    else:
        return "{:.1f}s".format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename="log.txt"):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), "a") as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip("/"))
    if os.path.exists(path):
        if remove and (
            basename.startswith("_")
            or input("{} exists, remove? (y/[n]): ".format(path)) == "y"
        ):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, "tensorboard"))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return "{:.1f}M".format(tot / 1e6)
        else:
            return "{:.1f}K".format(tot / 1e3)
    else:
        return tot


def make_optimizer(
    param_list: Iterable,
    optimizer_spec,
    load_sd: bool = False,
):
    valid_cls: dict[str, Type[Optimizer]] = {"sgd": SGD, "adam": Adam}
    optimizer_cls = valid_cls[optimizer_spec["name"]]
    optimizer = optimizer_cls(param_list, **optimizer_spec["args"])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec["sd"])
    return optimizer


def make_coord(
    shape,
    ranges=None,
    grid_indexing: Literal["ij", "xy"] = "ij",
    no_flatten: bool = False,
) -> torch.Tensor:
    """Make coordinates at grid centers."""
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


def to_pixel_samples(img: torch.Tensor, c: int = 3):
    """Convert the image to coord-RGB pairs.
    img: Tensor, (c, H, W)
    """
    coord = make_coord(img.shape[-2:])
    rgb = img.reshape(c, -1).permute(1, 0)
    return coord, rgb


def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == "benchmark":
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == "div2k":
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)
