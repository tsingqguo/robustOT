import math
import numpy as np
import random
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from .datasets import register
from .h5db import PariedH5ImageDB
from LRR.utils import make_coord, to_pixel_samples
from torch.utils.data import Dataset


def get_shifts(size: int, max_abs: float = 0.1):
    shift = random.random() * max_abs
    step = 2 * shift / size
    shifts = torch.arange(-shift, shift + step, step)
    if random.random() < 0.5:
        shifts = shifts.flip(0)
    return shifts


def get_scale_factors(n: int, max_scale: float):
    scales = random.random() * max_scale
    scales = torch.linspace(1 - max_scale, 1 + max_scale, n)
    if random.random() < 0.5:
        scales = scales.flip(0)
    return scales


def get_rotations(n: int, max_rotation: float):
    rot_l = random.random() * max_rotation
    rot_r = random.random() * max_rotation
    rots = torch.linspace(-rot_l, rot_r, n)
    if random.random() < 0.5:
        rots = rots.flip(0)
    return rots


def rotate_coord(coord: torch.Tensor, theta: torch.Tensor):
    theta = torch.deg2rad(theta)
    rotation_matrix = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta), torch.cos(theta)],
        ]
    )
    rotated_coords = torch.matmul(coord, rotation_matrix)
    return rotated_coords


class FrameWiseAugmentation:
    _input_size: tuple[int, int]
    max_shift_x: float
    max_shift_y: float
    max_scale_x: float
    max_scale_y: float
    max_rotation: float
    """in deg"""

    base_coord: torch.Tensor
    """[H, W, 2]"""

    disabled: bool

    def __init__(
        self,
        input_size: tuple[int, int],
        max_shift_x: float = 0,
        max_shift_y: float = 0,
        max_scale_x: float = 0,
        max_scale_y: float = 0,
        max_rotation: float = 0,
        disable: bool = False,
    ):
        self._input_size = input_size
        self.max_shift_x = max_shift_x
        self.max_shift_y = max_shift_y
        self.max_scale_x = max_scale_x
        self.max_scale_y = max_scale_y
        self.max_rotation = max_rotation

        self.base_coord = make_coord(
            input_size,
            grid_indexing="xy",
            no_flatten=True,
        )
        self.disabled = disable

    def augment(self, *imgs: torch.Tensor):
        if self.disabled:
            return imgs
        n = imgs[0].shape[0]
        coord = self.base_coord.unsqueeze(0).repeat(n, 1, 1, 1).clone()
        coord = self.shift_(coord)
        coord = self.scale_(coord)
        coord = self.rotate_(coord)
        outputs = []
        for img in imgs:
            outputs.append(
                F.grid_sample(
                    img,
                    coord,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
            )
        return tuple(outputs)

    def shift_(self, coord: torch.Tensor):
        n = coord.shape[0]
        if self.max_shift_x > 0 or self.max_shift_y > 0:
            x_s = get_shifts(n, self.max_shift_x)
            y_s = get_shifts(n, self.max_shift_y)
            for i in range(n):
                coord[i, :, :, 0] += x_s[i]
                coord[i, :, :, 1] += y_s[i]
        return coord

    def scale_(self, coord: torch.Tensor):
        """coord must be 4D"""
        n = coord.shape[0]
        if self.max_scale_x > 0 or self.max_scale_y > 0:
            x_scale = get_scale_factors(n, self.max_scale_x)
            y_scale = get_scale_factors(n, self.max_scale_y)

            for i in range(n):
                coord[i, :, :, 0] *= x_scale[i]
                coord[i, :, :, 1] *= y_scale[i]
        return coord

    def rotate_(self, coord: torch.Tensor):
        """coord must be 4D"""
        n = coord.shape[0]
        if self.max_rotation > 0:
            rots = get_rotations(n, self.max_rotation)
            for i in range(n):
                coord[i, :, :, :] = rotate_coord(coord[i, :, :, :], rots[i])
        return coord


@register("sr-implicit-paired-adv")
class SRImplicitPairedAdv(Dataset):
    N: int
    """shift in ratio of the image size"""
    base_coord: torch.Tensor | None
    central_crop_region: int
    include_obj_cls: bool
    fake_obj_cls: str | None

    frame_wise_augmentation: FrameWiseAugmentation
    frame_wise_augmentation_only_on_non_N: bool

    def __init__(
        self,
        dataset: PariedH5ImageDB,
        inp_size: int,
        augment: bool = False,
        sample_q: int | None = None,
        n_length: int = 1,
        central_crop_region: int = -1,
        include_obj_cls: bool = False,
        fake_obj_cls: str | None = None,
        frame_wise_augmentation_args: dict | None = None,
        frame_wise_augmentation_only_on_non_N: bool = False,
    ):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q
        self.N = n_length
        self.central_crop_region = central_crop_region
        if (
            self.central_crop_region > 0
            and self.central_crop_region < self.inp_size
        ):
            raise ValueError(
                f"central_crop_region ({self.central_crop_region}) "
                f"should be large than inp_size ({self.inp_size})"
            )
        self.include_obj_cls = include_obj_cls
        self.fake_obj_cls = fake_obj_cls
        if frame_wise_augmentation_args is None:
            fa_args = {}
        else:
            fa_args = frame_wise_augmentation_args
        fa_args = {**fa_args, "input_size": (255, 255)}

        self.frame_wise_augmentation = FrameWiseAugmentation(**fa_args)
        self.frame_wise_augmentation_only_on_non_N = (
            frame_wise_augmentation_only_on_non_N
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr: torch.Tensor
        img_hr: torch.Tensor
        data = self.dataset[idx]
        if len(data) == 2:
            # legacy db
            img_lr, img_hr = data  # type: ignore
            obj_cls = None
        elif len(data) == 3:
            img_lr, img_hr, obj_cls = data
        else:
            raise ValueError(f"Invalid data length: {len(data)}")

        s = img_hr.shape[-2] // img_lr.shape[-2]  # assume int scale
        assert s == 1

        n = img_hr.shape[0]
        if n < self.N:
            d = self.N - n
            l = math.ceil(d / 2)
            r = d - l
            for _ in range(l):
                img_lr = torch.cat((img_lr, img_lr[0:1]), dim=0)
                img_hr = torch.cat((img_hr, img_hr[0:1]), dim=0)
            for _ in range(r):
                img_lr = torch.cat((img_lr[-1:], img_lr), dim=0)
                img_hr = torch.cat((img_hr[-1:], img_hr), dim=0)

        # frame-wise augmentation
        if self.frame_wise_augmentation_only_on_non_N:
            if n < self.N:
                img_lr, img_hr = self.frame_wise_augmentation.augment(
                    img_lr, img_hr
                )
        else:
            img_lr, img_hr = self.frame_wise_augmentation.augment(
                img_lr, img_hr
            )

        # central crop
        if self.central_crop_region > 0:
            img_lr = TF.center_crop(img_lr, self.central_crop_region)
            img_hr = TF.center_crop(img_hr, self.central_crop_region)

        w_lr = self.inp_size
        x0 = random.randint(0, img_lr.shape[-2] - w_lr)
        y0 = random.randint(0, img_lr.shape[-1] - w_lr)
        crop_lr = img_lr[..., x0 : x0 + w_lr, y0 : y0 + w_lr]
        w_hr = w_lr * s
        x1 = x0 * s
        y1 = y0 * s
        crop_hr = img_hr[..., x1 : x1 + w_hr, y1 : y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        if crop_hr.ndim == 4:
            crop_hr = crop_hr.permute(1, 0, 2, 3)
            crop_hr = crop_hr.reshape(-1, *crop_hr.shape[-2:])
        if crop_lr.ndim == 4:
            crop_lr = crop_lr.permute(1, 0, 2, 3)
            crop_lr = crop_lr.reshape(-1, *crop_lr.shape[-2:])

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False
            )
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        data = {
            "inp": crop_lr,
            "coord": hr_coord,
            "cell": cell,
            "gt": hr_rgb,
        }
        if self.include_obj_cls:
            if obj_cls is not None:
                data["obj_cls"] = obj_cls
            elif self.fake_obj_cls is not None:
                data["obj_cls"] = self.fake_obj_cls
        return data
