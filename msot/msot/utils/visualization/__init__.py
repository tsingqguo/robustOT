import math

import cv2
import numpy as np
import numpy.typing as npt
import torch


def mat2tensor(mat: np.ndarray, no_batch: bool = False) -> torch.Tensor:
    img = torch.from_numpy(mat).float()
    if img.ndim == 3:
        if no_batch:
            img = img.permute(2, 0, 1)
            return img
        else:
            img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2)
    return img


def tensor2mat(
    t: torch.Tensor, allow_batch: bool = False, color_cvt: int | None = None
) -> npt.NDArray[np.uint8]:
    if t.ndim == 4:
        if allow_batch:
            ...
        elif t.shape[0] == 1:
            t = t[0]
        else:
            raise ValueError(
                f"tensor2mat: tensor has more than 1 batch (shape={t.shape})"
            )
    elif t.ndim == 3:
        return tensor2mat(
            t.unsqueeze(0),
            color_cvt=color_cvt,
        )
    elif t.ndim == 5 and t.shape[0] == 1:
        return tensor2mat(
            t[0],
            allow_batch=allow_batch,
            color_cvt=color_cvt,
        )
    else:
        raise ValueError(
            f"tensor2mat: tensor has invalid shape (shape={t.shape})"
        )

    t = t.clamp(0, 255)
    n: np.ndarray = t.cpu().detach().numpy()
    if n.ndim == 3:
        n = n.transpose(1, 2, 0)
        if color_cvt is not None:
            n = cv2.cvtColor(n, color_cvt)
    elif n.ndim == 4:
        _n = []
        for sub in n:
            sub = sub.transpose(1, 2, 0)
            if color_cvt is not None:
                sub = cv2.cvtColor(sub, color_cvt)
            _n.append(sub)
        n = np.stack(_n, axis=0)
    n = n.astype(np.uint8)
    return n


def merge_imgs(
    imgs: list[np.ndarray],
    ratio: float = 1,
    fixed_width: int | None = None,
) -> npt.NDArray[np.uint8]:
    fill_c = 255
    L = len(imgs)
    if L == 1:
        return imgs[0]
    else:
        w = max([im.shape[0] for im in imgs])
        h = max([im.shape[1] for im in imgs])
        c = imgs[0].shape[2]  # TODO:
        if fixed_width is not None:
            w_cnt = fixed_width
        else:
            w_cnt = round(math.sqrt(L * ratio))
        h_cnt = math.ceil(L / w_cnt)
        ln = []
        for h_i in range(h_cnt):
            ln_imgs = imgs[h_i * w_cnt : (h_i + 1) * w_cnt]
            for w_i, ln_im in enumerate(ln_imgs):
                w_local = ln_im.shape[0]
                h_local = ln_im.shape[1]
                if w_local != w:
                    w_l = round((w - w_local) / 2)
                    w_r = w - w_l - w_local
                    ln_im = np.pad(
                        ln_im,
                        ((w_l, w_r), (0, 0), (0, 0)),
                        mode="constant",
                        constant_values=fill_c,
                    )
                if h_local != h:
                    h_t = round((h - h_local) / 2)
                    h_b = h - h_t - h_local
                    ln_im = np.pad(
                        ln_im,
                        ((0, 0), (h_t, h_b), (0, 0)),
                        mode="constant",
                        constant_values=fill_c,
                    )
                ln_imgs[w_i] = ln_im
            while len(ln_imgs) < w_cnt:
                ln_imgs.append(fill_c * np.ones((w, h, c), dtype=np.uint8))
            ln.append(np.hstack(ln_imgs))
        img = np.vstack(ln)
        return img


def save_imgs(
    imgs: list[np.ndarray] | np.ndarray,
    path_str: str,
    ratio: float = 1,
    fixed_width: int | None = None,
):
    if isinstance(imgs, np.ndarray) and imgs.ndim == 4:
        imgs = [imgs[i] for i in range(imgs.shape[0])]
    if isinstance(imgs, np.ndarray) and imgs.ndim == 3:
        imgs = [imgs]
    elif isinstance(imgs, list):
        ...
    else:
        raise ValueError(f"invalid shape of imgs: {imgs.shape} (must be 4)")
    img = merge_imgs(imgs, ratio, fixed_width)
    cv2.imwrite(path_str, img)
