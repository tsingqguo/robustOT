from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .edsr import EDSR
from .mlp import MLP
from .utils import make, register
from ..utils import make_coord


def make_encoder(encode_spec):
    model = make(encode_spec)
    assert isinstance(model, EDSR)
    return model


def make_imnet(imnet_spec, args):
    model = make(imnet_spec, args=args)
    assert isinstance(model, MLP)
    return model


class STIR(nn.Module):
    encoder: EDSR
    feat_unfold: bool
    multiplier: int | None
    """only not None when feat_unfold is True"""
    imnet: MLP | None
    feat: torch.Tensor
    N: int
    N_eps: float

    _encoder_compiled: bool = False
    _encoder_cb: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self,
        encoder_spec: dict,
        imnet_spec: dict | None = None,
        local_ensemble: bool = True,
        feat_unfold: bool = True,
        cell_decode: bool = True,
        feat_unfold_kernel: int | None = None,
        n_length: int = 1,
        n_eps: float = 1e-2,
    ):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = make_encoder(encoder_spec)

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            if feat_unfold_kernel is None:
                self.feat_unfold_kernel = 3
            else:
                self.feat_unfold_kernel = feat_unfold_kernel
            if self.feat_unfold:
                self.multiplier = self.feat_unfold_kernel * self.feat_unfold_kernel
                imnet_in_dim *= self.multiplier
            else:
                self.multiplier = None
            imnet_in_dim += 2
            if self.cell_decode:
                imnet_in_dim += 2
            self.imnet = make_imnet(imnet_spec, args={"in_dim": imnet_in_dim})
        else:
            self.imnet = None

        self.N = n_length
        self.N_eps = n_eps

        def encoder_cb(feat):
            return self.encoder(feat)

        self._encoder_cb = encoder_cb

    def gen_feat(self, inp: torch.Tensor) -> torch.Tensor:
        if not self._encoder_compiled:
            self._encoder_cb = torch.compile(self._encoder_cb)

        self.feat = self._encoder_cb(inp)
        return self.feat

    def feat_unfolding(self, feat: torch.Tensor):
        """
        unfold feature if feat_unfold is enabled
        [B, E.out_dim, H, W] -> [B, E.out_dim * multiplier, H, W]
        """
        if self.feat_unfold and self.multiplier is not None:
            if self.feat_unfold_kernel == 3:
                feat = F.unfold(feat, self.feat_unfold_kernel, padding=1).view(
                    feat.shape[0],
                    feat.shape[1] * self.multiplier,
                    feat.shape[2],
                    feat.shape[3],
                )
            elif self.feat_unfold_kernel == 5:
                feat = F.unfold(feat, self.feat_unfold_kernel, padding=2).view(
                    feat.shape[0],
                    feat.shape[1] * self.multiplier,
                    feat.shape[2],
                    feat.shape[3],
                )
            elif self.feat_unfold_kernel == 7:
                feat = F.unfold(feat, self.feat_unfold_kernel, padding=3).view(
                    feat.shape[0],
                    feat.shape[1] * self.multiplier,
                    feat.shape[2],
                    feat.shape[3],
                )
        return feat

    def gen_latent(self, inp):
        latent = self.encoder.forward_latent(inp)
        return latent

    def gen_feat_and_latent(self, inp):
        self.feat, latent = self.encoder.forward_feat_and_latent(inp)
        return self.feat, latent

    def query_rgb(self, coord, cell=None):
        feat = self.feat

        if self.imnet is None:
            ret = F.grid_sample(
                feat,
                coord.flip(-1).unsqueeze(1),
                mode="nearest",
                align_corners=False,
            )[:, :, 0, :].permute(0, 2, 1)
            return ret

        feat = self.feat_unfolding(feat)

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = (
            make_coord(feat.shape[-2:], no_flatten=True)
            .cuda()
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(feat.shape[0], 2, *feat.shape[-2:])
        )

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat,
                    coord_.flip(-1).unsqueeze(1),
                    mode="nearest",
                    align_corners=False,
                )[:, :, 0, :].permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord,
                    coord_.flip(-1).unsqueeze(1),
                    mode="nearest",
                    align_corners=False,
                )[:, :, 0, :].permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode and cell is not None:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]
            areas[0] = areas[3]
            areas[3] = t
            t = areas[1]
            areas[1] = areas[2]
            areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        if not isinstance(ret, torch.Tensor):
            raise ValueError("ret is not a tensor")
        return ret

    # @torch.compile
    def query_rgb_2t3(
        self,
        coord: torch.Tensor,
        cell: torch.Tensor | None = None,
        feat_unfolded: torch.Tensor | None = None,
    ):
        feat = self.feat
        B = feat.shape[0]

        if self.imnet is None:
            ret = F.grid_sample(
                feat,
                coord.flip(-1).unsqueeze(1),
                mode="nearest",
                align_corners=False,
            )[:, :, 0, :].permute(0, 2, 1)
            return ret

        if feat_unfolded is None:
            feat = self.feat_unfolding(feat)
        else:
            feat = feat_unfolded

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        rx: float = 2 / feat.shape[-2] / 2
        ry: float = 2 / feat.shape[-1] / 2

        feat_coord = (
            make_coord(feat.shape[-2:], no_flatten=True)
            .cuda()
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(feat.shape[0], 2, *feat.shape[-2:])
        )

        preds: list[torch.Tensor] = []
        areas: list[torch.Tensor] = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift

                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat,
                    coord_.flip(-1).unsqueeze(1),
                    mode="nearest",
                    align_corners=False,
                )[:, :, 0, :].permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord,
                    coord_.flip(-1).unsqueeze(1),
                    mode="nearest",
                    align_corners=False,
                )[:, :, 0, :].permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode and cell is not None:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        """
        0: 00, 1: 01
        2: 10, 3: 11
        """
        if self.local_ensemble:
            t = areas[0]
            areas[0] = areas[3]
            areas[3] = t
            t = areas[1]
            areas[1] = areas[2]
            areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        if not isinstance(ret, torch.Tensor):
            raise ValueError("ret is not a tensor")

        if self.N != 1 and isinstance(ret, torch.Tensor):
            _ret = []
            ret = ret.view(B, ret.shape[-2], -1, self.N)
            ret = ret.permute(0, 3, 1, 2)
            for i in range(self.N):
                w = torch.arange(self.N) - i + self.N_eps
                w = torch.abs(1 / w)
                w = w / w.sum()
                w = w.to(ret.device)
                w = w.view(1, self.N, 1, 1)
                g = ret * w
                g = g.sum(dim=1)
                _ret.append(g)
            ret = torch.stack(_ret, dim=1)
            ret = ret.view(B, -1, 3)

        return ret

    def forward(
        self,
        inp: torch.Tensor,
        coord: torch.Tensor,
        cell: torch.Tensor | None = None,
    ):
        self.gen_feat(inp)
        feat_unfold = self.feat_unfolding(self.feat)
        return self.query_rgb_2t3(coord, cell, feat_unfolded=feat_unfold)

    def forward_with_latent(
        self,
        inp: torch.Tensor,
        coord: torch.Tensor,
        cell: torch.Tensor | None = None,
    ):
        _, latent = self.gen_feat_and_latent(inp)
        return self.query_rgb(coord, cell), latent


@register("stir")
def make_stir(
    encoder_spec: dict,
    imnet_spec: dict | None = None,
    local_ensemble: bool = True,
    feat_unfold: bool = True,
    cell_decode: bool = True,
    feat_unfold_kernel: int | None = None,
    n_length: int = 1,
    n_eps: float = 1e-2,
):
    return STIR(
        encoder_spec,
        imnet_spec,
        local_ensemble,
        feat_unfold,
        cell_decode,
        feat_unfold_kernel,
        n_length,
        n_eps,
    )
