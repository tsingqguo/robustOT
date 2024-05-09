import clip
import torch
import torch.nn as nn
from clip.model import CLIP

from .models import ResampleCNN, ResampleMLP, STIR, make_model
from .utils import make_coord


def make_stir(model_spec: dict, args: dict | None = None):
    if model_spec["name"] != "stir":
        raise ValueError(
            f'Invalid model name: {model_spec["name"]} for method `make_stir`'
        )
    model = make_model(model_spec, args, load_sd=False)
    assert isinstance(model, STIR)
    return model


def make_rsn(model_spec: dict):
    rsn_type = model_spec.pop("rsn_type")
    if rsn_type == "cnn":
        model = ResampleCNN(**model_spec)
    elif rsn_type == "mlp":
        model = ResampleMLP(**model_spec)
    else:
        raise NotImplementedError
    return model


class LRR(nn.Module):
    stir: STIR
    rsn: ResampleCNN | ResampleMLP | None
    device: torch.device
    coord: torch.Tensor
    cell: torch.Tensor

    _clip: CLIP | None

    N: int
    """accept n length frames as input"""
    H: int
    W: int

    def __init__(
        self,
        device: torch.device,
        saved: dict,
        height: int = 255,
        width: int = 255,
    ):
        super().__init__()
        self.device = device
        self.H = height
        self.W = width

        stir_spec = saved["stir_spec"]

        self.stir = make_stir(
            stir_spec,
        ).to(self.device)
        self.N = stir_spec["args"]["n_length"]
        self.stir.N = self.N

        rsn_spec = saved["rsn_spec"]
        clip_model = rsn_spec.pop("clip_model", None)
        self.rsn = make_rsn(rsn_spec)

        self.load_state_dict(saved["sd"])

        if clip_model is not None:
            self._clip, _ = clip.load(clip_model)
        else:
            self._clip = None

        self.coord = make_coord((self.H, self.W)).to(self.device)

        self.cell = torch.ones_like(self.coord)
        self.cell[:, 0] *= 2 / self.H
        self.cell[:, 1] *= 2 / self.W

        if self.rsn is not None:
            self.rsn = self.rsn.to(self.device)

    @torch.no_grad()
    def predict(
        self,
        inp: torch.Tensor,
        coord: torch.Tensor,
        cell: torch.Tensor,
        text: torch.Tensor | None = None,
    ):
        """
        inp: [B, N*C, H, W]
        """
        self.stir.gen_feat(inp)
        if self.rsn is not None:
            coord = coord.clone()
            feat_unfold = self.stir.feat_unfolding(self.stir.feat)
            offsets = self.rsn.forward(feat_unfold, text=text)
            if coord.ndim == 3 and offsets.ndim == 4:
                offsets = offsets.squeeze(1)
            coord += offsets
        else:
            feat_unfold = None

        pred = self.stir.query_rgb_2t3(coord, cell, feat_unfolded=feat_unfold)

        return pred

    def forward(self, x: torch.Tensor, text: str | None = None):
        if self._clip is not None and text is not None:
            t = clip.tokenize(text).to(self.device)
            t = self._clip.encode_text(t)
        else:
            t = None

        results: list[torch.Tensor] = []
        for img in x:
            img = img.unsqueeze(0)
            if img.ndim == 5:
                _, fc, c, h, w = img.shape
                if fc != self.N:
                    if fc < self.N:
                        img = torch.cat(
                            [img[:, 0].repeat(1, self.N - fc, 1, 1, 1), img],
                            dim=1,
                        )
                    else:
                        raise ValueError(
                            f"input frames not match: fc({fc}) - N({self.N}"
                        )

                img = img.permute(0, 2, 1, 3, 4)
                img = img.reshape(-1, self.N * c, h, w)

            out = self.predict(
                ((img - 0.5) / 0.5),
                self.coord.unsqueeze(0),
                self.cell.unsqueeze(0),
                text=t,
            )[0]

            out = (
                (out * 0.5 + 0.5)
                .clamp(0, 1)
                .reshape(self.N, self.H, self.W, 3)
                .permute(0, 3, 1, 2)
            )
            results.append(out)

        gen = torch.stack(results)
        """[B, N, C, H, W]"""

        return gen
