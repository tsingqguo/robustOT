import torch
import torch.nn as nn
from LRR.models.mlp import MLP
from LRR.utils import InputPadder


class ResampleMLP(MLP):
    def __init__(
        self,
        feat_dim: int,
        n_length: int,
        hidden_list: list[int],
        text_dim: int = 512,
    ):
        self.feat_dim = feat_dim
        self.n_length = n_length
        self.layers = hidden_list
        self.text_dim = text_dim
        super().__init__(
            self.feat_dim + self.text_dim,
            2 * self.n_length,
            hidden_list,
        )

        self.tanh = nn.Tanh()

    def forward(self, feat: torch.Tensor, text: torch.Tensor | None = None):
        b = feat.shape[0]

        if self.text_dim != 0:
            if text is None:
                raise ValueError("text is None but text_dim != 0")

            text = (
                text.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(-1, -1, *feat.shape[-2:])
            )
            x = torch.cat([feat, text], dim=1)
        else:
            x = feat

        x = x.permute(0, 2, 3, 1)
        x = x.view(b, -1, x.shape[-1])
        x = x.reshape(-1, x.shape[-1])

        out = super().forward(x)
        out = self.tanh(out)
        out = out.view(b, -1, out.shape[-1])
        out = out.reshape(b, out.shape[1], 2, -1)
        out = out.permute(0, 3, 1, 2)
        return out


class ResampleCNN(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        n_length: int,
        layers: list[int] = [256],
        text_dim: int = 512,
        debug: bool = False,
    ):
        super().__init__()

        self._padder = None

        self.feat_dim = feat_dim
        self.n_length = n_length
        self.layers = layers
        self.text_dim = text_dim
        layers = [self.feat_dim + self.text_dim, *layers, 2 * self.n_length]

        self.cnn = nn.Sequential()
        for i in range(len(layers) - 1):
            self.cnn.add_module(
                f"conv{i}",
                nn.Conv2d(layers[i], layers[i + 1], kernel_size=3, padding=1),
            )
            if i != len(layers) - 2:
                self.cnn.add_module(f"relu{i}", nn.ReLU())
        self.cnn.add_module("tanh", nn.Tanh())

    def forward(self, feat: torch.Tensor, text: torch.Tensor | None = None):
        b = feat.shape[0]

        if self.text_dim != 0:
            if text is None:
                raise ValueError("text is None but text_dim != 0")

            text = (
                text.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(-1, -1, *feat.shape[-2:])
            )
            x = torch.cat([feat, text], dim=1)
        else:
            x = feat

        if self._padder is None:
            self._padder = InputPadder(x.shape[-2:])
        x = self._padder.pad(x)[0]
        x = self.cnn(x)
        x = self._padder.unpad(x)
        x = x.reshape(b, self.n_length * 2, -1)
        x = x.reshape(b, 2, self.n_length, -1)
        x = x.permute(0, 2, 3, 1)
        return x
