import torch
import torch.nn as nn

from .utils import register


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_list: list[int]):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


@register("mlp")
def make_mlp(in_dim: int, out_dim: int, hidden_list: list[int]):
    return MLP(in_dim, out_dim, hidden_list)
