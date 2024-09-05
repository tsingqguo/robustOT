import torch
from pix2pix.models.base_model_csa import BaseModel_CSA
from typing import Tuple


class Base_L2_500(BaseModel_CSA):
    """
    ONLY ONE IS ACTUALLY IN USE
    found in `GAN_utils_{search/template}_{co/cs}.py`
    """


class Base_L2_500_Search(Base_L2_500):
    search_clean1: torch.Tensor
    search_clean255: torch.Tensor
    search_adv1: torch.Tensor
    search_adv255: torch.Tensor
    num_search: int

    def transform(
        self,
        patch_clean1: torch.Tensor,
        target_sz: Tuple[int, int],
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, target_sz=(255, 255)) -> None:
        raise NotImplementedError


class Base_L2_500_Template(Base_L2_500):
    template_clean1: torch.Tensor
    template_clean255: torch.Tensor
    template_adv1: torch.Tensor
    template_adv255: torch.Tensor

    def forward(self) -> None:
        raise NotImplementedError
