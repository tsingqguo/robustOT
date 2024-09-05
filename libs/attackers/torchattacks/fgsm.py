import torch
import torch.nn as nn
from pysot.models import MB_M_Input, MB_M_Output
from torchattacks.attack import Attack
from typing import Any, Callable, Optional


def _empty_loss_fn(t: MB_M_Input, model):
    raise NotImplementedError("loss function is not defined for FGSM")


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    """
    loss_fn: Callable[[MB_M_Input, Any], torch.Tensor]

    def __init__(self, model, eps=8 / 255):
        super().__init__("FGSM", model)
        self.eps = eps
        # self.supported_mode = ['default', 'targeted']
        self.supported_mode = ["default"]
        self.loss_fn = _empty_loss_fn

    # def forward(self, images, labels):
    def attack_search(self, data: MB_M_Input) -> torch.Tensor:
        r"""
        ATTACK SEARCH ONLY
        """
        images = data["search"] / 255
        images = images.clone().detach().to(self.device)
        # labels = labels.clone().detach().to(self.device)

        # if self.targeted:
        #     target_labels = self.get_target_label(images, labels)

        # loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        # outputs = self.get_logits(images)

        # # Calculate loss
        # if self.targeted:
        #     cost = -loss(outputs, target_labels)
        # else:
        #     cost = loss(outputs, labels)
        cost = self.loss_fn({**data, "search": images * 255}, self.model)

        # Update adversarial images
        grad = torch.autograd.grad(
            cost, images, retain_graph=False, create_graph=False
        )[0]

        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images * 255
