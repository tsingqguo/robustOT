import torch
import torch.nn as nn
from pysot.models import MB_M_Input, MB_M_Output
from torchattacks.attack import Attack
from typing import Any, Callable, Optional


def _empty_loss_fn(t: MB_M_Input, model):
    raise NotImplementedError("loss function is not defined for FGSM")



class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    loss_fn: Callable[[MB_M_Input, Any], torch.Tensor]

    def __init__(
        self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True
    ):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        # self.supported_mode = ['default', 'targeted']
        self.supported_mode = ["default"]
        self.loss_fn = _empty_loss_fn

    # def forward(self, images, labels):
    def attack_search(self, data: MB_M_Input) -> torch.Tensor:
        r"""
        Overridden.
        """
        images = data["search"] / 255
        images = images.clone().detach().to(self.device)
        # labels = labels.clone().detach().to(self.device)

        # if self.targeted:
        #     target_labels = self.get_target_label(images, labels)

        # loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            # outputs = self.get_logits(adv_images)

            # # Calculate loss
            # if self.targeted:
            #     cost = -loss(outputs, target_labels)
            # else:
            #     cost = loss(outputs, labels)
            cost = self.loss_fn({**data, "search": adv_images * 255}, self.model)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(
                adv_images - images, min=-self.eps, max=self.eps
            )
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images * 255
