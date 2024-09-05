import torch
import torch.nn as nn
from pysot.models import MB_M_Input, MB_M_Output
from torchattacks.attack import Attack
from typing import Any, Callable, Optional


def _empty_loss_fn(t: MB_M_Input, model):
    raise NotImplementedError("loss function is not defined for FGSM")



class BIM(Attack):
    r"""
    BIM or iterative-FGSM in the paper 'Adversarial Examples in the Physical World'
    [https://arxiv.org/abs/1607.02533]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)

    .. note:: If steps set to 0, steps will be automatically decided following the paper.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.BIM(model, eps=8/255, alpha=2/255, steps=10)
        >>> adv_images = attack(images, labels)
    """
    loss_fn: Callable[[MB_M_Input, Any], torch.Tensor]

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10):
        super().__init__("BIM", model)
        self.eps = eps
        self.alpha = alpha
        if steps == 0:
            self.steps = int(min(eps * 255 + 4, 1.25 * eps * 255))
        else:
            self.steps = steps
        # self.supported_mode = ['default', 'targeted']
        self.supported_mode = ["default"]
        self.loss_fn = _empty_loss_fn

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

        ori_images = images.clone().detach()

        for _ in range(self.steps):
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

            adv_images = images + self.alpha * grad.sign()
            a = torch.clamp(ori_images - self.eps, min=0)
            b = (adv_images >= a).float() * adv_images + (
                adv_images < a
            ).float() * a
            c = (b > ori_images + self.eps).float() * (
                ori_images + self.eps
            ) + (b <= ori_images + self.eps).float() * b
            images = torch.clamp(c, max=1).detach()

        return images * 255
