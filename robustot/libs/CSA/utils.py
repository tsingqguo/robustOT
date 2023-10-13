import torch


def normalize(im_tensor: torch.Tensor) -> torch.Tensor:
    """(0,255) ---> (-1,1)"""
    im_tensor = im_tensor / 255.0
    im_tensor = im_tensor - 0.5
    im_tensor = im_tensor / 0.5
    return im_tensor


def adv_attack_template(img_tensor: torch.Tensor, GAN) -> torch.Tensor:
    """adversarial attack to template"""
    """input: pytorch tensor(0,255) ---> output: pytorch tensor(0,255)"""
    """step1: Normalization"""
    img_tensor = normalize(img_tensor)
    """step2: pass to G"""
    with torch.no_grad():
        GAN.template_clean1 = img_tensor
        GAN.forward()
    img_adv = GAN.template_adv255
    return img_adv


def adv_attack_template_S(
    img_tensor: torch.Tensor, GAN, target_sz: tuple[int, ...]
) -> torch.Tensor:
    """adversarial attack to template"""
    """input: pytorch tensor(0,255) ---> output: pytorch tensor(0,255)"""
    """step1: Normalization"""
    img_tensor = normalize(img_tensor)
    """step2: pass to G"""
    with torch.no_grad():
        img_adv = GAN.transform(img_tensor, target_sz)
        return img_adv


def adv_attack_search(
    img_tensor: torch.Tensor, GAN, search_sz: tuple[int, ...]
) -> torch.Tensor:
    """adversarial attack to search region"""
    """input: pytorch tensor(0,255) ---> output: pytorch tensor(0,255)"""
    """step1: Normalization"""
    img_tensor = normalize(img_tensor)
    """step2: pass to G"""
    with torch.no_grad():
        GAN.search_clean1 = img_tensor
        GAN.num_search = img_tensor.size(0)
        GAN.forward(search_sz)
    img_adv = GAN.search_adv255
    return img_adv
