import torch
import numpy as np
import numpy.typing as npt


def overlap_ratio(
    rect1: npt.NDArray[np.float64], rect2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    """
    rect1 = np.transpose(rect1)

    if rect1.ndim == 1:
        rect1 = rect1[None, :]
    if rect2.ndim == 1:
        rect2 = rect2[None, :]

    left = np.maximum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect
    iou = np.clip(intersect / union, 0, 1)

    return iou


def orthogonal_perturbation(delta, prev_sample, target_sample):
    size = int(max(prev_sample.shape[0] / 4, prev_sample.shape[1] / 4, 224))
    prev_sample_temp = np.resize(prev_sample, (size, size, 3))
    target_sample_temp = np.resize(target_sample, (size, size, 3))
    # Generate perturbation
    perturb = np.random.randn(size, size, 3)
    perturb /= get_diff(perturb, np.zeros_like(perturb))
    perturb *= delta * np.mean(get_diff(target_sample_temp, prev_sample_temp))
    # Project perturbation onto sphere around target
    diff = (target_sample_temp - prev_sample_temp).astype(np.float32)
    diff /= get_diff(target_sample_temp, prev_sample_temp)
    diff = diff.reshape(3, size, size)
    perturb = perturb.reshape(3, size, size)
    for i, channel in enumerate(diff):
        perturb[i] -= np.dot(perturb[i], channel) * channel
    perturb = perturb.reshape(size, size, 3)
    perturb_temp = np.resize(
        perturb, (prev_sample.shape[0], prev_sample.shape[1], 3)
    )
    return perturb_temp


def orthogonal_perturbation_torch(
    delta: float, prev_sample: torch.Tensor, target_sample: torch.Tensor
):
    size = max(prev_sample.shape[1] // 4, prev_sample.shape[2] // 4, 224)
    prev_sample_temp = prev_sample.clone().resize_(3, size, size)
    target_sample_temp = target_sample.clone().resize_(3, size, size)
    # Generate perturbation
    perturb = torch.randn(3, size, size).to(prev_sample.device)
    perturb /= get_diff_torch(perturb, torch.zeros_like(perturb)).reshape(3, 1, 1)
    perturb *= delta * torch.mean(
        get_diff_torch(target_sample_temp, prev_sample_temp)
    )
    # Project perturbation onto sphere around target
    diff = target_sample_temp - prev_sample_temp
    diff /= get_diff_torch(target_sample_temp, prev_sample_temp).reshape(3, 1, 1)
    for i, channel in enumerate(diff):
        perturb[i] -= (perturb[i].view(-1) @ channel.view(-1)) * channel
    perturb_temp = perturb.resize_(3, *prev_sample.shape[1:])
    return perturb_temp


def forward_perturbation(epsilon, prev_sample, target_sample):
    perturb = (target_sample - prev_sample).astype(np.float32)
    perturb /= get_diff(target_sample, prev_sample)
    perturb *= epsilon
    return perturb


def forward_perturbation_torch(
    epsilon: torch.Tensor,
    prev_sample: torch.Tensor,
    target_sample: torch.Tensor,
):
    pert = target_sample - prev_sample
    diff = get_diff_torch(target_sample, prev_sample)
    if diff.size() != torch.Size([1]):
        diff = diff.reshape(3, 1, 1)
    pert /= diff
    if epsilon.size() != torch.Size([1]):
        epsilon = epsilon.reshape(3, 1, 1)
    pert *= epsilon
    return pert


def get_diff(sample_1, sample_2):
    sample_1 = sample_1.reshape(3, sample_1.shape[0], sample_1.shape[1])
    sample_2 = sample_2.reshape(3, sample_2.shape[0], sample_2.shape[1])
    sample_1 = np.resize(sample_1, (3, 271, 271))
    sample_2 = np.resize(sample_2, (3, 271, 271))

    diff = []
    for i, channel in enumerate(sample_1):
        diff.append(np.linalg.norm((channel - sample_2[i]).astype(np.float32)))
    return np.array(diff)


def get_diff_torch(lhs: torch.Tensor, rhs: torch.Tensor):
    lhs = lhs.clone().resize_(3, 271, 271)
    rhs = rhs.clone().resize_(3, 271, 271)
    diff = []
    for i, channel in enumerate(lhs):
        diff.append(torch.norm((channel - rhs[i])))
    return torch.tensor(diff).to(lhs.device)
