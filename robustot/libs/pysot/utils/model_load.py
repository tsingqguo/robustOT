import logging
from typing import TypeVar

import torch
import torch.nn as nn

LOG = logging.getLogger("global")


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # filter 'num_batches_tracked'
    missing_keys = [
        x for x in missing_keys if not x.endswith("num_batches_tracked")
    ]
    if len(missing_keys) > 0:
        LOG.info("[Warning] missing keys: {}".format(missing_keys))
        LOG.info("missing keys:{}".format(len(missing_keys)))
    if len(unused_pretrained_keys) > 0:
        LOG.info(
            "[Warning] unused_pretrained_keys: {}".format(
                unused_pretrained_keys
            )
        )
        LOG.info(
            "unused checkpoint keys:{}".format(len(unused_pretrained_keys))
        )
    LOG.info("used keys:{}".format(len(used_pretrained_keys)))
    assert (
        len(used_pretrained_keys) > 0
    ), "load NONE from pretrained checkpoint"
    return True


def remove_prefix(state_dict, prefix):
    """Old style model is stored with all names of parameters
    share common prefix 'module.'"""
    LOG.info("remove prefix '{}'".format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


M = TypeVar("M", bound=nn.Module)


def load_pretrain(model: M, pretrained_path: str) -> M:
    LOG.info("load pretrained model from {}".format(pretrained_path))
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(
        pretrained_path, map_location=lambda storage, loc: storage.cuda(device)
    )
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(
            pretrained_dict["state_dict"], "module."
        )
    else:
        pretrained_dict = remove_prefix(pretrained_dict, "module.")

    try:
        check_keys(model, pretrained_dict)
    except:
        LOG.info(
            '[Warning]: using pretrain as features.\
                Adding "features." as prefix'
        )
        new_dict = {}
        for k, v in pretrained_dict.items():
            k = "features." + k
            new_dict[k] = v
        pretrained_dict = new_dict
        check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def restore_from(model, optimizer, ckpt_path):
    device = torch.cuda.current_device()
    ckpt = torch.load(
        ckpt_path, map_location=lambda storage, loc: storage.cuda(device)
    )
    epoch = ckpt["epoch"]

    ckpt_model_dict = remove_prefix(ckpt["state_dict"], "module.")
    check_keys(model, ckpt_model_dict)
    model.load_state_dict(ckpt_model_dict, strict=False)

    check_keys(optimizer, ckpt["optimizer"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return model, optimizer, epoch
