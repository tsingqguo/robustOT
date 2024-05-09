import copy
from typing import Callable

import torch.nn as nn

ModelCtor = Callable[..., nn.Module]

_MODELS: dict[str, ModelCtor] = {}


def register(name: str):
    def decorator(ctor: ModelCtor) -> ModelCtor:
        _MODELS[name] = ctor
        return ctor

    return decorator


def make(model_spec: dict, args: dict | None = None, load_sd: bool = False):
    if args is not None:
        model_args = copy.deepcopy(model_spec["args"])
        model_args.update(args)
    else:
        model_args = model_spec["args"]
    name = model_spec["name"]
    if name not in _MODELS:
        raise KeyError(f"{name} not found in model register")
    model = _MODELS[name](**model_args)
    if load_sd:
        model.load_state_dict(model_spec["sd"])
    return model
