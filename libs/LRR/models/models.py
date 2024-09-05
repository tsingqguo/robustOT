import copy
import torch.nn as nn
from typing import Type, TypeVar

M = TypeVar("M", bound=nn.Module)

models: dict[str, Type[nn.Module]] = {}


def register(name: str):
    def decorator(cls: Type[M]) -> Type[M]:
        models[name] = cls
        return cls

    return decorator


def make(model_spec: dict, args: dict | None = None, load_sd: bool = False):
    if args is not None:
        model_args = copy.deepcopy(model_spec["args"])
        model_args.update(args)
    else:
        model_args = model_spec["args"]
    model = models[model_spec["name"]](**model_args)
    if load_sd:
        model.load_state_dict(model_spec["sd"])
    return model
