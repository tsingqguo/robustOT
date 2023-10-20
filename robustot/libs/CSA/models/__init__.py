from argparse import Namespace
from copy import deepcopy

from pix2pix.models.base_model import BaseModel

from robustot.utils import suppress_print


def get_test_model(options: Namespace, checkpoint_dir: str) -> BaseModel:
    opt = deepcopy(options)
    opt.checkpoints_dir = checkpoint_dir

    # import importlib
    # modellib = importlib.import_module(...)
    if opt.model == "G_search_L2_500_regress":
        from .G_search_L2_500_regress import CSAModel
    else:
        # TODO:
        raise RuntimeError(f"Unknown model: {opt.model}")

    with suppress_print(enabled=(hasattr(opt, "silent") and opt.silent)):
        model = CSAModel(opt)
        model.setup(opt)

    model.eval()
    return model
