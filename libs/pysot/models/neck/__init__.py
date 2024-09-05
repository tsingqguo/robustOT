from pysot.models.neck.neck import AdjustLayer, AdjustAllLayer
from pysot.types import VALID_ADJUST  # TODO:


def get_neck(name: VALID_ADJUST, **kwargs):
    NECKS = {
        "AdjustLayer": AdjustLayer,
        "AdjustAllLayer": AdjustAllLayer,
    }
    return NECKS[name](**kwargs)
