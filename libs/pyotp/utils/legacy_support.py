import yaml
from pyotp.utils.option import Option, Some
from pyotp.utils.config import Config
from typing import Any, Dict, List
from yaml.loader import SafeLoader


def merge_from_file(cfg: Config, fp: str):
    fh = open(fp)
    lc = yaml.load(fh, SafeLoader)

    def assign_val(c: Config, k: str, v: Any):
        cur_v = c.__getattribute__(k)
        # exceptions
        if k == "anchor_num" and type(v) is int:
            return
        # Option::Some
        if isinstance(cur_v, Option):
            v = Some(v)
        # type checking
        if type(v) is not type(cur_v):
            raise TypeError(type(cur_v))

        c.__setattr__(k, v)

    def assign_cfg(c: Config, data: Dict[str, Any], prev_k: List[str]):
        for key, val in data.items():
            key = key.lower()
            if type(val) is dict and key != "kwargs":
                assign_cfg(c.__getattribute__(key), val, [*prev_k, key])
            else:
                try:
                    assign_val(c, key, val)
                except TypeError as err:
                    k = ".".join(["CFG", *prev_k, key])
                    t = err.args[0].__name__
                    print("[ERR] legacy_support.merge_from_file")
                    print(f"\ttrying to assign {k}({t}) with value `{val}`")

    print('[DEBUG] assign legacy config from file')
    print(lc)
    assign_cfg(cfg, lc, [])
