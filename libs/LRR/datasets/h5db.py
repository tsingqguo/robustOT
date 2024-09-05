import cv2
import h5py
import json
import numpy as np
import numpy.typing as npt
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Optional, TypedDict

from .datasets import register

_GLOBAL_H5F: dict[str, h5py.File] = {}


class _DBInfo(TypedDict):
    h5f: h5py.File
    index: list[str]


@register("paired-h5-image-dbs")
class PariedH5ImageDB(Dataset):
    root: str
    db: dict[str, _DBInfo]
    ids: list[int]
    N: int
    N_static: int
    """length of per data in this h5 file"""
    repeat: int

    object_cls_dict: dict[str, str]
    """class annotations"""

    def __init__(
        self,
        root: str,
        h5fp: list[str | list[str | int]],
        first_k: Optional[int] = None,
        repeat: int = 1,
        shuffle: bool = False,
        cache_index_fp: Optional[list[str]] = None,
        n_length: int = 1,
        db_n_length: int = 1,
        object_cls_files: list[str] | None = None,
    ) -> None:
        self.root = os.path.expandvars(root)

        self.db = {}
        self.N = n_length
        self.N_static = db_n_length
        self.repeat = repeat
        self.object_cls_dict = {}

        if object_cls_files is not None:
            for fp in object_cls_files:
                fp = os.path.expandvars(fp)
                data = json.load(open(fp, "r"))
                self.object_cls_dict = {**self.object_cls_dict, **data}

        pre_index: Optional[dict[str, list[str]]] = None
        if cache_index_fp is not None:
            pre_index = {}
            for index in cache_index_fp:
                print(f"[INFO] reading index from {index}")
                pre_index = {
                    **pre_index,
                    **json.load(open(os.path.join(self.root, index), "r")),
                }

        for fp in h5fp:
            if isinstance(fp, list):
                if len(fp) == 2:
                    fp, size = fp
                    size_mod = "rd"
                elif len(fp) == 3:
                    fp, size, size_mod = fp
                else:
                    raise ValueError("list fp must be of len 2 or 3")
                if not isinstance(fp, str):
                    raise ValueError
                if not isinstance(size, int):
                    raise ValueError
                if size_mod not in ["rd", "index"]:
                    raise ValueError(f"size_mod {size_mod} not supported")
            else:
                size = None
                size_mod = "rd"

            fp = os.path.expandvars(fp)
            fn = os.path.basename(fp)

            if fp not in _GLOBAL_H5F:
                if not os.path.isabs(fp):
                    fp_abs = os.path.join(self.root, fp)
                else:
                    fp_abs = fp
                    # fp = os.path.basename(fp)
                _GLOBAL_H5F[fn] = h5py.File(fp_abs, "r")
                # _GLOBAL_H5F[fn].close() # TODO:
            h5f = _GLOBAL_H5F[fn]

            if pre_index is not None:
                self.db[fn] = {"h5f": h5f, "index": pre_index[fn]}
            else:
                sub_index = []
                for d in ["VID", "DET", "COCO", "YOUTUBEBB"]:
                    if d in h5f:
                        grp = h5f[d]
                        if isinstance(grp, h5py.Group):
                            sub_index += [
                                os.path.join(d, x) for x in grp.keys()
                            ]
                self.db[fn] = {"h5f": h5f, "index": sub_index}

            if size is not None:
                if size_mod == "rd":
                    mode_str = f" (random {size})"
                    if size > len(self.db[fn]["index"]):
                        raise ValueError(
                            f"size {size} is larger than available {len(self.db[fn]['index'])}"
                        )
                    print(len(self.db[fn]["index"]))
                    print("size:", size)
                    sample = np.random.choice(
                        len(self.db[fn]["index"]), size, replace=False
                    )
                    self.db[fn]["index"] = [
                        self.db[fn]["index"][i] for i in sorted(sample)
                    ]
                    exp_dir = os.environ.get("LRR_EXP_DIR", None)
                    if exp_dir is not None:
                        json.dump(
                            self.db[fn]["index"],
                            open(
                                os.path.join(exp_dir, f"{fn}.rd_index.json"),
                                "w",
                            ),
                        )
                else:
                    slice = size
                    mode_str = f" (slice {slice})"
                    if slice < 0:
                        self.db[fn]["index"] = self.db[fn]["index"][slice:]
                    else:
                        self.db[fn]["index"] = self.db[fn]["index"][:slice]
            else:
                mode_str = ""

            print(
                f'[INFO] loaded {len(self.db[fn]["index"]):>7} items from {fp}{mode_str}'
            )

        self.ids = list(
            range(sum(len(self.db[fn]["index"]) for fn in self.db))
        )

        print(f"tot available: {len(self.ids)} items")
        if first_k is not None:
            if first_k >= 0:
                self.ids = self.ids[:first_k]
            else:
                self.ids = self.ids[first_k:]
        print(f"   tot in-use: {len(self.ids)} items")
        if shuffle:
            np.random.shuffle(self.ids)

    def _get_from_db(self, idx: int):
        try:
            idx = self.ids[idx]
        except IndexError as e:
            print(
                f"[ERR.IndexError] idx: {idx}; len(self.ids): {len(self.ids)}"
            )
            raise e
        for i, db in enumerate(self.db.items()):
            fp, db_info = db
            if idx < len(db_info["index"]):
                return idx, fp, db_info["index"][idx]
            else:
                idx -= len(db_info["index"])
        raise KeyError(f"Index {idx} not found in database")

    @staticmethod
    def get_im_from_h5db(
        db: h5py.Dataset, pick: int = -1
    ) -> npt.NDArray[np.uint8]:
        if np.issubclass_(db.dtype.type, np.integer):
            if pick == -1:
                return db[()]
            else:
                # TODO:
                return db[:pick]
        elif np.issubclass_(db.dtype.type, np.bytes_):
            # assume single image
            im = np.frombuffer(db[0], dtype=np.uint8)
            im = cv2.imdecode(im, cv2.IMREAD_COLOR)
            return im[None, ...]
        else:
            raise ValueError(
                f"Invalid dtype {db.dtype} from PariedH5ImageDB._get_im_from_h5db"
            )

    def __len__(self):
        return len(self.ids) * self.repeat

    def __getitem__(self, idx: int):
        idx = idx % len(self.ids)
        _, fp, dp = self._get_from_db(idx)
        # dp in format of "/{DB}/{split_folder}.{im_id:0<12}.{track_id:0<2}.{frame:0<6}"

        grp = self.db[fp]["h5f"][dp]
        if not isinstance(grp, h5py.Group):
            raise ValueError(f"Invalid Grp {fp} {dp}")
        adv = grp["adv"]
        cln = grp["cln"]

        dp_without_frame = dp.rsplit(".", 1)[0]
        obj_cls = self.object_cls_dict.get(dp_without_frame, None)

        if isinstance(cln, h5py.Dataset) and isinstance(adv, h5py.Dataset):
            adv = self.get_im_from_h5db(adv, self.N)
            cln = self.get_im_from_h5db(cln, self.N)
            # if self.N != 1:
            if True:
                _adv = []
                _cln = []
                for cln_im, adv_im in zip(cln, adv):
                    _adv.append(
                        transforms.ToTensor()(
                            cv2.cvtColor(adv_im, cv2.COLOR_BGR2RGB)
                        )
                    )
                    _cln.append(
                        transforms.ToTensor()(
                            cv2.cvtColor(cln_im, cv2.COLOR_BGR2RGB)
                        )
                    )
                adv = torch.stack(_adv)
                cln = torch.stack(_cln)
            else:
                adv = transforms.ToTensor()(adv)
                cln = transforms.ToTensor()(cln)

            return adv, cln, obj_cls
        else:
            raise ValueError(f"Invalid Dataset {fp} {dp}.cln/adv")
