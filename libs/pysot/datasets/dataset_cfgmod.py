# Copyright (c) SenseTime. All Rights Reserved.

import cv2
import h5py
import json
import logging
import numpy as np
import numpy.typing as npt
import os
import sys
import torch
from pyotp.config.pysot import CFG, _Cfg_Dataset, _Cfg_DatasetCfg, _Cfg_Train
from pyotp.utils import pick_from_frame_id
from pysot.datasets.anchor_target_cfgmod import AnchorTarget
from pysot.datasets.augmentation import Augmentation
from pysot.utils.bbox import Corner, center2corner, Center
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import (
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

logger = logging.getLogger("global")

# setting opencv
if sys.version_info.major == 3:
    cv2.ocl.setUseOpenCL(False)

_FramePath = str  # video file path or h5py Dataset path
_MetaDataTrackAnno = List[int]  # length fixed to 4
_MetaDataTrack = Dict[str, _MetaDataTrackAnno]
_MetaDataVideo = Dict[Union[np.str_, str], _MetaDataTrack]
_MetaData = Dict[str, _MetaDataVideo]


class _ImagePath(NamedTuple):
    video: list[str]
    track: str
    frame_idx: str
    region: str
    """`x` or `z`"""


class H5AdvDBInfo(NamedTuple):
    root: str
    h5fp: list[str]
    index: str
    N: int


class H5AdvDB:
    _h5_sto: dict[str, h5py.File]
    _inner_map: dict[str, dict[str, dict[int, tuple[h5py.File, str, int]]]]
    _index: dict[str, list[str]]
    _d_name: str
    # {video: trackinfo}
    # _trackinfo: dict
    # # {track: frameinfo}
    # _frameinfo: dict
    # # {frame: (h5_sto.ptr, dp, data_slice_idx)}

    def __init__(self, info: H5AdvDBInfo, d_name: str):
        self.info = info
        self._h5_sto = {}
        self._inner_map = {}
        self._index = json.load(
            open(
                os.path.join(
                    os.path.expandvars(self.info.root), self.info.index
                ),
                "r",
            )
        )
        self._d_name = d_name

        for fp in self.info.h5fp:
            fp = os.path.join(
                os.path.expandvars(self.info.root),
                fp,
            )
            self._h5_sto[fp] = h5py.File(fp, "r")

    def init_inner_map(self, dataset_meta_data: dict):
        ignored_db = 0
        ignored_dp = 0
        for fi, (fp, h5f) in enumerate(self._h5_sto.items()):
            if os.path.basename(fp) not in self._index:
                logger.warning(
                    f"H5AdvDB.init_inner_map: ignore {fp} since not in index"
                )
                ignored_db += 1
                continue
            index = self._index[os.path.basename(fp)]

            grp = h5f[self._d_name]
            if not isinstance(grp, h5py.Group):
                raise ValueError(f"Dataset {self._d_name} not found in {fp}")
            dp_list = grp.keys()

            pbar = tqdm(
                index,
                desc=f"[{self._d_name}] init_inner_map ({fi+1}/{len(self._h5_sto)})",
            )
            for dp in pbar:
                path = TrkDataset.teardown_h5_dp(dp)
                # assume every key in index is exist in h5

                vp = os.path.join(*path.video)
                v_info = dataset_meta_data.get(vp, None)
                if v_info is None:
                    raise KeyError(f"video {vp} not found in anno")
                t_info = v_info.get(path.track, None)
                if t_info is None:
                    raise KeyError(f"track {path.track} not found in anno")
                frames: list[int] = t_info["frames"]
                if self.info.N != 1:
                    hits = pick_from_frame_id(
                        frames, self.info.N, int(path.frame_idx)
                    )
                    for slice_idx, frame in enumerate(hits):
                        v = self._inner_map.setdefault(vp, {})
                        t = v.setdefault(path.track, {})
                        t[frame] = (h5f, dp, slice_idx)
                else:
                    v = self._inner_map.setdefault(vp, {})
                    t = v.setdefault(path.track, {})
                    t[int(path.frame_idx)] = (h5f, dp, -1)

        logger.info(
            f"[{self._d_name}] init_inner_map done, {ignored_db} db ignored, {ignored_dp} dp ignored"
        )

    def filter_meta_data(self, meta_data: dict):
        pbar = tqdm(
            list(meta_data.keys()),
            desc=f"[{self._d_name}] filtering meta data with h5 adv",
        )
        rm_track = 0
        rm_video = 0
        for v_name in pbar:
            v_info = meta_data[v_name]
            for t_name in list(v_info.keys()):
                t_info = v_info[t_name]
                frames: list[int] = t_info["frames"]
                valid_frames: list[int] = []
                for frame in frames:
                    if self._get(v_name, t_name, frame) is not None:
                        valid_frames.append(frame)
                if len(valid_frames) == 0:
                    del meta_data[v_name][t_name]
                    rm_track += 1
                else:
                    meta_data[v_name][t_name]["frames"] = valid_frames
            if len(meta_data[v_name].keys()) == 0:
                del meta_data[v_name]
                rm_video += 1
            # else:
            #     logger.debug(f'{v_name}: {meta_data[v_name].keys()}')
        logger.info(
            f"[{self._d_name}] filtering meta data with h5 adv done, {rm_track} tracks & {rm_video} videos been removed"
        )

    def _get(self, v_name: str, track: str, frame: int):
        if v_name not in self._inner_map:
            return None
        v_info = self._inner_map[v_name]
        if track not in v_info:
            return None
        t_info = v_info[track]
        if frame not in t_info:
            return None
        return t_info[frame]

    def get_data(self, v_name: str, track: str, frame: str, clean: bool):
        if v_name not in self._inner_map:
            logger.fatal(
                f"[{self._d_name}.adv] video {v_name} not found in inner map"
            )
        v_info = self._inner_map[v_name]
        t_info = v_info[track]
        h5f, dp, slice_idx = t_info[int(frame)]
        dp = os.path.join("", self._d_name, dp)
        try:
            grp = h5f[dp]
        except KeyError:
            raise KeyError(f"Dataset {dp} not found in {h5f}")
        if not isinstance(grp, h5py.Group):
            raise KeyError(f"Dataset {dp} not found in {h5f}")
        if clean:
            sp = "cln"
        else:
            sp = "adv"
        db = grp[sp]
        if not isinstance(db, h5py.Dataset):
            raise KeyError(f"Dataset {sp} not found in {dp}")
        if self.info.N != 1:
            data = db[slice_idx]
        else:
            data = db[0]
        return data

    def get_data_by_fp(self, fp: str, clean: bool):
        fp_split = fp.split(os.path.sep)
        root = fp_split[: fp_split.index("crop511") + 1]
        path = TrkDataset.teardown_pysot_fp(fp, root_rm=os.path.sep.join(root))
        return self.get_data(
            os.path.join(*path.video),
            path.track,
            path.frame_idx,
            clean,
        )


class SubDataset:
    labels: _MetaData
    _videos: Dict[int, str]
    pick: List[int]
    num_use: int
    no_shuffle: bool
    """
    len of items if use repeat == -1 (`_Cfg_DatasetCfg.num_use`)
    else repeat times
    """
    h5_sto: dict[str, h5py.File]
    clone_from_h5dp_list: Optional[list[str]]
    h5_adv: Optional[H5AdvDBInfo]
    h5_adv_db: Optional[H5AdvDB]

    def __init__(
        self,
        name: str,
        root: str,
        anno: str,
        frame_range: int,
        num_use: int,
        start_idx: int,
        no_shuffle: bool = False,
        use_h5: bool = False,
        clone_from_h5dp_list: Optional[list[str]] = None,
        h5_adv: Optional[H5AdvDBInfo] = None,
    ):
        self.name = name
        self.root = os.path.expandvars(root)
        self.anno = os.path.expandvars(anno)
        self.frame_range = frame_range
        self.start_idx = start_idx
        self.use_h5 = use_h5
        self.clone_from_h5dp_list = clone_from_h5dp_list
        self.h5_adv = h5_adv
        logger.info("loading " + name)
        with open(self.anno, "r") as f:
            meta_data = json.load(f)
            meta_data = self._filter_zero(meta_data)

        if self.clone_from_h5dp_list is not None:
            if h5_adv is not None:
                raise NotImplementedError(
                    "h5 adv db is not allowed in clone mode"
                )

            fp_list: list[str] = []
            for dp in self.clone_from_h5dp_list:
                path = TrkDataset.teardown_h5_dp(dp)
                fp_list.append(os.path.join(*path.video))
            pbar = tqdm(
                list(meta_data.keys()),
                desc=f"[{name}] filtering clone targets",
            )
            for video in pbar:
                if video not in fp_list:
                    del meta_data[video]

        pbar = tqdm(
            list(meta_data.keys()), desc=f"[{name}] filtering empty targets"
        )
        for video in pbar:
            for track in meta_data[video]:
                frames = meta_data[video][track]
                frames = list(
                    map(int, filter(lambda x: x.isdigit(), frames.keys()))
                )
                frames.sort()
                meta_data[video][track]["frames"] = frames
                if len(frames) <= 0:
                    logger.warning("{}/{} has no frames".format(video, track))
                    del meta_data[video][track]

        pbar = tqdm(
            list(meta_data.keys()), desc=f"[{name}] filtering empty videos"
        )
        for video in pbar:
            if len(meta_data[video]) <= 0:
                logger.warning("{} has no tracks".format(video))
                del meta_data[video]

        if self.h5_adv is not None:
            if self.use_h5:
                raise NotImplementedError(
                    "normal h5 can not be used with h5 adv db"
                )

            self.h5_adv_db = H5AdvDB(self.h5_adv, self.name)
            self.h5_adv_db.init_inner_map(meta_data)
            self.h5_adv_db.filter_meta_data(meta_data)
            # exit(0)
        else:
            self.h5_adv_db = None

        self._videos = list(meta_data.keys())
        self.labels = meta_data
        if self.clone_from_h5dp_list is not None:
            self.num = len(self.clone_from_h5dp_list)
            self.num_use = self.num
        else:
            self.num = len(self.labels)
            self.num_use = self.num if num_use == -1 else num_use
        logger.info(
            f"[{self.name}] loaded with {self.num} items ({self.num_use} in use)"
        )
        self.path_format = "{}.{}.{}.jpg"
        self.no_shuffle = no_shuffle

        if self.num == 0:
            print("[WARN] NO DATA IN DATASET")
            return

        if self.no_shuffle:
            self.pick = list(range(self.start_idx, self.start_idx + self.num))
        else:
            if self.clone_from_h5dp_list is not None:
                raise NotImplementedError(
                    "shuffle is not supported in clone mode"
                )
            self.pick = self.shuffle()
        logger.debug(f"pick.len: {len(self.pick)}")
        if self.use_h5:
            self.h5_sto = {}
            h5s = [f for f in os.listdir(self.root) if f.endswith(".h5")]
            pbar = tqdm(h5s, desc=f"[{self.name}] loading h5 files")
            for f in pbar:
                if f.endswith(".h5"):
                    self.h5_sto[f[:-3]] = h5py.File(
                        os.path.join(self.root, f), "r"
                    )
            if len(self.h5_sto.keys()) == 0:
                raise ValueError(f"no h5 file found in {self.root}")

    def get_video(
        self, index: int
    ) -> tuple[str, Optional[str], Optional[str]]:
        """
        return (video_name, track_idx, start_frame_idx) if using clone mode
        """
        if self.clone_from_h5dp_list is not None:
            h5dp = self.clone_from_h5dp_list[index]
            path = TrkDataset.teardown_h5_dp(h5dp)
            return os.path.join(*path.video), path.track, path.frame_idx
        else:
            return self._videos[index], None, None

    def get_image_from_h5(self, path: str) -> npt.NDArray[np.uint8]:
        h5_split_fn = path.split("/")[0]
        db_path = os.path.relpath(path, h5_split_fn)
        if h5_split_fn not in self.h5_sto:
            raise ValueError(
                f"[{self.name}] rel_root {h5_split_fn} not in h5_sto; (path: {path}; use_h5: {self.use_h5})"
            )
        db = self.h5_sto[h5_split_fn][db_path]
        if not isinstance(db, h5py.Dataset):
            raise ValueError(f"{h5_split_fn}.{db_path} is not a dataset")
        image = np.frombuffer(db[0], dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image

    def _filter_zero(self, meta_data):
        meta_data_new = {}
        for video, tracks in meta_data.items():
            new_tracks = {}
            for trk, frames in tracks.items():
                new_frames = {}
                for frm, bbox in frames.items():
                    if not isinstance(bbox, dict):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w <= 0 or h <= 0:
                            continue
                    new_frames[frm] = bbox
                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:
                meta_data_new[video] = new_tracks
        return meta_data_new

    def log(self):
        logger.info(
            "{} start-index {} select [{}/{}] path_format {}".format(
                self.name,
                self.start_idx,
                self.num_use,
                self.num,
                self.path_format,
            )
        )

    def shuffle(self) -> List[int]:
        lists = list(range(self.start_idx, self.start_idx + self.num))
        pick = []
        while len(pick) < self.num_use:
            np.random.shuffle(lists)
            pick += lists
        return pick[: self.num_use]

    def get_image_anno(
        self,
        video: str,
        track: Union[np.str_, str],
        frame: Union[np.int64, int],
    ) -> Tuple[_FramePath, _MetaDataTrackAnno]:
        frame_str = f"{frame:06d}"
        if self.use_h5:
            image_path = os.path.join(video, track, frame_str, "x")
        else:
            image_path = os.path.join(
                self.root,
                video,
                self.path_format.format(frame_str, track, "x"),
            )
        try:
            image_anno = self.labels[video][track][frame_str]
        except KeyError as e:
            print("[ERR] KeyError: ")
            print(f"\tvideo: {video}")
            print(f"\ttrack: {track}")
            print(f"\tframe_str: {frame_str}")
            import json

            print(json.dumps(self.labels[video], indent=4))
            raise e
        return image_path, image_anno

    def get_positive_pair(
        self, index: int
    ) -> Tuple[
        Tuple[_FramePath, _MetaDataTrackAnno],
        Tuple[_FramePath, _MetaDataTrackAnno],
    ]:
        video_name = self._videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]

        frames = track_info["frames"]
        template_frame = np.random.randint(0, len(frames))
        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames) - 1) + 1
        search_range = frames[left:right]
        template_frame = frames[template_frame]
        search_frame = np.random.choice(search_range)
        return self.get_image_anno(
            video_name, track, template_frame
        ), self.get_image_anno(video_name, track, search_frame)

    def get_random_target(self, index: int = -1):
        if self.no_shuffle:
            raise ValueError(
                "`get_random_target` can not be used in no_shuffle mode"
            )
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self._videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info["frames"]
        frame = np.random.choice(frames)
        return self.get_image_anno(video_name, track, frame)

    def __len__(self) -> int:
        if self.clone_from_h5dp_list is not None:
            return len(self.clone_from_h5dp_list)
        else:
            return self.num


class TrkDatasetData(TypedDict):
    template: npt.NDArray[np.float32]
    search: npt.NDArray[np.float32]
    label_cls: npt.NDArray[np.int64]
    label_loc: npt.NDArray[np.float32]
    label_loc_weight: npt.NDArray[np.float32]
    bbox: np.ndarray  # TODO: List[float]->np.ndarray
    _template_fp: str
    _search_fp: str
    _dataset: str


class TrkDataLoaderData(TypedDict):
    template: torch.Tensor  # torch.float32 (N, 3, 127, 127)
    search: torch.Tensor  # torch.float32 (N, 3, 255, 255)
    label_cls: torch.Tensor  # torch.int64 (N, 5, 25, 25)
    label_loc: torch.Tensor  # torch.int64 (N, 4, 5, 25, 25)
    label_loc_weight: torch.Tensor  # torch.float32 (N, 5, 25, 25)
    bbox: torch.Tensor  # torch.float64 (N, 4)
    """
    bbox in corner format `(x1, y1, x2, y2)`
    """
    _template_fp: list[str]
    _search_fp: list[str]
    _dataset: list[str]


class TrkDataset(Dataset):
    dset_config: _Cfg_Dataset
    train_config: _Cfg_Train
    all_dataset: list[SubDataset]
    subsets: dict[str, SubDataset]

    use_h5: bool

    def __init__(
        self,
        dset_config: _Cfg_Dataset,
        train_config: _Cfg_Train,
        anchor_stride: int,
        no_shuffle: bool = False,
        use_h5: bool = False,
        clone_from_index: Optional[dict[str, list]] = None,
        h5_adv: Optional[dict[str, H5AdvDBInfo]] = None,
    ):
        """
        anchor_stride: int == 8
        clone_from_index: dict[DSET_NAME, H5DP]
        """
        super(TrkDataset, self).__init__()

        self.dset_config = dset_config
        self.train_config = train_config

        desired_size = (
            (self.train_config.search_size - self.train_config.exemplar_size)
            / anchor_stride
            + 1
            + self.train_config.base_size
        )
        print('desired_size', desired_size)
        # FIXME: CODE BELOW IS NOT WORKING ON ALEXNET
        # if desired_size != self.train_config.output_size:
        #     raise Exception("size not match!")

        # create anchor target
        self.anchor_target = AnchorTarget()

        # create sub dataset
        self.all_dataset = []
        self.subsets = {}
        start = 0
        self.num = 0
        for name in dset_config.names:
            sub_dataset_cfg: _Cfg_DatasetCfg = getattr(dset_config, name)
            if clone_from_index is not None:
                clone_list = clone_from_index.get(name, None)
                if clone_list is None:
                    print("[WARN] skip empty clone target dataset:", name)
                    continue
            else:
                clone_list = None
            sub_dataset = SubDataset(
                name,
                sub_dataset_cfg.root.unwrap(),
                sub_dataset_cfg.anno.unwrap(),
                sub_dataset_cfg.frame_range,
                sub_dataset_cfg.num_use,
                start,
                no_shuffle=no_shuffle,
                use_h5=sub_dataset_cfg.use_hdf5 or use_h5,
                clone_from_h5dp_list=clone_list,
                h5_adv=h5_adv.get(name, None) if h5_adv is not None else None,
            )
            start += sub_dataset.num
            self.num += sub_dataset.num_use

            sub_dataset.log()
            self.all_dataset.append(sub_dataset)
            self.subsets[name] = sub_dataset

        # data augmentation
        self.template_aug = Augmentation(
            self.dset_config.template.shift,
            self.dset_config.template.scale,
            self.dset_config.template.blur,
            self.dset_config.template.flip,
            self.dset_config.template.color,
            self.dset_config.template.gn_mean,
            self.dset_config.template.gn_var,
        )
        self.search_aug = Augmentation(
            self.dset_config.search.shift,
            self.dset_config.search.scale,
            self.dset_config.search.blur,
            self.dset_config.search.flip,
            self.dset_config.search.color,
            self.dset_config.search.gn_mean,
            self.dset_config.search.gn_var,
        )
        videos_per_epoch = self.dset_config.videos_per_epoch
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num
        self.num *= train_config.epoch
        if no_shuffle:
            self.pick = []
            for sub_dataset in self.all_dataset:
                self.pick += sub_dataset.pick
        else:
            self.pick = self.shuffle()

        self.use_h5 = use_h5

    def shuffle(self) -> List[int]:
        pick = []
        m = 0
        while m < self.num:
            p = []
            for sub_dataset in self.all_dataset:
                sub_p = sub_dataset.pick
                p += sub_p
            np.random.shuffle(p)
            pick += p
            m = len(pick)
        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))
        return pick[: self.num]

    def _find_dataset(self, index: int) -> Tuple[SubDataset, int]:
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:
                return dataset, index - dataset.start_idx
        raise KeyError(f"index {index} not found")

    def _get_bbox(self, image: cv2.Mat, shape: _MetaDataTrackAnno) -> Corner:
        anno = shape
        #
        imh, imw = image.shape[:2]
        if len(anno) == 4:
            w, h = anno[2] - anno[0], anno[3] - anno[1]
        else:
            w, h = anno
        context_amount = 0.5
        exemplar_size = self.train_config.exemplar_size
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w * scale_z
        h = h * scale_z
        cx, cy = imw // 2, imh // 2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    @staticmethod
    def teardown_h5_fp(fp: str) -> _ImagePath:
        sep = os.path.sep
        fp_arr = fp.split(sep)
        if len(fp_arr) < 4:
            raise ValueError(f"invalid fp {fp}")
        region = fp_arr[-1]
        frame_idx = fp_arr[-2]
        track = fp_arr[-3]
        video = fp_arr[:-3]
        return _ImagePath(
            video=video,
            track=track,
            frame_idx=frame_idx,
            region=region,
        )

    @staticmethod
    def teardown_h5_dp(dp: str, no_dataset_name: bool = False) -> _ImagePath:
        sep = "/"
        dp_arr = dp.split(sep)
        if dp_arr[0] == "":
            dp_arr = dp_arr[1:]
        if no_dataset_name:
            if len(dp_arr) != 1:
                raise ValueError(f"invalid h5dp {dp}")
            dp_arr = dp_arr[0].split(".")
        else:
            if len(dp_arr) != 2:
                raise ValueError(f"invalid h5dp {dp}")
            dp_arr = dp_arr[1].split(".")
        region = "!"  # TODO: potential issue
        frame_idx = dp_arr[-1]
        track = dp_arr[-2]
        video = dp_arr[:-2]
        return _ImagePath(
            video=video,
            track=track,
            frame_idx=frame_idx,
            region=region,
        )

    @staticmethod
    def teardown_pysot_fp(fp: str, root_rm: str) -> _ImagePath:
        fn_split = os.path.basename(fp).split(".")
        if len(fn_split) != 4:
            raise ValueError(f"invalid pysot fn {fp}")
        frame_idx = fn_split[0]
        track = fn_split[1]
        region = fn_split[2]
        video = os.path.dirname(fp)
        video = os.path.relpath(video, root_rm)
        return _ImagePath(
            video=video.split(os.path.sep),
            track=track,
            frame_idx=frame_idx,
            region=region,
        )

    def __len__(self) -> int:
        # return self.num # origin code
        return len(self.pick)

    def __getitem__(self, index: int) -> TrkDatasetData:
        index = self.pick[index]
        _idx_debug = index
        dataset, index = self._find_dataset(index)

        gray = (
            self.dset_config.gray
            and self.dset_config.gray > np.random.random()
        )
        neg = (
            self.dset_config.neg and self.dset_config.neg > np.random.random()
        )

        # get one dataset
        if neg:
            dset_t = dataset
            dset_s = np.random.choice(self.all_dataset)
            template = dset_t.get_random_target(index)
            search = dset_s.get_random_target()
        else:
            dset_t = dataset
            dset_s = dataset
            template, search = dataset.get_positive_pair(index)

        # logger.debug(
        #     f"picking {_idx_debug} -> dataset: {dataset.name}.{index}"
        # )

        # get image
        template_fp = template[0]
        search_fp = search[0]

        if dset_t.h5_adv_db is not None:
            template_image = dset_t.h5_adv_db.get_data_by_fp(
                template_fp,
                clean=True,
            )
        elif dset_t.use_h5:
            template_image = dset_t.get_image_from_h5(template_fp)
        else:
            template_image = cv2.imread(template_fp)

        if dset_s.h5_adv_db is not None:
            search_image = dset_s.h5_adv_db.get_data_by_fp(
                search_fp,
                clean=False,
            )
        elif dset_s.use_h5:
            search_image = dset_s.get_image_from_h5(search_fp)
        else:
            search_image = cv2.imread(search_fp)

        # get bounding box
        template_box = self._get_bbox(template_image, template[1])
        search_box = self._get_bbox(search_image, search[1])

        # augmentation
        template, _ = self.template_aug(
            template_image,
            template_box,
            self.train_config.exemplar_size,
            gray=gray,
        )

        search, bbox = self.search_aug(
            search_image, search_box, self.train_config.search_size, gray=gray
        )

        # print('[debug] search:', search.shape)

        # get labels
        cls, delta, delta_weight, overlap = self.anchor_target(
            bbox, self.train_config.output_size, neg
        )
        template = template.transpose((2, 0, 1)).astype(np.float32)
        search = search.transpose((2, 0, 1)).astype(np.float32)
        return {
            "template": template,
            "search": search,
            "label_cls": cls,
            "label_loc": delta,
            "label_loc_weight": delta_weight,
            "bbox": np.array(bbox),
            "_template_fp": template_fp,
            "_search_fp": search_fp,
            "_dataset": dataset.name,
        }


class TrkTrainDatasetDummy:
    def __init__(self) -> None:
        self.anchor_target = AnchorTarget()

        self.template_aug = Augmentation(
            CFG.dataset.template.shift,
            CFG.dataset.template.scale,
            CFG.dataset.template.blur,
            CFG.dataset.template.flip,
            CFG.dataset.template.color,
            CFG.dataset.template.gn_mean,
            CFG.dataset.template.gn_var,
        )
        self.search_aug = Augmentation(
            CFG.dataset.search.shift,
            CFG.dataset.search.scale,
            CFG.dataset.search.blur,
            CFG.dataset.search.flip,
            CFG.dataset.search.color,
            CFG.dataset.search.gn_mean,
            CFG.dataset.search.gn_var,
        )

    def process(
        self,
        template_im: cv2.Mat,
        search_im: cv2.Mat,
        bbox: Union[Corner, Center],
    ):
        if isinstance(bbox, Center):
            bbox = center2corner(bbox)

        template, _ = self.template_aug(
            template_im, bbox, CFG.train.exemplar_size, gray=False
        )

        search, bbox = self.search_aug(
            search_im, bbox, CFG.train.search_size, gray=False
        )

        cls, delta, delta_weight, overlap = self.anchor_target(
            bbox, CFG.train.output_size, False
        )

        template = template.transpose((2, 0, 1)).astype(np.float32)
        search = search.transpose((2, 0, 1)).astype(np.float32)

        return {
            "template": template,
            "search": search,
            "label_cls": cls,
            "label_loc": delta,
            "label_loc_weight": delta_weight,
            "bbox": np.array(bbox),
        }
