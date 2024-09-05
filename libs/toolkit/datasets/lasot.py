import json
import numpy as np
import numpy.typing as npt
import os
from .dataset import Dataset
from .video import Video
from tqdm import tqdm
from typing import Literal, Union


ValidDatasetNames = Literal["LaSOT"]


class LaSOTVideo(Video):
    absent: Union[npt.NDArray[np.int8], int]

    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
        attr: attribute of video
    """

    def __init__(
        self,
        name,
        root,
        video_dir,
        init_rect,
        img_names,
        gt_rect,
        attr,
        absent,
        load_img=False,
    ):
        super().__init__(
            name,
            root,
            video_dir,
            init_rect,
            img_names,
            gt_rect,
            attr,
            load_img,
        )
        self.absent = np.array(absent, np.int8)

    def load_tracker(
        self,
        path: str,
        tracker_names=None,
        variant=None,
        store: bool = True,
    ):
        tracker_names, variant = self._prepare_tracker_names(
            path,
            tracker_names,
            variant,
        )
        for name in tracker_names:
            traj_file = os.path.join(path, name, variant, self.name + ".txt")
            if os.path.exists(traj_file):
                with open(traj_file, "r") as f:
                    pred_traj = [
                        list(map(float, x.strip().split(",")))
                        for x in f.readlines()
                    ]
            else:
                print("File not exists: ", traj_file)
                continue
            if self.name == "monkey-17":
                pred_traj = pred_traj[: len(self.gt_traj)]
            if store:
                self.pred_trajs[name] = pred_traj
            else:
                return pred_traj
        self.tracker_names = list(self.pred_trajs.keys())


class LaSOTDataset(Dataset[LaSOTVideo]):
    def __init__(
        self,
        name: ValidDatasetNames,
        dataset_root: str,
        load_img: bool = False,
    ):
        super().__init__(name, dataset_root, LaSOTVideo)
        with open(os.path.join(dataset_root, name + ".json"), "r") as f:
            meta_data = json.load(f)

        # load videos
        pbar = tqdm(meta_data.keys(), desc="loading " + name)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            self.videos[video] = self._create_video(
                video,
                dataset_root,
                meta_data[video]["video_dir"],
                meta_data[video]["init_rect"],
                meta_data[video]["img_names"],
                meta_data[video]["gt_rect"],
                meta_data[video]["attr"],
                meta_data[video]["absent"],
            )

        # set attr
        attr = []
        for x in self.videos.values():
            attr += x.attr
        attr = set(attr)
        self.attr = {}
        self.attr["ALL"] = list(self.videos.keys())
        for x in attr:
            self.attr[x] = []
        for k, v in self.videos.items():
            for attr_ in v.attr:
                self.attr[attr_].append(k)
