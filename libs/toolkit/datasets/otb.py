import json
import os
from .dataset import Dataset
from .video import Video
from glob import glob
from tqdm import tqdm
from typing import Any, Literal

ValidDatasetNames = Literal["OTB100", "CVPR13", "OTB50"]


class OTBVideo(Video):
    def __init__(
        self,
        name,
        root,
        video_dir,
        init_rect,
        img_names,
        gt_rect,
        attr,
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

    def load_tracker(
        self,
        path,
        tracker_names=None,
        variant=None,
        store=True,
    ):
        def fallback_traj_file(path: str, t_name: str, variant: str) -> str:
            v_name = self.name.lower().replace("-", "_")
            return os.path.join(path, t_name, variant, f"{v_name}.txt")

        return super().load_tracker(
            path,
            tracker_names,
            variant,
            store,
            fallback_traj_file,
        )


class OTBDataset(Dataset[OTBVideo]):
    def __init__(
        self,
        name: ValidDatasetNames,
        dataset_root: str,
        load_img: bool = False,
    ):
        super().__init__(name, dataset_root, OTBVideo)
        with open(os.path.join(dataset_root, name + ".json"), "r") as f:
            meta_data: dict[str, Any] = json.load(f)

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
                load_img,
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
