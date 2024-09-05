import cv2
import h5py
import json
import numpy as np
import numpy.typing as npt
import os
import torch
from pysot.datasets.dataset_cfgmod import (
    TrkDataset,
    TrkDatasetData,
    TrkDataLoaderData,
)
from pyotp import ENV
from pyotp.config.pysot import CFG
from pyotp.tools.train import (
    TrainingProcessor,
    CSATrainingProcessor,
    TATK_BIMTrainingProcessor,
    TATK_FGSMTrainingProcessor,
    TATK_PGDTrainingProcessor,
)
from pyotp.utils import (
    logging,
    read_config,
    pick_from_frame_id,
    random_pick,
    save_imgs,
    tensor2mat,
)
from pysot.models.model_builder_cfgmod import ModelBuilder
from pysot.utils.model_load import load_pretrain
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Iterable, Literal, Optional, Type, TypedDict, TypeVar
from typed_cap import Cap


_LOGGER_NAME = "unet.bkpn.train"
logging.init_logger("global", level=logging.LT.Debug, ignore_exist=True)
# logging.set_valid_rank("global", {0})
logging.init_logger(_LOGGER_NAME, level=logging.LT.Debug, ignore_exist=True)
# logging.set_valid_rank(_LOGGER_NAME, {0})
LOG = logging.get_logger(_LOGGER_NAME)


def initial_model(pretrained: str) -> ModelBuilder:
    model = ModelBuilder()
    model = load_pretrain(model, pretrained).cuda()
    return model


from pysot.datasets.dataset_cfgmod import (
    SubDataset,
    _Cfg_Dataset,
    _Cfg_Train,
    _MetaDataTrackAnno,
)


class TrkNLengthDatasetData(TypedDict):
    template: npt.NDArray[np.float32]
    search: npt.NDArray[np.float32]
    label_cls: npt.NDArray[np.int64]
    label_loc: npt.NDArray[np.float32]
    label_loc_weight: npt.NDArray[np.float32]
    bbox: np.ndarray  # TODO: List[float]->np.ndarray
    _template_fp: list[str]
    _search_fp: list[str]
    _dataset: str


class TrkNLengthDataset(TrkDataset):
    n_length: int
    tot_pair_get: int

    def __init__(
        self,
        dset_config: _Cfg_Dataset,
        train_config: _Cfg_Train,
        anchor_stride: int,
        no_shuffle: bool = False,
        use_h5: bool = False,
        clone_from_index: Optional[dict[str, list]] = None,
    ):
        super().__init__(
            dset_config,
            train_config,
            anchor_stride,
            no_shuffle,
            use_h5,
            clone_from_index,
        )
        self.tot_pair_get = 0

    def get_n_length_positive_pair(
        self, sub: SubDataset, index: int, n: int
    ) -> list[
        tuple[
            tuple[str, _MetaDataTrackAnno],
            tuple[str, _MetaDataTrackAnno],
        ]
    ]:
        # video_name = sub._videos[index] # FIXME:
        video_name, track, frame_start = sub.get_video(index)
        video = sub.labels[video_name]

        if track is None:
            track = np.random.choice(list(video.keys()))

        track_info = video[track]

        frames = track_info["frames"]
        template_frame_id = frames[np.random.randint(0, len(frames))]

        if frame_start is None:
            search_selects = random_pick(
                frames, pick_length=n, gap_tolerance=2
            )
        else:
            # non-random from clone record
            search_selects = pick_from_frame_id(
                frames, pick_size=n, start_id=int(frame_start)
            )

        if search_selects is None:
            # raise ValueError(f"failed in `get_n_length_positive_pair`")
            return self.get_n_length_positive_pair(
                sub, np.random.randint(0, len(self)), n
            )

        # if frame_start is not None:
        #     print('[debug] reconstruct clone record')
        #     print(' video:', video_name)
        #     print(' track:', track)
        #     print(' frame:', search_selects)

        self.tot_pair_get += 1
        # print(
        #     f'{self.tot_pair_get}; track_length: {len(track_info["frames"])}; name: {video_name}'
        # )
        paris = []
        # use same template for n search images
        fixed_template = sub.get_image_anno(
            video_name, track, template_frame_id
        )
        cache_search: dict = {}
        # print(f'{n} pick; selection: {search_selects}')
        for i in range(len(search_selects) - n, len(search_selects)):
            idx = max(0, i)
            _search = cache_search.get(idx, None)
            if _search is None:
                _search = sub.get_image_anno(
                    video_name, track, search_selects[idx]
                )
                cache_search[idx] = _search
            paris.append(
                (
                    fixed_template,
                    _search,
                )
            )
        return paris

    def __getitem__(self, index: int) -> TrkNLengthDatasetData:
        index = self.pick[index]
        dataset, index = self._find_dataset(index)

        gray = (
            self.dset_config.gray
            and self.dset_config.gray > np.random.random()
        )
        neg = (
            self.dset_config.neg and self.dset_config.neg > np.random.random()
        )

        # always positive
        paris = self.get_n_length_positive_pair(dataset, index, self.n_length)

        templates: list[np.ndarray] = []
        searches: list[np.ndarray] = []
        label_cls_list: list[np.ndarray] = []
        label_loc_list: list[np.ndarray] = []
        label_loc_weight_list: list[np.ndarray] = []
        bboxes: list = []
        template_fp_list: list[str] = []
        search_fp_list: list[str] = []

        for template, search in paris:
            # get image
            template_fp = template[0]
            search_fp = search[0]
            if dataset.use_h5:
                template_image = dataset.get_image_from_h5(template_fp)
                search_image = dataset.get_image_from_h5(search_fp)
            else:
                template_image = cv2.imread(template_fp)
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
                search_image,
                search_box,
                self.train_config.search_size,
                gray=gray,
            )

            # get labels
            cls, delta, delta_weight, overlap = self.anchor_target(
                bbox, self.train_config.output_size, bool(neg)
            )
            template = template.transpose((2, 0, 1)).astype(np.float32)
            search = search.transpose((2, 0, 1)).astype(np.float32)

            templates.append(template)
            searches.append(search)
            label_cls_list.append(cls)
            label_loc_list.append(delta)
            label_loc_weight_list.append(delta_weight)
            bboxes.append(bbox)
            template_fp_list.append(template_fp)
            search_fp_list.append(search_fp)

        return {
            "template": np.stack(templates),
            "search": np.stack(searches),
            "label_cls": np.stack(label_cls_list),
            "label_loc": np.stack(label_loc_list),
            "label_loc_weight": np.stack(label_loc_weight_list),
            "bbox": np.stack(bboxes),
            "_template_fp": template_fp_list,
            "_search_fp": search_fp_list,
            "_dataset": dataset.name,
        }


T = TypeVar("T", bound=TrkDataset)


def build_dataloader(
    dataset_cls: Type[T],
    n_length: int,
    start: int = 0,
    end: Optional[int] = None,
    use_h5: bool = False,
    clone_fp: Optional[str | dict] = None,
    clone_subset: Optional[list[str]] = None,
) -> DataLoader[T]:
    LOG.info("build train dataset ...")

    if clone_fp is not None:
        if isinstance(clone_fp, str):
            LOG.info(f"using clone index from {clone_fp}")
            with open(clone_fp, "r") as f:
                list_index = json.load(f)
        else:
            list_index = clone_fp
        if isinstance(list_index, dict):
            if clone_subset is not None:
                _list_index: list[str] = []
                for k in clone_subset:
                    try:
                        _list_index.extend(list_index[k])
                        LOG.info(
                            f"clone from subset {k} ({len(list_index[k])} items)"
                        )
                    except KeyError:
                        LOG.error(f"sub {k} not found in clone index")
                        exit(1)
                list_index = _list_index
            else:
                LOG.error("clone index is in dict, but not subset is provided")
                exit(1)
        clone_index = {}
        pbar = tqdm(list_index, desc="parse clone index")
        for item in pbar:
            # /D/video
            db = item.rsplit("/", 1)[0]
            if db.startswith("/"):
                db = db[1:]
            clone_index.setdefault(db, []).append(item)
    else:
        clone_index = None

    train_dataset = dataset_cls(
        CFG.dataset,
        CFG.train,
        anchor_stride=8,
        no_shuffle=True,
        use_h5=use_h5,
        clone_from_index=clone_index,
    )
    if isinstance(train_dataset, TrkNLengthDataset):
        train_dataset.n_length = n_length
    LOG.info("build dataset done")

    train_sampler = None

    LOG.info(f"total available data: {len(train_dataset.pick)}")

    train_dataset.pick = train_dataset.pick[start:end]

    return DataLoader(
        train_dataset,
        batch_size=CFG.train.batch_size,
        num_workers=CFG.train.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )


class Args:
    # @alias=c
    config: str

    # @alias=s
    pretrained_model: str
    """pre-trained tracker model"""

    # @alias=N
    n_length: int = 1

    # @alias=H
    use_h5: bool = False

    # @alias=r
    repeats: int = 1

    # @alias=O
    output: Optional[str]
    """output h5 filepath"""

    start: int = 0
    end: Optional[int] = None

    # @alias=A
    attacks: list[Literal["fgsm", "pgd", "bim", "csa"]]
    eps: float = 8 / 255

    report_freq: int = 10

    clone: Optional[str]
    """fp of clone index in json"""

    clone_subset: Optional[list[str]]


def generate(
    save_name: str,
    dummy: ModelBuilder,
    dataset_cls: Type[T],
    dataloader: DataLoader[T],
    load_dir_prefix: str,
    save_dir: str,
    training_processors: list[TrainingProcessor],
    idx_range: Optional[tuple[int, Optional[int]]] = None,
    repeat_idx: int = -1,
    output_h5: Optional[h5py.File] = None,
    report_freq: int = 10,
):
    def vis_search(search: torch.Tensor, name: str) -> None:
        save_imgs(
            tensor2mat(search, allow_batch=True),
            f"rm_{name}.png",
        )

    def save_adv_results(
        search_dataset_list: Iterable[str],
        search_fp_list: Iterable[Iterable[str]],
        search_cln: torch.Tensor,
        search_adv: torch.Tensor,
        silent: bool = True,
    ):
        for dset_name, fp_list, clean, attacked in zip(
            search_dataset_list, search_fp_list, search_cln, search_adv
        ):
            clean = tensor2mat(clean, allow_batch=True)
            attacked = tensor2mat(attacked, allow_batch=True)
            saved_fp: set[str] = set()
            if output_h5:
                if dataloader.dataset.use_h5:
                    p = dataset_cls.teardown_h5_fp(list(fp_list)[0])
                else:
                    p = dataset_cls.teardown_pysot_fp(
                        list(fp_list)[0],
                        # root_rm=os.path.expandvars("$TRAIN_DSET_PATH/coco/crop511"),  # FIXME:
                        root_rm=os.path.expandvars("$TRAIN_DSET_PATH/det/crop511"),  # FIXME:
                    )
                for t in ["cln", "adv"]:
                    h5dp = os.path.join(
                        dset_name,
                        ".".join([*p.video, p.track, p.frame_idx]),
                        t,
                    )
                    # print(p)
                    # print(h5dp)
                    if h5dp in output_h5:
                        continue
                    else:
                        output_h5.create_dataset(
                            h5dp,
                            data=clean if t == "cln" else attacked,
                            compression="gzip",
                        )
                continue

            # non-h5 only
            for ori_fp, cln, atk in zip(fp_list, clean, attacked):
                if not ori_fp.startswith(load_dir_prefix):
                    raise ValueError(f"invalid fp: {ori_fp}")
                r_ori_fp = ori_fp[len(load_dir_prefix) :]
                if r_ori_fp in saved_fp:
                    continue
                saved_fp.add(r_ori_fp)
                for t in ["cln", "adv"]:
                    fp = os.path.join(save_dir, save_name, t, r_ori_fp)
                    base = os.path.dirname(fp)
                    if not os.path.isdir(base):
                        os.makedirs(base)

                    if t == "adv":
                        cv2.imwrite(fp, atk)
                    else:
                        cv2.imwrite(fp, cln)

    for idx, data in enumerate(dataloader):
        B = data["template"].shape[0]

        data["template"] = data["template"].view(
            -1, *data["template"].shape[2:]
        )
        data["search"] = data["search"].view(-1, *data["search"].shape[2:])
        data["label_cls"] = data["label_cls"].view(
            -1, *data["label_cls"].shape[2:]
        )
        data["label_loc"] = data["label_loc"].view(
            -1, *data["label_loc"].shape[2:]
        )
        data["label_loc_weight"] = data["label_loc_weight"].view(
            -1, *data["label_loc_weight"].shape[2:]
        )
        data["bbox"] = data["bbox"].view(-1, *data["bbox"].shape[2:])

        # vis_search(data["search"], "search")
        ori_search = data["search"].clone()
        # print(data["search"].shape)
        for tp in training_processors:
            data = tp(data)
            # vis_search(data["search"], "search_adv")
            # print(data["search"].shape)
            # _ = input('...')

        save_adv_results(
            data["_dataset"],
            list(zip(*data["_search_fp"])),
            search_cln=ori_search.view(B, -1, *ori_search.shape[1:]),
            search_adv=data["search"].view(B, -1, *data["search"].shape[1:]),
            silent=(idx != 0),
        )

        if idx % report_freq == 0:
            if idx_range is None:
                idx_range_str = ""
            elif idx_range[0] == 0 and idx_range[1] is None:
                idx_range_str = ""
            else:
                idx_range_str = (
                    ""
                    if idx_range is None
                    else f"[{idx_range[0]}:{idx_range[1]}] "
                )
            if output_h5 is None:
                h5fp = ""
            else:
                h5fp = f"\n\t->{output_h5.filename} "
            LOG.info(
                f"[{repeat_idx}] {idx_range_str}{save_name} progress: {idx/len(dataloader)*100:.2f}% ({idx}/{len(dataloader)}); got: {dataloader.dataset.tot_pair_get}{h5fp}"
            )


if __name__ == "__main__":
    args = Cap(Args).parse().value

    try:
        read_config(args.config, CFG)
    except FileNotFoundError as e:
        raise e

    model = initial_model(args.pretrained_model)

    dataset_cls = TrkDataset
    dataset_cls = TrkNLengthDataset

    tp_name = []

    tp_list: list[TrainingProcessor] = []
    for adv in args.attacks:
        if adv == "fgsm":
            tp_name.append("fgsm")
            tp_list.append(TATK_FGSMTrainingProcessor(model, eps=args.eps))
        elif adv == "pgd":
            tp_name.append("pgd")
            tp_list.append(
                TATK_PGDTrainingProcessor(model, steps=100, eps=args.eps)
            )
        elif adv == "bim":
            tp_name.append("bim")
            tp_list.append(TATK_BIMTrainingProcessor(model, eps=args.eps))
        elif adv == "csa":
            tp_name.append("csa")
            tp_list.append(CSATrainingProcessor(model))
        else:
            raise ValueError(f"unknown attack: {adv}")
        LOG.warning(f"ADD ATTACK: {tp_name[-1]}")

    CLONE = args.clone

    if args.output is None:
        h5fh = None
    else:
        if os.path.exists(args.output):
            if CLONE is not None:
                if args.clone_subset is None:
                    LOG.fatal(
                        "clone_subset is missing, which is required in clone mode"
                    )
                    exit(1)
                LOG.warning(
                    f"{args.output} exists and you are in clone mode, do you want to resume?"
                )
                a = input("[y/N]")
                if a != "y":
                    LOG.fatal("pls manually remove the exist file and retry")
                    exit(1)
                h5fh = h5py.File(args.output, "r+")
                print(f"[DEBUG] loading {args.output}")
                CLONE = json.load(open(CLONE, "r"))
                LOG.debug("filtering clone index...")
                for sub in args.clone_subset:
                    keys_keep = []
                    pbar = tqdm(CLONE[sub], desc=f"filtering {sub}", ncols=60)
                    rm_cnt = 0
                    for check_k in pbar:
                        # if isinstance(h5fh[check_k], h5py.Group):
                        if check_k in h5fh:
                            rm_cnt += 1
                        else:
                            keys_keep.append(check_k)
                    LOG.warning(
                        f"removed {rm_cnt} items in index subset {sub}"
                    )
                    CLONE[sub] = keys_keep

                LOG.debug("filtering clone index done")
                h5fh.close()
                h5fh = h5py.File(args.output, "a")
            else:
                a = input(f"{args.output} exists, overwrite? [y/N]")
                if a != "y":
                    exit(1)
                h5fh = h5py.File(args.output, "w")

        else:
            h5fh = h5py.File(args.output, "a")

    dataloader = build_dataloader(
        dataset_cls,
        args.n_length,
        start=args.start,
        end=args.end,
        use_h5=args.use_h5,
        clone_fp=CLONE,
        clone_subset=args.clone_subset,
    )

    for repeat_idx in range(args.repeats):
        try:
            generate(
                ",".join(tp_name),
                model,
                dataset_cls,
                dataloader,
                ENV.dset_root_training,
                os.path.join(
                    ENV.dset_root_training, "..", f"train_mod_N{args.n_length}"
                ),
                tp_list,
                (args.start, args.end),
                repeat_idx=repeat_idx,
                output_h5=h5fh,
                report_freq=args.report_freq,
            )
        except KeyboardInterrupt:
            LOG.info("KeyboardInterrupt")
            break

    LOG.info("ALL DONE")
    if h5fh is not None:
        h5fh.close()
