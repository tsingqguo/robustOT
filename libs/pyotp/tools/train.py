import math
import numpy as np
import os
import random
import time
import torch
import torch.backends.cudnn
import torch.nn as nn
from pyotp.config.pysot import CFG
from pyotp.env import ENV
from pyotp.utils import read_config
from pyotp.utils import logging
from pysot.datasets.dataset_cfgmod import (
    H5AdvDBInfo,
    TrkDataset,
    TrkDatasetData,
    TrkDataLoaderData,
)
from pysot.models import MB_M_Input, MB_M_Output
from pysot.models.model_builder_cfgmod import ModelBuilder
from pysot.utils.average_meter import AverageMeter
from pysot.utils.distributed import (
    DistModule,
    average_reduce,
    dist_init,
    get_rank,
    get_world_size,
    reduce_gradients,
)
from pysot.utils.log_helper import print_speed  # FIXME: remove this
from pysot.utils.misc import describe
from pysot.utils.model_load import load_pretrain
from pysot.utils.lr_scheduler_cfgmod import build_lr_scheduler, LRScheduler
from tensorboardX import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typed_cap import Cap
from typing import (
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
)


_LOGGER_NAME = "pyotp.train"
logging.init_logger("global", level=logging.LT.Debug, ignore_exist=True)
logging.set_valid_rank("global", {0})
logging.init_logger(_LOGGER_NAME, level=logging.LT.Debug, ignore_exist=True)
logging.set_valid_rank(_LOGGER_NAME, {0})
LOG = logging.get_logger(_LOGGER_NAME)


TD = TypeVar("TD", bound=TrkDataset)


class TrainCliArgs:
    # @alias=c
    config: str = "config.yaml"
    """path to config file"""

    # @alias=b
    batch_size: int = 64
    """batch size"""

    seed: int = 123456
    """random seed"""

    local_rank: int = 0
    """compulsory for pytorch launcher"""

    # @alias=g
    gpu: int = 0
    """gpu id"""

    test: bool = False
    """test mode"""


def seed_torch(seed: int = 0) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_dataloader(dataset_cls: Type[TD]) -> DataLoader[TD]:
    LOG.info("build train dataset ...")
    # ATK = "fgsm"
    ATK = "pgd"
    # ATK = "csa"
    if ATK not in CFG.meta_arc.lower():
        LOG.fatal('potential DB, aborting')
        exit(1)
    H5_ADV = None
    H5_ADV = {
        "COCO": H5AdvDBInfo(
            root=f"$TRAIN_DSET_PATH/../train_mod/{ATK}",
            h5fp=[
                "coco.h5",
            ],
            index="index_coco.json",
            N=1,
        ),
        "DET": H5AdvDBInfo(
            root=f"$TRAIN_DSET_PATH/../train_mod/{ATK}",
            h5fp=[
                "det.h5",
            ],
            index="index_det.json",
            N=1,
        ),
        "VID": H5AdvDBInfo(
            root=f"$TRAIN_DSET_PATH/../train_mod/{ATK}",
            h5fp=[
                "N5/vid00.h5",
                "N5/vid01.h5",
                "N5/vid02.h5",
                "N5/vid03.h5",
                "N5/vid04.h5",
                "N5/vid05.h5",
                "N5/vid06.h5",
            ],
            index="index_vid_ytbb_unique.json",
            N=5,
        ),
        "YOUTUBEBB": H5AdvDBInfo(
            root=f"$TRAIN_DSET_PATH/../train_mod/{ATK}",
            h5fp=[
                "N5/ytbb0_50k_0.h5",
                "N5/ytbb0_50k_1.h5",
                "N5/ytbb50k_100k_0.h5",
                "N5/ytbb50k_100k_1.h5",
            ],
            index="index_vid_ytbb_unique.json",
            N=5,
        ),
    }
    # FIXME:
    train_dataset = dataset_cls(
        CFG.dataset, CFG.train, anchor_stride=CFG.anchor.stride, h5_adv=H5_ADV
    )
    LOG.info("build dataset done")

    train_sampler = None

    if get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset)

    return DataLoader(
        train_dataset,
        batch_size=CFG.train.batch_size,
        num_workers=CFG.train.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )


class _TrainParams(TypedDict):
    params: Iterable[torch.nn.parameter.Parameter]
    lr: float


def build_opt_lr(
    model: ModelBuilder,
    current_epoch: int = 0,
) -> Tuple[Optimizer, LRScheduler]:
    for param in model.backbone.parameters():
        param.requires_grad = False

    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    if current_epoch >= CFG.backbone.train_epoch:
        for layer in CFG.backbone.train_layers:
            for param in getattr(model.backbone, layer).parameters():
                param.requires_grad = True
            for m in getattr(model.backbone, layer).modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()

    trainable_params: List[_TrainParams] = []

    if CFG.train.partial.backbone:
        params: _TrainParams = {
            "params": filter(
                lambda x: x.requires_grad, model.backbone.parameters()
            ),
            "lr": CFG.backbone.layers_lr * CFG.train.base_lr,
        }
        trainable_params.append(params)

    if CFG.train.partial.neck and CFG.adjust.adjust:
        params: _TrainParams = {
            "params": model.neck.parameters(),
            "lr": CFG.train.base_lr,
        }
        trainable_params.append(params)

    if CFG.train.partial.rpn_head:
        params: _TrainParams = {
            "params": model.rpn_head.parameters(),
            "lr": CFG.train.base_lr,
        }
        trainable_params.append(params)

    if CFG.train.partial.mask_head and CFG.mask.mask:
        params: _TrainParams = {
            "params": model.mask_head.parameters(),
            "lr": CFG.train.base_lr,
        }
        trainable_params.append(params)

    if CFG.train.partial.refine_head and CFG.refine.refine:
        params: _TrainParams = {
            "params": model.refine_head.parameters(),
            "lr": CFG.train.base_lr,
        }
        trainable_params.append(params)

    optimizer = torch.optim.SGD(
        trainable_params,
        momentum=CFG.train.momentum,
        weight_decay=CFG.train.weight_decay,
    )  # type: ignore

    lr_scheduler = build_lr_scheduler(
        optimizer,
        epochs=CFG.train.epoch,
    )
    start_epoch = CFG.train.start_epoch
    if start_epoch.is_some():
        lr_scheduler.step(start_epoch.unwrap())
    else:
        lr_scheduler.step(None)

    return optimizer, lr_scheduler


def log_grads(model: ModelBuilder, tb_writer: SummaryWriter, tb_index: int):
    def weights_grads(model: ModelBuilder):
        grad = {}
        weights = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad[name] = param.grad
                weights[name] = param.data
        return grad, weights

    grad, weights = weights_grads(model)
    feature_norm, rpn_norm = 0, 0
    for k, g in grad.items():
        _norm = g.data.norm(2)
        weight = weights[k]
        w_norm = weight.norm(2)
        if "feature" in k:
            feature_norm += _norm**2
        else:
            rpn_norm += _norm**2

        tb_writer.add_scalar(
            "grad_all/" + k.replace(".", "/"), _norm, tb_index
        )
        tb_writer.add_scalar(
            "weight_all/" + k.replace(".", "/"), w_norm, tb_index
        )
        tb_writer.add_scalar(
            "w-g/" + k.replace(".", "/"), w_norm / (1e-20 + _norm), tb_index
        )
    tot_norm = feature_norm + rpn_norm
    tot_norm = tot_norm**0.5
    feature_norm = feature_norm**0.5
    rpn_norm = rpn_norm**0.5

    tb_writer.add_scalar("grad/tot", tot_norm, tb_index)
    tb_writer.add_scalar("grad/feature", feature_norm, tb_index)
    tb_writer.add_scalar("grad/rpn", rpn_norm, tb_index)


_TrkModel = Union[DistModule[ModelBuilder], ModelBuilder]
TM = TypeVar("TM", bound=_TrkModel)


class TrainingProcessor(Generic[TM]):
    name: str = "Untitled Training Processor"
    model: TM

    def __init__(self, name: str, model: TM) -> None:
        self.name = name
        self.model = model

    def search_process(self, search: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, data: TrkDataLoaderData) -> TrkDataLoaderData:
        ...


from attackers.csa.CSA.attack_utils import adv_attack_search
from attackers.csa.CSA.base_model import Base_L2_500
from attackers.csa.utils import AtkOnEnum, AtkType, get_CSA_GAN


class CSATrainingProcessor(TrainingProcessor[TM]):
    name: str
    attacker: Base_L2_500

    def __init__(self, model: TM) -> None:
        super().__init__("", model)
        gan, opt, suffix = get_CSA_GAN(AtkOnEnum.S, "cs")
        self.attacker = gan
        self.name = "CSA_tp_" + opt.model + suffix

    def search_process(self, search: torch.Tensor) -> torch.Tensor:
        org_device = search.device
        search = search.to(self.attacker.device)
        search_adv = adv_attack_search(search, self.attacker)
        return search_adv.to(org_device)

    def __call__(self, data: TrkDataLoaderData) -> TrkDataLoaderData:
        data["search"] = self.search_process(data["search"])
        return data


from attackers.torchattacks import (
    PGD as TATK_PGD,
    FGSM as TATK_FGSM,
    BIM as TATK_BIM,
)


def _torch_attack_loss_fn(
    data: MB_M_Input, model: ModelBuilder
) -> torch.Tensor:
    output = model.forward(data)
    return output["total_loss"]


class TATK_FGSMTrainingProcessor(TrainingProcessor[TM]):
    name: str

    attacker: TATK_FGSM
    _eps: float

    def __init__(
        self,
        model: TM,
        eps: float = 8 / 255,
    ) -> None:
        super().__init__("", model)
        self.name = "TATK_FGSM_tp"
        self._eps = eps
        self.attacker = TATK_FGSM(model, eps=self._eps)
        self.attacker.loss_fn = _torch_attack_loss_fn

    def __call__(self, data: TrkDataLoaderData) -> TrkDataLoaderData:
        search = self.attacker.attack_search(data)
        data["search"] = search
        return data


class TATK_PGDTrainingProcessor(TrainingProcessor[TM]):
    name: str

    attacker: TATK_PGD
    _eps: float
    _alpha: float
    _steps: int
    _random_start: bool

    def __init__(
        self,
        model: TM,
        eps: float = 8 / 255,
        alpha: float = 2 / 255,
        steps: int = 100,
        random_start: bool = True,
    ) -> None:
        super().__init__("", model)
        self.name = "TATK_PGD_tp"
        self._eps = eps
        self._alpha = alpha
        self._steps = steps
        self._random_start = random_start
        self.attacker = TATK_PGD(
            model,
            eps=self._eps,
            alpha=self._alpha,
            steps=self._steps,
            random_start=self._random_start,
        )
        self.attacker.loss_fn = _torch_attack_loss_fn

    def __call__(self, data: TrkDataLoaderData) -> TrkDataLoaderData:
        search = self.attacker.attack_search(data)
        data["search"] = search
        return data


class TATK_BIMTrainingProcessor(TrainingProcessor[TM]):
    name: str

    attacker: TATK_BIM
    _eps: float
    _alpha: float
    _steps: int

    def __init__(
        self,
        model: TM,
        eps: float = 8 / 255,
        alpha: float = 2 / 255,
        steps: int = 100,
    ) -> None:
        super().__init__("", model)
        self.name = "TATK_BIM_tp"
        self._eps = eps
        self._alpha = alpha
        self._steps = steps
        self.attacker = TATK_BIM(
            model,
            eps=self._eps,
            alpha=self._alpha,
            steps=self._steps,
        )
        self.attacker.loss_fn = _torch_attack_loss_fn

    def __call__(self, data: TrkDataLoaderData) -> TrkDataLoaderData:
        search = self.attacker.attack_search(data)
        data["search"] = search
        return data


def train(
    train_loader: DataLoader[TrkDataset],
    model: _TrkModel,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    tb_writer: Optional[SummaryWriter] = None,
    train_processors: Optional[List[TrainingProcessor]] = [],
):
    cur_lr = lr_scheduler.get_lr()[0]
    average_meter = AverageMeter()

    def is_valid_number(x):
        return not (math.isnan(x) or math.isinf(x) or x > 1e4)

    # TrkDataset.__len__ == TrkDataset.num
    dset_len = len(train_loader.dataset)  # type: ignore
    num_per_epoch = (
        dset_len
        // CFG.train.epoch
        // (CFG.train.batch_size * get_world_size())
    )
    if CFG.train.start_epoch.is_some():
        start_epoch = CFG.train.start_epoch.unwrap()
    else:
        start_epoch = 0
    epoch = start_epoch

    if not os.path.exists(CFG.train.snapshot_dir) and get_rank() == 0:
        os.makedirs(CFG.train.snapshot_dir)

    # LOG.info(f"model\n{describe(model.module)}")
    t0 = time.time()
    for idx, data in enumerate(train_loader):
        data: TrkDataLoaderData
        if epoch != idx // num_per_epoch + start_epoch:
            epoch = idx // num_per_epoch + start_epoch

            if get_rank() == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(
                        CFG.train.snapshot_dir,
                        f"checkpoint_e{epoch}.pth",
                    ),
                )

            if epoch == CFG.train.epoch:
                return

            if CFG.backbone.train_epoch == epoch:
                LOG.info("start training backbone.")
                optimizer, lr_scheduler = build_opt_lr(model.module, epoch)

            lr_scheduler.step(epoch)
            cur_lr = lr_scheduler.get_lr()[0]
            LOG.info(f"epoch: {epoch + 1}")

        if idx % num_per_epoch == 0 and idx != 0:
            for i, pg in enumerate(optimizer.param_groups):
                LOG.info(f"epoch: {epoch + 1}, lr group {i}: {pg['lr']}")
                if get_rank() == 0 and tb_writer is not None:
                    tb_writer.add_scalar(f"lr/group{i}", pg["lr"], idx)
        data_time = average_reduce(time.time() - t0)
        if get_rank() == 0 and tb_writer is not None:
            tb_writer.add_scalar("time/data", data_time, idx)

        if train_processors is not None:
            for tp in train_processors:
                data = tp(data)

        outputs: MB_M_Output = model.forward(data)

        loss = outputs["total_loss"]  # type: ignore

        if is_valid_number(loss.data.item()):
            optimizer.zero_grad()
            loss.backward()
            reduce_gradients(model)

            if get_rank() == 0 and CFG.train.log_grads:
                if tb_writer is not None:
                    log_grads(model.module, tb_writer, idx)

            # clip gradient
            clip_grad_norm_(model.parameters(), CFG.train.grad_clip)
            optimizer.step()

        batch_time = time.time() - t0
        batch_info = {}
        batch_info["batch_time"] = average_reduce(batch_time)
        batch_info["data_time"] = average_reduce(data_time)
        for k, v in sorted(outputs.items()):
            if not k.endswith("loss"):
                continue
            if isinstance(v, torch.Tensor):
                try:
                    batch_info[k] = average_reduce(v.data.item())
                except Exception as e:
                    LOG.error(f'error in output["{k}"]: {list(v.shape)}')
                    raise e
            elif v is None:
                continue  # FIXME: mask_loss
            else:
                raise TypeError(f"invalid type of {k}: {type(v)}")

        average_meter.update(**batch_info)

        if get_rank() == 0:
            if tb_writer is not None:
                for k, v in batch_info.items():
                    tb_writer.add_scalar(k, v, idx)

            if (idx + 1) % CFG.train.print_freq == 0:
                info = f"Epoch: [{epoch + 1}][{idx + 1}/{num_per_epoch}], lr: {cur_lr:.6f}"
                for cc, (k, v) in enumerate(batch_info.items()):
                    if cc % 2 == 0:
                        info += f"\t{getattr(average_meter,k):s}\t"
                    else:
                        info += f"{getattr(average_meter,k):s}\n"
                LOG.info(info)
                print_speed(
                    idx + 1 + start_epoch * num_per_epoch,
                    average_meter.batch_time.avg,
                    CFG.train.epoch * num_per_epoch,
                    logger_name=_LOGGER_NAME,
                )  # TODO: rewrite this
        t0 = time.time()

    if get_rank() == 0:
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(
                CFG.train.snapshot_dir,
                f"checkpoint_e{epoch}_done.pth",
            ),
        )

    LOG.info(f'train done: {CFG.meta_arc}')


if __name__ == "__main__":
    dist_init()

    args = Cap(TrainCliArgs).parse().value

    seed_torch(args.seed)

    config_path = args.config
    if not os.path.exists(config_path):
        LOG.fatal(f"config file {config_path} not found")
        exit(1)

    read_config(config_path, CFG)
    CFG.train.batch_size = args.batch_size

    if args.test:
        LOG.warning("WE ARE IN TEST MODE")
        CFG.train.snapshot_dir = os.path.join(
            ENV.experiments_path, "saved", "test_rm"
        )

    # logging stuff
    log_dir = CFG.train.log_dir.unwrap_or(CFG.train.snapshot_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_fp = os.path.join(log_dir, "log.txt")
    if os.path.exists(log_fp):
        if args.test:
            LOG.warning(f"remove {log_fp}")
            os.remove(log_fp)
        else:
            LOG.warning(
                f"Detected existing log file, override snapdir? [y/N]\n{log_dir}"
            )
            if input().lower() == "y":
                os.remove(log_fp)
            else:
                LOG.fatal("training aborted.")
                exit(1)
    for name in ["global", _LOGGER_NAME]:
        logging.add_file_handler(name, log_fp, append=True)

    model = ModelBuilder().cuda().train()

    # TODO: add resume
    if CFG.backbone.pretrained.is_some():
        pretrained_path = CFG.backbone.pretrained.unwrap()
        LOG.warning(f"loading pretrained backbone from: {pretrained_path}")
        load_pretrain(model.backbone, pretrained_path)
        LOG.warning("pretrained backbone loaded.")
        time.sleep(1)

    if CFG.train.pretrained.is_some():
        pretrained_path = CFG.train.pretrained.unwrap()
        LOG.warning(f"loading pretrained model from: {pretrained_path}")
        load_pretrain(model, pretrained_path)
        LOG.warning("pretrained model loaded.")
        time.sleep(1)

    dataloader = build_dataloader(TrkDataset)
    if CFG.train.start_epoch.is_some():
        optimizer, scheduler = build_opt_lr(
            model,
            CFG.train.start_epoch.unwrap(),
        )
    else:
        optimizer, scheduler = build_opt_lr(model)

    dist_model = DistModule(model)
    LOG.info(scheduler)
    LOG.info("model prepare done")

    # TODO: fixme
    processors: List[TrainingProcessor] = [
        # CSATrainingProcessor(dist_model),
        # TATK_FGSMTrainingProcessor(dist_model),
        # TATK_BIMTrainingProcessor(dist_model),
        # TATK_PGDTrainingProcessor(dist_model),
    ]

    for ti, tp in enumerate(processors):
        LOG.warning(
            f"using training processor: {tp.__class__.__name__} ({ti+1} of {len(processors)}"
        )
    if len(processors) == 0:
        LOG.warning("no training processor is used.")

    train(
        dataloader,
        dist_model,
        optimizer,
        scheduler,
        train_processors=processors,
    )
