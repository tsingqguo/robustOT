import clip
import os
import torch
import torch.nn as nn
import yaml
from clip.model import CLIP
from LRR import utils
from LRR.datasets.wrappers import SRImplicitPairedAdv
from LRR.models.rsn import ResampleCNN, ResampleMLP
from LRR.models import make as make_models
from LRR.models.liif import STIR
from LRR.tools.test import eval_psnr
from LRR.tools.train import make_data_loader
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
from typed_cap import Cap
from typing import Literal

ResampleNet = ResampleCNN | ResampleMLP


class Args:
    # @alias=c
    config: str
    """fp to yml; only used part of config"""

    # @alias=n
    name: str

    trial: str

    # @alias=s
    saved_stir: str

    # @alias=C
    careless: bool = False
    """only test last frame in N length"""

    # @alias=t
    rsn_type: Literal["mlp", "cnn"] = "mlp"

    # @alias=T
    text_guidance: bool = False
    """use clip's text encoder to guide"""

    text_template: str = "A photo of a {}."

    layers: list[int] = [256, 32]

    # @alias=V
    visualization: bool = False


class Config:
    epoch_max: int = 5
    epoch_val: int = 1
    epoch_save: int = 1
    optimizer: dict = {"name": "adam", "args": {"lr": 1e-4}}
    multi_step_lr: dict = {"milestones": [200, 400, 600, 800], "gamma": 0.5}


def make_stir(saved: str) -> STIR:
    net = make_models(
        torch.load(saved)["model"],
        # args=model_args,
        load_sd=True,
    )
    return net  # type: ignore


def get_LRR_save(rsn: ResampleNet, saved_stir_fp: str):
    if isinstance(rsn, ResampleCNN):
        rsn_type = "cnn"
    else:
        rsn_type = "mlp"

    rsn_spec = {
        "rsn_type": rsn_type,
        "feat_dim": rsn.feat_dim,
        "n_length": rsn.n_length,
        "layers": rsn.layers,
        "text_dim": rsn.text_dim,
        "clip_model": "ViT-B/32",  # fixed
    }
    stir = torch.load(saved_stir_fp)
    stir_spec = {"name": "stir", "args": stir["model"]["args"]}
    sd = {}

    for k, v in stir["model"]["sd"].items():
        sd[f"stir.{k}"] = v

    for k, v in model_rsn.state_dict().items():
        sd[f"rsn.{k}"] = v

    return {
        "stir_spec": stir_spec,
        "rsn_spec": rsn_spec,
        "sd": sd,
    }


def make_data_loaders(yml_config: dict):
    train_loader = make_data_loader(
        yml_config.get("train_dataset"), tag="train"
    )
    val_loader = make_data_loader(yml_config.get("val_dataset"), tag="val")
    return train_loader, val_loader


def build_schedular(model: nn.Module, cfg: Config):
    optimizer = utils.make_optimizer(model.parameters(), cfg.optimizer)
    lr_scheduler = MultiStepLR(optimizer, **cfg.multi_step_lr)
    return optimizer, lr_scheduler


def train(
    yml_config: dict,
    train_loader,
    model_stir: utils.DataParallel[STIR] | STIR,
    model_rsn: utils.DataParallel[ResampleNet] | ResampleNet,
    model_clip: utils.DataParallel[CLIP] | CLIP | None,
    text_template: str | None,
    optimizer: Optimizer,
    only_care_last_nf: bool = False,
    visualization: bool = False,
):
    model_rsn = model_rsn.cuda().train()
    model_stir = model_stir.cuda().eval()
    if model_clip is not None:
        model_clip = model_clip.cuda()
        model_clip.eval()

    # FIXME:
    for name, params in model_stir.named_parameters():
        # print(f'[liif] {name}.grad: {params.requires_grad}')
        params.requires_grad = False
    for name, params in model_rsn.named_parameters():
        # print(f'[rsn] {name}.grad: {params.requires_grad}')
        params.requires_grad = True
    if model_clip is not None:
        for name, params in model_clip.named_parameters():
            # print(f"[clip] {name}.grad: {params.requires_grad}")
            params.requires_grad = False

    loss_fn = nn.L1Loss()
    train_losses = {}
    train_losses["purifier_loss"] = utils.Averager()
    train_losses["loss_all"] = utils.Averager()

    data_norm = yml_config["data_norm"]
    t = data_norm["inp"]
    inp_sub = torch.FloatTensor(t["sub"]).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t["div"]).view(1, -1, 1, 1).cuda()
    t = data_norm["gt"]
    gt_sub = torch.FloatTensor(t["sub"]).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t["div"]).view(1, 1, -1).cuda()

    pbar = tqdm(train_loader, leave=False, desc="[train]", ncols=80)
    for batch in pbar:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda()
            # else:
            #     print(f"[debug] non-tensor value: {v}")

        inp = (batch["inp"] - inp_sub) / inp_div

        gt = (batch["gt"] - gt_sub) / gt_div

        if isinstance(model_stir, nn.DataParallel):
            _model_liif = model_stir.module
        else:
            _model_liif = model_stir

        # with torch.no_grad():
        #     # pred = model_liif(inp, batch["coord"], batch["cell"])
        #     _model_liif.gen_feat(inp)
        #     feat_unfold = _model_liif.feat_unfolding(_model_liif.feat)
        _model_liif.gen_feat(inp)
        feat_unfold = _model_liif.feat_unfolding(_model_liif.feat)

        if model_clip is not None:
            if isinstance(model_clip, nn.DataParallel):
                _model_clip = model_clip.module
            else:
                _model_clip = model_clip
            if text_template is None:
                text = batch["obj_cls"]
            else:
                text = [text_template.format(t) for t in batch["obj_cls"]]
            if pbar.n == 0:
                pbar.write(f"[debug] text: {text}")
            text = clip.tokenize(texts=text, context_length=77)
            text = text.to(inp.device)
            text = _model_clip.encode_text(text)
        else:
            text = None

        coord_offsets = model_rsn.forward(feat_unfold, text=text)
        if batch["coord"].ndim == 3 and coord_offsets.ndim == 4:
            coord_offsets = coord_offsets.squeeze(1)
        coord = batch["coord"] + coord_offsets

        # with torch.no_grad():
        pred = _model_liif.query_rgb_2t3(
            coord, batch["cell"], feat_unfolded=feat_unfold
        )

        if only_care_last_nf:
            b = gt.shape[0]
            gt = gt.view(b, 5, -1, 3)
            gt = gt[:, -1, :, :]
            gt = gt.view(b, -1, 3)
            pred = pred.view(b, 5, -1, 3)
            pred = pred[:, -1, :, :]
            pred = pred.view(b, -1, 3)
            # print('[debug] gt.shape  :', list(gt.shape))
            # print('[debug] pred.shape:', list(pred.shape))

        purifier_loss = loss_fn(pred, gt)
        train_losses["purifier_loss"].add(purifier_loss.item())
        loss = purifier_loss

        # for name, params in _model_liif.named_parameters():
        #     print(f'[liif] {name}.grad: {params.requires_grad}')
        if visualization and pbar.n % 50 == 0:
            import cv2
            from pyotp.utils import save_imgs, tensor2mat

            pred_vis = (
                # pred.clamp(0, 1).reshape(-1, 128, 128, 3).permute(0, 3, 1, 2)
                pred.clamp(0, 1)
                .reshape(-1, 100, 100, 3)
                .permute(0, 3, 1, 2)
            )
            pred_vis = tensor2mat(
                pred_vis * 255,
                allow_batch=True,
                color_cvt=cv2.COLOR_BGR2RGB,
            )
            save_imgs(pred_vis, "train_rsn_preview.png")

        pbar.desc = f"[train] p_loss = {loss.item():.4f}"

        train_losses["loss_all"].add(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = None
        loss = None

    return train_losses


if __name__ == "__main__":
    cap = Cap(Args)
    args = cap.parse().val

    with open(args.config, "r") as f:
        yml_config: dict = yaml.load(f, Loader=yaml.FullLoader)

    save_dir = os.path.join(
        os.environ["PYOTP_EXP"],
        "lrr_saves",
        args.name,
        f"trial_{args.trial}",
    )
    os.environ["CVR_EXP_DIR"] = save_dir
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir, exist_ok=False)

    log, writer = utils.set_save_path(save_dir)

    train_loader, val_loader = make_data_loaders(yml_config)

    if train_loader is None or val_loader is None:
        print("[fatal] train_loader or val_loader is None")
        exit(1)

    if yml_config.get("data_norm") is None:
        yml_config["data_norm"] = {
            "inp": {"sub": [0], "div": [1]},
            "gt": {"sub": [0], "div": [1]},
        }

    if args.text_guidance:
        models = clip.available_models()
        model_clip, _ = clip.load("ViT-B/32", jit=False)
        for loader in [train_loader, val_loader]:
            db = loader.dataset
            if isinstance(db, SRImplicitPairedAdv):
                db.include_obj_cls = True
                db.fake_obj_cls = "object"
            else:
                print("[fatal] loader.dataset is not SRImplicitPairedAdv")
                exit(1)
    else:
        model_clip = None

    model_stir = make_stir(args.saved_stir)
    if model_stir.multiplier is None:
        raise ValueError("model_liif must be unfold enabled")
    if args.rsn_type == "mlp":
        model_rsn = ResampleMLP(
            feat_dim=model_stir.multiplier * model_stir.encoder.out_dim,
            n_length=1,
            hidden_list=args.layers,
            text_dim=0 if model_clip is None else 512,
        )
    elif args.rsn_type == "cnn":
        model_rsn = ResampleCNN(
            feat_dim=model_stir.multiplier * model_stir.encoder.out_dim,
            n_length=1,
            layers=args.layers,
            text_dim=0 if model_clip is None else 512,
        )
    else:
        raise NotImplementedError

    print(f"[INFO] using model: {model_rsn.__class__.__name__}")
    print(model_rsn)

    cfg = Config()
    optimizer, lr_scheduler = build_schedular(model_rsn, cfg)

    n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    if n_gpus > 1:
        model_stir = utils.DataParallel(model_stir)
        model_rsn = utils.DataParallel(model_rsn)
        if model_clip is not None:
            model_clip = utils.DataParallel(model_clip)

    max_val_v = -1e18

    timer = utils.Timer()
    epoch_start = 1

    for epoch in range(epoch_start, cfg.epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ["epoch {}/{}".format(epoch, cfg.epoch_max)]

        train_losses = train(
            yml_config,
            train_loader,
            model_stir,
            model_rsn,
            model_clip,
            args.text_template,
            optimizer,
            only_care_last_nf=args.careless,
            visualization=args.visualization,
        )

        if lr_scheduler is not None:
            lr_scheduler.step()

        for k in train_losses.keys():
            log_info.append(
                "train " + k + " : loss={:.4f}".format(train_losses[k].item())
            )

        save = get_LRR_save(
            model_rsn.module
            if isinstance(model_rsn, nn.DataParallel)
            else model_rsn,
            args.saved_stir,
        )
        torch.save(save, os.path.join(save_dir, "lrr-epoch-last.pt"))

        if (cfg.epoch_save is not None) and (epoch % cfg.epoch_save == 0):
            torch.save(
                save, os.path.join(save_dir, "lrr-epoch-{}.pth".format(epoch))
            )

        if (cfg.epoch_val is not None) and (epoch % cfg.epoch_val == 0):

            def coord_fn(
                coord: torch.Tensor,
                feat: torch.Tensor,
                text: torch.Tensor | None,
            ):
                offset = model_rsn.forward(feat, text)
                if coord.ndim == 3 and offset.ndim == 4:
                    offset = offset.squeeze(1)
                coord = coord + offset
                return coord

            val_res = eval_psnr(
                val_loader,
                model_stir,
                model_clip=model_clip,
                data_norm=yml_config["data_norm"],
                coord_fn=coord_fn,
                only_care_last_nf=args.careless,
            )
            log_info.append("val: psnr={:.4f}".format(val_res))
            # writer.add_scalars("psnr", {"val": val_res}, epoch)
            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(save, os.path.join(save_dir, "lrr-epoch-best.pth"))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (cfg.epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append("{} {}/{}".format(t_epoch, t_elapsed, t_all))

        print(", ".join(log_info))
