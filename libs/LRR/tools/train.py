import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import yaml
from LRR import datasets, models, utils
from LRR.utils.test import eval_psnr
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR


def make_data_loader(spec, tag="", log_fn=print):
    if spec is None:
        return None

    dataset = datasets.make(spec["dataset"])
    dataset = datasets.make(spec["wrapper"], args={"dataset": dataset})

    log_fn(f"{tag} dataset: size = {len(dataset)}")
    for k, v in dataset[0].items():
        if isinstance(v, torch.Tensor):
            log_fn(f"\t{k:<7}: shape = {tuple(v.shape)}")
        else:
            log_fn(f"\t{k:<7}: {type(v)}")

    loader = DataLoader(
        dataset,
        batch_size=spec["batch_size"],
        shuffle=(tag == "train"),
        num_workers=2,
        pin_memory=True,
    )
    return loader


def make_data_loaders(log_fn=print):
    train_loader = make_data_loader(
        config.get("train_dataset"), tag="train", log_fn=log_fn
    )
    val_loader = make_data_loader(
        config.get("val_dataset"), tag="val", log_fn=log_fn
    )
    return train_loader, val_loader


def prepare_training():
    if config.get("resume") is not None:
        sv_file = torch.load(config["resume"])
        model = models.make(sv_file["model"], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file["optimizer"], load_sd=True
        )
        epoch_start = sv_file["epoch"] + 1
        if config.get("multi_step_lr") is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config["multi_step_lr"])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
    else:
        model = models.make(config["model"]).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config["optimizer"]
        )
        epoch_start = 1
        if config.get("multi_step_lr") is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config["multi_step_lr"])

    log("model: #params={}".format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, optimizer):
    model.train()
    loss_fn = nn.L1Loss()
    train_losses = {}
    train_losses["purifier_loss"] = utils.Averager()
    train_losses["loss_all"] = utils.Averager()

    if config["L2_reg"]["flag"]:
        train_losses["L2_reg_loss"] = utils.Averager()
    if config["gram_reg"]["flag"]:
        train_losses["gram_reg_loss"] = utils.Averager()

    data_norm = config["data_norm"]
    t = data_norm["inp"]
    inp_sub = torch.FloatTensor(t["sub"]).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t["div"]).view(1, -1, 1, 1).cuda()
    t = data_norm["gt"]
    gt_sub = torch.FloatTensor(t["sub"]).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t["div"]).view(1, 1, -1).cuda()

    debug_latency = False  # TODO:
    import time

    t_pool = [time.time()]

    pbar = tqdm(train_loader, leave=False, desc="[train]", ncols=80)
    for batch in pbar:
        if debug_latency:
            t_pool.append(time.time())
            if len(t_pool) % 1 == 0:
                print(f"avg latency: {np.mean(np.diff(t_pool))}s")
        else:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()

            inp = (batch["inp"] - inp_sub) / inp_div
            gt = (batch["gt"] - gt_sub) / gt_div

            pred = model(inp, batch["coord"], batch["cell"])

            purifier_loss = loss_fn(pred, gt)
            train_losses["purifier_loss"].add(purifier_loss.item())
            loss = purifier_loss

            pbar.desc = f"[train] p_loss = {loss.item():.4f}"

            train_losses["loss_all"].add(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = None
            loss = None

            if args.debug:
                break

    return train_losses


def main(config_, save_path):
    os.environ["CVR_EXP_DIR"] = save_path

    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, "config.yaml"), "w") as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders(log_fn=log)
    if config.get("data_norm") is None:
        config["data_norm"] = {
            "inp": {"sub": [0], "div": [1]},
            "gt": {"sub": [0], "div": [1]},
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config["epoch_max"]
    epoch_val = config.get("epoch_val")
    epoch_save = config.get("epoch_save")
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ["epoch {}/{}".format(epoch, epoch_max)]

        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        train_losses = train(train_loader, model, optimizer)
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k in train_losses.keys():
            log_info.append(
                "train " + k + " : loss={:.4f}".format(train_losses[k].item())
            )
            writer.add_scalars(k, {"train": train_losses[k].item()}, epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config["model"]
        model_spec["sd"] = model_.state_dict()
        optimizer_spec = config["optimizer"]
        optimizer_spec["sd"] = optimizer.state_dict()
        sv_file = {
            "model": model_spec,
            "optimizer": optimizer_spec,
            "epoch": epoch,
        }

        torch.save(sv_file, os.path.join(save_path, "epoch-last.pth"))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(
                sv_file, os.path.join(save_path, "epoch-{}.pth".format(epoch))
            )

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1 and (config.get("eval_bsize") is not None):
                model_ = model.module
            else:
                model_ = model
            val_res = eval_psnr(
                val_loader,
                model_,
                data_norm=config["data_norm"],
                eval_type=config.get("eval_type"),
                eval_bsize=config.get("eval_bsize"),
            )
            log_info.append("val: psnr={:.4f}".format(val_res))
            writer.add_scalars("psnr", {"val": val_res}, epoch)
            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, "epoch-best.pth"))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append("{} {}/{}".format(t_epoch, t_elapsed, t_all))

        log(", ".join(log_info))
        writer.flush()

        if args.debug:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--name", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--trial", type=str)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print("config loaded.")

    save_dir = "lrr_saves" if not args.debug else "lrr_saves_debug"
    save_name = os.path.join(save_dir, args.name)

    if args.tag is not None:
        save_name += "_" + args.tag

    if not os.path.exists(save_name):
        os.makedirs(save_name, exist_ok=False)
    save_path = os.path.join(
        os.environ["PYOTP_EXP"],
        save_dir,
        args.name,
        f"trial_{args.trial}",
    )
    if os.path.exists(save_path):
        print("Experiment trial exists. Please double check.")
        if args.debug:
            main(config, save_path)
        if args.trial == "999":
            main(config, save_path)
    else:
        main(config, save_path)
