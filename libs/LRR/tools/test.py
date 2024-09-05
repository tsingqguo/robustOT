import clip
import torch
import torch.nn as nn
from clip.model import CLIP
from LRR.utils import Averager, DataParallel, calc_psnr
from LRR.models.liif import LIIF
from tqdm import tqdm
from typing import Callable


def eval_psnr(
    loader,
    model: DataParallel[LIIF] | LIIF,
    model_clip: DataParallel[CLIP] | CLIP | None = None,
    data_norm: dict | None = None,
    coord_fn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor
    ]
    | None = None,
    only_care_last_nf: bool = False,
):
    model.eval()

    if only_care_last_nf:
        print("[WARN] Only testing last frame in N length !!!")

    if data_norm is None:
        data_norm = {
            "inp": {"sub": [0], "div": [1]},
            "gt": {"sub": [0], "div": [1]},
        }
    t = data_norm["inp"]
    inp_sub = torch.FloatTensor(t["sub"]).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t["div"]).view(1, -1, 1, 1).cuda()
    t = data_norm["gt"]
    gt_sub = torch.FloatTensor(t["sub"]).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t["div"]).view(1, 1, -1).cuda()

    metric_fn = calc_psnr

    val_res = Averager()

    pbar = tqdm(loader, leave=False, desc="[val]", ncols=80)
    for batch in pbar:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda()

        inp = (batch["inp"] - inp_sub) / inp_div

        if isinstance(model, nn.DataParallel):
            _model = model.module
        else:
            _model = model

        with torch.no_grad():
            _model.gen_feat(inp)
            feat_unfold = _model.feat_unfolding(_model.feat)

            if model_clip is not None:
                if isinstance(model_clip, nn.DataParallel):
                    _model_clip = model_clip.module
                else:
                    _model_clip = model_clip
                text = clip.tokenize(texts=batch["obj_cls"], context_length=77)
                text = text.to(inp.device)
                text = _model_clip.encode_text(text)
            else:
                text = None

            if coord_fn is not None:
                coord = coord_fn(batch["coord"], feat_unfold, text)
            else:
                coord = batch["coord"]

            pred = _model.query_rgb_2t3(coord, batch["cell"], feat_unfold)

        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        gt = batch["gt"]
        if only_care_last_nf:
            b = gt.shape[0]
            gt = gt.view(b, 5, -1, 3)
            gt = gt[:, -1, :, :]
            gt = gt.view(b, -1, 3)
            pred = pred.view(b, 5, -1, 3)
            pred = pred[:, -1, :, :]
            pred = pred.view(b, -1, 3)

        res = metric_fn(pred, gt)
        val_res.add(res.item(), inp.shape[0])

        pbar.desc = f"[val] p_loss = {val_res.item():.4f}"

    return val_res.item()
