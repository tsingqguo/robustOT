import numpy as np
import numpy.typing as npt
import torch
import random
import torch.nn.functional as F
import visdom
from attackers.spark.SPARK.cfg import CFG
import matplotlib.pyplot as plt
from torch import Tensor
from attackers.spark.SPARK.types import AtkType, NormType, RegType
from pysot.datasets.anchor_target_cfgmod import AnchorTarget
from pysot.models import _MB_Track_R
from pysot.models.loss import select_cross_entropy_loss
from pysot.tracker.base_tracker import BaseTracker
from pysot.utils.bbox import center2corner, Center, get_axis_aligned_bbox
from math import cos, sin, pi
import os
import cv2

from typing import Annotated, Dict, List, Literal, Optional, TypedDict


class AtkData(TypedDict):
    template_zf: Tensor
    search: Tensor
    pert: Tensor
    prev_perts: Tensor
    adv_cls: Tensor
    weights: Tensor
    momen: Tensor


class OIMAttacker:
    atk_type: AtkType
    eplison: int
    inta: int
    norm_type: NormType
    reg_type: RegType
    max_num: Annotated[int, "max iteration"]
    lamb: float
    v_id: int
    apts_num: int
    target_traj: List
    # prev_delta: Optional
    tacc: bool
    meta_path: List
    acc_iters: int
    weights: List
    weight_eplison: int
    lamb_momen: int
    target_pos: List
    accframes: int

    def __init__(
        self,
        atk_type,
        max_num=10,
        eplison=1,
        inta=10,
        lamb=0.00001,
        norm_type=NormType.L_inf,
        apts_num=2,
        reg_type=RegType.Weighted,
        accframes=30,
    ):
        self.atk_type = atk_type
        # parameters for bim
        self.eplison = eplison
        self.inta = inta
        self.norm_type = norm_type
        self.reg_type = reg_type
        self.max_num = max_num
        self.lamb = lamb
        self.v_id = 0
        # self.st = st()
        self.apts_num = apts_num
        self.target_traj = []
        self.prev_delta = None
        self.tacc = True
        self.meta_path = []
        self.acc_iters = 0
        self.weights = []
        self.weight_eplison = 1
        self.lamb_momen = 1
        self.target_pos = []
        self.accframes = accframes

    def save_tensorasimg(self, path, var):
        np_var = torch.squeeze(var).detach().cpu()
        if len(np_var.shape) == 2:
            np_var = np_var.view(1, np_var.shape[0], np_var.shape[1])
        np_var = np_var.numpy()
        np_var = np.transpose(np_var, (1, 2, 0))
        cv2.imwrite(path, np_var, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    def attack(
        self,
        tracker: BaseTracker,
        img: np.ndarray,
        prev_perts: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        APTS: Optional[bool] = False,
        OPTICAL_FLOW: Optional[bool] = False,
        ADAPT: Optional[bool] = False,
        Enable_same_prev=True,
    ):
        """
        args:
            tracker, img(np.ndarray): BGR image
        return:
            adversarial image
        """
        # print("[DEBUG] called attack") # REMOVE
        w_z: np.float64 = tracker.size[0] + CFG.track.context_amount * np.sum(
            tracker.size
        )
        h_z: np.float64 = tracker.size[1] + CFG.track.context_amount * np.sum(
            tracker.size
        )
        s_z: np.float64 = np.sqrt(w_z * h_z)
        scale_z: np.float64 = CFG.track.exemplar_size / s_z
        s_x: np.float64 = s_z * (
            CFG.track.instance_size / CFG.track.exemplar_size
        )
        x_crop = tracker.get_subwindow(
            img,
            tracker.center_pos,
            CFG.track.instance_size,
            round(s_x),
            tracker.channel_average,
        )
        outputs = tracker.model.track(x_crop)
        cls = tracker.model.log_softmax(outputs["cls"])
        diff_cls = cls[:, :, :, :, 1] - cls[:, :, :, :, 0]
        label_cls = diff_cls.ge(0).float()

        if self.atk_type is AtkType.UA:
            adv_cls, same_prev = self.ua_label(tracker, scale_z, outputs)
            adv_cls = adv_cls.long()
        elif self.atk_type is AtkType.TA:
            adv_cls, same_prev = self.ta_label(tracker, scale_z, outputs)
            adv_cls = adv_cls.long()
        else:
            raise NotImplementedError(f"unknown attack type: {self.atk_type}")

        max_iteration = self.max_num

        # initial perturbation tensor
        if CFG.cuda:
            pert = torch.zeros(x_crop.size()).cuda()
        else:
            raise NotImplementedError  # TODO: 2023-05-11
            pert = torch.zeros(x_crop.size())

        if prev_perts is None or (
            same_prev == False and Enable_same_prev == True
        ):
            # print('[DEBUG] prev_perts is None or same_prev == False ...')  # REMOVE
            if CFG.cuda:
                prev_perts = torch.zeros(x_crop.size()).cuda()
                weights = torch.ones(
                    x_crop.size()
                ).cuda()  # torch.ones(1).cuda()
            else:
                prev_perts = torch.zeros(x_crop.size())
                weights = torch.ones(x_crop.size())  # torch.ones(1)
        else:
            if APTS == False:
                # print('[DEBUG] APTS == False ...')  # REMOVE
                if self.reg_type is RegType.Weighted:
                    pert_sum = torch.mul(weights, prev_perts).sum(0)
                else:
                    pert_sum = prev_perts.sum(0)
                adv_x_crop = x_crop + pert_sum
                adv_x_crop = torch.clamp(adv_x_crop, 0, 255)
                pert_true = adv_x_crop - x_crop

                adv_img: Optional[np.ndarray] = None
                if CFG.attacker.gen_adv_img:
                    adv_img = tracker.get_origin_img(
                        img,
                        x_crop,
                        tracker.center_pos,
                        CFG.track.instance_size,
                        round(s_x),
                        tracker.channel_average,
                    )

                if CFG.attacker.save_meta:
                    raise NotImplementedError  # TODO: 2023-05-11
                    # validate the score
                    outputs = tracker.model.track(adv_x_crop)
                    disp_score = self.log_softmax(outputs["cls"].cpu())
                    disp_score = disp_score[:, :, :, :, 1].permute(1, 0, 2, 3)
                    disp_score_min = torch.min(torch.min(disp_score, 2)[0], 2)[
                        0
                    ]
                    disp_score_max = torch.max(torch.max(disp_score, 2)[0], 2)[
                        0
                    ]
                    dispmin = torch.zeros(disp_score.size())
                    dispmax = torch.zeros(disp_score.size())
                    for i in range(4):
                        dispmin[i, :, :, :] = disp_score_min[i].repeat(
                            1, 1, disp_score.shape[2], disp_score.shape[3]
                        )
                        dispmax[i, :, :, :] = disp_score_max[i].repeat(
                            1, 1, disp_score.shape[2], disp_score.shape[3]
                        )
                    disp_score = (
                        (disp_score - dispmin) / (dispmax - dispmin) * 255
                    )
                    #
                    res_path = os.path.join(
                        self.meta_path, str(self.v_id) + "_xcrop.jpg"
                    )
                    self.save_tensorasimg(res_path, x_crop)
                    res_path = os.path.join(
                        self.meta_path, str(self.v_id) + "_adv_xcrop.jpg"
                    )
                    self.save_tensorasimg(res_path, adv_x_crop)
                    res_path = os.path.join(
                        self.meta_path, str(self.v_id) + "_pert.jpg"
                    )
                    self.save_tensorasimg(res_path, pert_true)
                    res_path = os.path.join(
                        self.meta_path, str(self.v_id) + "_pert_sum.jpg"
                    )
                    self.save_tensorasimg(res_path, pert_true)
                    for i in range(disp_score.shape[0]):
                        res_path = os.path.join(
                            self.meta_path,
                            str(self.v_id) + "_score" + str(i) + ".jpg",
                        )
                        self.save_tensorasimg(res_path, disp_score[i, :, :, :])

                return x_crop, pert_true, prev_perts, weights, adv_img
            else:  # no APTS
                # print('[DEBUG] no APTS ...')  # REMOVE
                max_iteration = self.apts_num
                if self.tacc == False:
                    if CFG.cuda:
                        prev_perts = torch.zeros(x_crop.size()).cuda()
                        weights = torch.ones(
                            x_crop.size()
                        ).cuda()  # torch.ones(1).cuda()
                    else:
                        prev_perts = torch.zeros(x_crop.size())
                        weights = torch.ones(x_crop.size())  # torch.ones(1)

        if CFG.attacker.show:
            raise NotImplementedError  # TODO: 2023-05-11
            vis = visdom.Visdom(
                env="Adversarial Example Showing"
            )  # (server='172.28.144.132',port=8022,env='Adversarial Example Showing')
            vis.images(x_crop, win="X_org")
            vis.images(adv_cls.permute(1, 0, 2, 3), win="adv_cls")

        # start attack
        losses: List[Tensor] = []
        m = 0

        if CFG.cuda:
            momen = torch.zeros(x_crop.size()).cuda()
        else:
            momen = torch.zeros(x_crop.size())

        self.acc_iters += max_iteration
        # _iter = 0  # REMOVE
        # print(f'[DEBUG] m: {m}; max_iteration: {max_iteration}') # REMOVE
        while m < max_iteration:
            # _iter += 1 # REMOVE
            if isinstance(tracker.model.zf, list):
                zf = torch.cat(tracker.model.zf, 0)
            else:
                zf = tracker.model.zf

            _d = {
                "template_zf": zf.detach(),
                "search": x_crop.detach(),
                "pert": pert.detach(),
                "prev_perts": prev_perts.detach(),
                "label_cls": label_cls.detach(),
                "adv_cls": adv_cls.detach(),
                "weights": weights.detach(),
                "momen": momen.detach(),
            }
            data: AtkData = _d  # type: ignore

            data["pert"].requires_grad = True
            # data['weights'].requires_grad = True
            pert, loss, update_cls, momen, weights = self.oim_once(
                tracker, data
            )
            losses.append(loss)
            m += 1

            # disp x_crop for each iteration and the pert
            if CFG.attacker.show:
                raise NotImplementedError  # TODO: 2023-05-11
                if self.reg_type is RegType.Weighted:
                    pert_sum = torch.mul(weights, prev_perts).sum(0)
                else:
                    pert_sum = prev_perts.sum(0)
                x_crop_t = x_crop + pert_sum + pert
                x_crop_t = torch.clamp(x_crop_t, 0, 255)
                vis.images(x_crop_t, win="X_attack")
                vis.images(pert, win="Pert_incremental")
                vis.images(pert_sum + pert, win="Pert_attack")
                plt.plot(losses)
                plt.ylabel("Loss")
                plt.xlabel("Iteration")
                vis.matplot(plt, win="Loss_attack")
                # validate the score
                outputs = tracker.model.track(x_crop_t)
                disp_score = self.log_softmax(outputs["cls"].cpu())
                disp_score = disp_score[:, :, :, :, 1].permute(1, 0, 2, 3)
                disp_score_min = torch.min(torch.min(disp_score, 2)[0], 2)[0]
                disp_score_max = torch.max(torch.max(disp_score, 2)[0], 2)[0]
                dispmin = torch.zeros(disp_score.size())
                dispmax = torch.zeros(disp_score.size())
                for i in range(4):
                    dispmin[i, :, :, :] = disp_score_min[i].repeat(
                        1, 1, disp_score.shape[2], disp_score.shape[3]
                    )
                    dispmax[i, :, :, :] = disp_score_max[i].repeat(
                        1, 1, disp_score.shape[2], disp_score.shape[3]
                    )
                disp_score = (disp_score - dispmin) / (dispmax - dispmin) * 255
                vis.images(disp_score, win="Response_attack")

        # print('[DEBUG] iter:', _iter)  # REMOVE

        if CFG.attacker.save_meta:
            raise NotImplementedError  # TODO: 2023-05-11
            if self.reg_type is RegType.Weighted:
                pert_sum = torch.mul(weights, prev_perts).sum(0)
            else:
                pert_sum = prev_perts.sum(0)
            x_crop_t = x_crop + pert_sum + pert
            x_crop_t = torch.clamp(x_crop_t, 0, 255)
            # validate the score
            outputs = tracker.model.track(x_crop_t)
            disp_score = self.log_softmax(outputs["cls"].cpu())
            disp_score = self.norm_score(disp_score)
            # original score
            outputs = tracker.model.track(x_crop)
            disp_score_org = self.log_softmax(outputs["cls"].cpu())
            disp_score_org = self.norm_score(disp_score_org)
            #
            res_path = os.path.join(
                self.meta_path, str(self.v_id) + "_xcrop.jpg"
            )
            self.save_tensorasimg(res_path, x_crop)
            res_path = os.path.join(
                self.meta_path, str(self.v_id) + "_adv_xcrop.jpg"
            )
            self.save_tensorasimg(res_path, x_crop_t)
            res_path = os.path.join(
                self.meta_path, str(self.v_id) + "_pert.jpg"
            )
            self.save_tensorasimg(res_path, pert)
            res_path = os.path.join(
                self.meta_path, str(self.v_id) + "_pert_sum.jpg"
            )
            self.save_tensorasimg(res_path, pert_sum)
            for i in range(disp_score.shape[0]):
                res_path = os.path.join(
                    self.meta_path, str(self.v_id) + "_score" + str(i) + ".jpg"
                )
                res_path_org = os.path.join(
                    self.meta_path,
                    str(self.v_id) + "_score_org" + str(i) + ".jpg",
                )
                self.save_tensorasimg(res_path, disp_score[i, :, :, :])
                self.save_tensorasimg(res_path_org, disp_score_org[i, :, :, :])

            res_path = os.path.join(
                self.meta_path, str(self.v_id) + "_loss.txt"
            )
            with open(res_path, "w") as f:
                for x in losses:
                    f.write(str(x.cpu().detach().numpy()) + "\n")

        self.opt_flow_prev_xcrop = x_crop
        if CFG.cuda:
            prev_perts = torch.cat((prev_perts, pert), 0).cuda()
            weights = torch.cat(
                (weights, torch.ones(x_crop.size()).cuda()), 0
            ).cuda()
        else:
            prev_perts = torch.cat((prev_perts, pert), 0)
            weights = torch.cat((weights, torch.ones(x_crop.size())), 0)

        if self.reg_type is RegType.Weighted:
            pert_sum = torch.mul(weights, prev_perts).sum(0)
        else:
            pert_sum = prev_perts.sum(0)

        adv_x_crop = x_crop + pert_sum
        adv_x_crop = torch.clamp(adv_x_crop, 0, 255)
        pert_true = adv_x_crop - x_crop
        if CFG.attacker.gen_adv_img:
            adv_img = tracker.get_orgimg(
                img,
                x_crop,
                tracker.center_pos,
                CFG.track.instance_size,
                round(s_x),
                tracker.channel_average,
            )
        else:
            adv_img = None

        if prev_perts.shape[0] > self.accframes:
            prev_perts = prev_perts[-self.accframes :, :, :, :]
            weights = weights[-self.accframes :, :, :, :]

        return adv_x_crop, pert_true, prev_perts, weights, adv_img

    def ua_label(
        self, tracker: BaseTracker, scale_z: np.float64, outputs: _MB_Track_R
    ):
        score = tracker._convert_score(outputs["cls"])
        pred_bbox = tracker._convert_bbox(outputs["loc"], tracker.anchors)

        def change(r):
            return np.maximum(r, 1.0 / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(
            sz(pred_bbox[2, :], pred_bbox[3, :])
            / (sz(tracker.size[0] * scale_z, tracker.size[1] * scale_z))
        )

        # aspect ratio penalty
        r_c = change(
            (tracker.size[0] / tracker.size[1])
            / (pred_bbox[2, :] / pred_bbox[3, :])
        )
        penalty = np.exp(-(r_c * s_c - 1) * CFG.track.penalty_k)
        pscore = penalty * score

        # window penalty
        pscore = (
            pscore * (1 - CFG.track.window_influence)
            + tracker.window * CFG.track.window_influence
        )
        best_idx = np.argmax(pscore)

        obj_bbox = pred_bbox[:, best_idx]
        obj_pos = np.array([obj_bbox[0], obj_bbox[1]])

        b, a2, h, w = outputs["cls"].size()
        size = tracker.size
        template_size = np.array(
            [
                size[0] + CFG.track.context_amount * (size[0] + size[1]),
                size[1] + CFG.track.context_amount * (size[0] + size[1]),
            ]
        )
        context_size = template_size * (
            CFG.track.instance_size / CFG.track.exemplar_size
        )

        same_prev = False
        # validate delta meets the requirement of obj
        if self.prev_delta is not None:
            diff_pos = np.abs(self.prev_delta - obj_pos)
            if (
                size[0] // 2 < diff_pos[0]
                and diff_pos[0] < context_size[0] // 2
            ) and (
                size[1] // 2 < diff_pos[1]
                and diff_pos[1] < context_size[1] // 2
            ):
                delta = self.prev_delta
                same_prev = True
            else:
                delta = []
                delta.append(
                    random.choice((1, -1))
                    * random.randint(size[0] // 2, context_size[0] // 2)
                )
                delta.append(
                    random.choice((1, -1))
                    * random.randint(size[1] // 2, context_size[1] // 2)
                )
                delta = obj_pos + np.array(delta)
                self.prev_delta = delta
        else:
            delta = []
            delta.append(
                random.choice((1, -1))
                * random.randint(
                    size[0] // 2, context_size[0] // 2 - size[0] // 2
                )
            )
            delta.append(
                random.choice((1, -1))
                * random.randint(
                    size[1] // 2, context_size[1] // 2 - size[1] // 2
                )
            )
            delta = obj_pos + np.array(delta)
            self.prev_delta = delta

        desired_pos = context_size / 2 + delta
        upbound_pos = context_size
        downbound_pos = np.array([0, 0])
        desired_pos[0] = np.clip(
            desired_pos[0], downbound_pos[0], upbound_pos[0]
        )
        desired_pos[1] = np.clip(
            desired_pos[1], downbound_pos[1], upbound_pos[1]
        )
        if CFG.attacker.eval:
            desired_pos_abs = tracker.center_pos + delta
            tpos = []
            tpos.append(desired_pos_abs[0])
            tpos.append(desired_pos_abs[1])
            self.target_traj.append(tpos)

        desired_bbox = self._get_bbox(desired_pos, size)

        anchor_target = AnchorTarget(
            im_c_overwrite=CFG.track.instance_size,
            size_overwrite=w,
        )

        results = anchor_target(desired_bbox, w)
        overlap = results[3]
        max_val = np.max(overlap)
        max_pos = np.where(overlap == max_val)
        adv_cls = torch.zeros(results[0].shape)
        adv_cls[:, max_pos[1], max_pos[2]] = 1
        adv_cls = adv_cls.view(
            1, adv_cls.shape[0], adv_cls.shape[1], adv_cls.shape[2]
        )

        self.target_pos = desired_pos

        return adv_cls, same_prev

    def ta_label(
        self, tracker: BaseTracker, scale_z: np.float64, outputs: _MB_Track_R
    ):
        b, a2, h, w = outputs["cls"].size()
        center_pos = tracker.center_pos
        size = tracker.size
        try:
            desired_pos = np.array(self.target_traj[self.v_id])
        except IndexError:
            print("len of self.target_traj:", len(self.target_traj))
            print("self.v_id:", self.v_id)
        same_prev = False
        if self.v_id > 1:
            prev_desired_pos = np.array(self.target_traj[self.v_id - 1])
            if np.linalg.norm(prev_desired_pos - desired_pos) < 5:
                same_prev = True
        template_size = np.array(
            [
                size[0] + CFG.track.context_amount * (size[0] + size[1]),
                size[1] + CFG.track.context_amount * (size[0] + size[1]),
            ]
        )
        context_size = template_size * (
            CFG.track.instance_size / CFG.track.exemplar_size
        )
        delta = desired_pos - center_pos
        desired_pos = delta + context_size / 2
        upbound_pos = context_size
        downbound_pos = np.array([0, 0])
        desired_pos[0] = np.clip(
            desired_pos[0], downbound_pos[0], upbound_pos[0]
        )
        desired_pos[1] = np.clip(
            desired_pos[1], downbound_pos[1], upbound_pos[1]
        )
        desired_bbox = self._get_bbox(desired_pos, size)

        anchor_target = AnchorTarget()

        results = anchor_target(desired_bbox, w)
        overlap = results[3]
        max_val = np.max(overlap)
        max_pos = np.where(overlap == max_val)
        adv_cls = torch.zeros(results[0].shape)
        adv_cls[:, max_pos[1], max_pos[2]] = 1
        adv_cls = adv_cls.view(
            1, adv_cls.shape[0], adv_cls.shape[1], adv_cls.shape[2]
        )

        self.target_pos = desired_pos

        return adv_cls, same_prev

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def oim_once(self, tracker: BaseTracker, data: AtkData):
        if CFG.cuda:
            zf = data["template_zf"].cuda()
            search = data["search"].cuda()
            pert = data["pert"].cuda()
            prev_perts = data["prev_perts"].cuda()
            adv_cls = data["adv_cls"].cuda()
            weights = data["weights"].cuda()
            momen = data["momen"].cuda()
        else:
            zf = data["template_zf"]
            search = data["search"]
            pert = data["pert"]
            prev_perts = data["prev_perts"]
            adv_cls = data["adv_cls"]
            weights = data["weights"]
            momen = data["momen"]

        zf_list = []
        if zf.shape[0] > 1:
            for i in range(0, zf.shape[0]):
                zf_list.append(
                    zf[i, :, :, :].resize_(
                        1, zf.shape[1], zf.shape[2], zf.shape[3]
                    )
                )
        else:
            zf_list = zf

        # get feature
        if self.reg_type is RegType.Weighted:
            pert_sum = torch.mul(weights, prev_perts).sum(0)
        else:
            pert_sum = prev_perts.sum(0)

        xf = tracker.model.backbone(
            search
            + pert
            + pert_sum.view(
                1,
                prev_perts.shape[1],
                prev_perts.shape[2],
                prev_perts.shape[3],
            )
        )
        if CFG.adjust.adjust:
            xf = tracker.model.neck(xf)
        cls, loc = tracker.model.rpn_head(zf_list, xf)

        # get loss1
        cls = tracker.model.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, adv_cls)

        # regularization loss
        if CFG.cuda:
            c_prev_perts = torch.cat((prev_perts, pert), 0).cuda()
            # c_weights = torch.cat((weights, torch.ones(pert.size()).cuda()), 0).cuda()
        else:
            c_prev_perts = torch.cat((prev_perts, pert), 0)
            # c_weights = torch.cat((weights, torch.ones(pert.size())), 0)

        t_prev_perts = c_prev_perts.view(
            c_prev_perts.shape[0] * c_prev_perts.shape[1],
            c_prev_perts.shape[2] * c_prev_perts.shape[3],
        )
        # t_weights = c_weights.view(c_weights.shape[0],
        #  c_weights.shape[1] * c_weights.shape[2] * c_weights.shape[3])

        if self.reg_type is RegType.L21:
            reg_loss = torch.norm(
                t_prev_perts, 2, 1
            ).sum()  # +torch.norm(pert,2)
        elif self.reg_type is RegType.L2:
            reg_loss = torch.norm(t_prev_perts, 2)
        elif self.reg_type is RegType.L_inf:
            reg_loss = torch.max(torch.abs(t_prev_perts))
            # print("reg_loss{}".format(reg_loss))
        elif self.reg_type is RegType.Weighted:
            reg_loss = torch.norm(t_prev_perts, 2, 1).sum()
        else:
            reg_loss = 0.0

        total_loss: Tensor = cls_loss + self.lamb * reg_loss  # type: ignore
        total_loss.backward()

        pert_grad = data["pert"].grad
        if isinstance(pert_grad, Tensor):
            x_grad = -pert_grad
        else:
            raise Exception('non grad for data["pert"]')  # TODO:

        if self.reg_type is RegType.Weighted:
            pert_sum = torch.mul(weights, prev_perts).sum(0)

        adv_x = search

        if self.norm_type is NormType.L_inf:
            x_grad = torch.sign(x_grad)
            adv_x = adv_x + pert + pert_sum + self.eplison * x_grad
            pert = adv_x - search - pert_sum
            pert = torch.clamp(pert, -self.inta, self.inta)
        elif self.norm_type is NormType.L_1:
            adv_x = adv_x + pert + pert_sum + self.eplison * x_grad
            pert = adv_x - search - pert_sum
            norm = torch.sum(torch.abs(pert))
            pert = torch.min(pert * self.inta / norm, pert)
        elif self.norm_type is NormType.L_2:
            x_grad = x_grad / torch.norm(x_grad)
            adv_x = adv_x + pert + pert_sum + self.eplison * x_grad
            pert = adv_x - search - pert_sum
            pert = torch.clamp(pert / torch.norm(pert), -self.inta, self.inta)
        elif self.norm_type is NormType.Momen:
            momen = self.lamb_momen * momen + x_grad / torch.norm(x_grad, 1)
            adv_x = adv_x + pert + pert_sum + self.eplison * torch.sign(momen)
            pert = adv_x - search - pert_sum
            pert = torch.clamp(pert, -self.inta, self.inta)

        p_search = search + pert + pert_sum
        p_search = torch.clamp(p_search, 0, 255)
        pert = p_search - search - prev_perts.sum(0)

        return pert, total_loss, cls, momen, weights

    def target_traj_gen(self, init_rect, vid_h, vid_w, vid_l):
        target_traj = []
        pos = []
        pos.append(init_rect[0] + init_rect[2] / 2)
        pos.append(init_rect[1] + init_rect[3] / 2)
        target_traj.append(pos)
        for i in range(0, vid_l - 1):
            tpos = []
            if i % 50 == 0:
                deltay = random.randint(-10, 10)
                deltax = random.randint(-10, 10)
            elif i % 10 == 0:
                deltay = random.randint(0, 1)
                deltax = random.randint(0, 1)
            elif i % 5 == 0:
                deltay = random.randint(-1, 0)
                deltax = random.randint(-1, 0)
            tpos.append(np.clip(target_traj[i][0] + deltax, 0, vid_w - 1))
            tpos.append(np.clip(target_traj[i][1] + deltay, 0, vid_h - 1))
            target_traj.append(tpos)
        self.target_traj = target_traj
        return target_traj

    def target_traj_gen_supervised(
        self, init_rect, vid_h, vid_w, vid_l, gt_traj
    ):
        target_traj = []
        w, h = gt_traj[0][2], gt_traj[0][3]
        w_z = w + CFG.track.context_amount * np.sum(np.array([w, h]))
        h_z = h + CFG.track.context_amount * np.sum(np.array([w, h]))
        s_z = np.sqrt(w_z * h_z)
        scale_z = CFG.track.exemplar_size / s_z
        s_x = s_z * (CFG.track.instance_size / CFG.track.exemplar_size)
        deltax, deltay = -s_x / 5, -s_x / 5
        for i in range(vid_l):
            pos = []
            cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_traj[i]))
            pos.append(cx + deltax)
            pos.append(cy + deltay)
            target_traj.append(pos)
        self.target_traj = target_traj
        return target_traj

    def target_traj_gen_custom(self, init_rect, vid_h, vid_w, vid_l, type=1):
        """
        type 1 : ellipse
        type 2 : rectangle
        type 3 : triangle
        """
        target_traj = []
        initpos = np.array(
            [init_rect[0] - init_rect[2] / 2, init_rect[1] - init_rect[3] / 2]
        )
        target_traj.append(initpos)

        # initial start_point , shape of traj
        def ellipse(t, a):
            return a * t * cos(t), a * t * sin(t)

        def rectangle(t, a):
            if t < 0.5:
                return t * a, 0
            elif t >= 0.5 and t < 1:
                return 0.5 * a, -(t - 0.5) * a
            elif t >= 1 and t < 2:
                return -(t - 1) * a + 0.5 * a, -0.5 * a
            elif t >= 2 and t < 3:
                return -0.5 * a, (t - 2) * a - 0.5 * a
            else:
                return (t - 3) * a - 0.5 * a, 0.5 * a

            return 0, 0

        def triangle(t, r):
            if t < 1:
                return (
                    0.5 * r - r * t * cos(pi / 3),
                    -r * t * sin(pi / 3),
                )
            elif t < 2:
                return -(t - 1) * r * cos(pi / 3), (t - 2) * r * sin(pi / 3)
            else:
                return (t - 2.5) * r, 0

        def line(t, r, theta):
            """line genetate line traj

            Arguments:
                t {float} -- time
                r {float} -- length
                theta {float} -- angle

            Returns:
                [float,float] -- position
            """
            return t * r * cos(theta), t * r * sin(theta)

        r = 2 * min(vid_w, vid_h) / 2

        for i in range(0, vid_l - 1):
            tpos = []
            if type == 1:
                t = 6 * pi * i / vid_l
                x, y = ellipse(t, r / (pi * 8))
            if type == 2:
                t = 4.0 * i / vid_l
                x, y = rectangle(t, r / 2)

            if type == 3:
                t = 3.0 * i / vid_l
                x, y = triangle(t, r / 2)

            if type == 4:
                t = 1.0 * i / vid_l
                x, y = line(t, r, -pi * 0.4)

            tpos.append(np.clip(x + initpos[0], 0, vid_w - 1))
            tpos.append(np.clip(y + initpos[1], 0, vid_h - 1))
            target_traj.append(tpos)
        self.target_traj = target_traj
        return target_traj

    def _get_bbox(self, center_pos, shape):
        if len(shape) == 4:
            w, h = shape[2] - shape[0], shape[3] - shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = CFG.train.exemplar_size
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w * scale_z
        h = h * scale_z
        cx, cy = center_pos * scale_z
        bbox = center2corner(Center(cx - w / 2, cy - h / 2, w, h))
        return bbox

    def norm_score(self, score):
        disp_score = score[:, :, :, :, 1].permute(1, 0, 2, 3)
        disp_score_min = torch.min(torch.min(disp_score, 2)[0], 2)[0]
        disp_score_max = torch.max(torch.max(disp_score, 2)[0], 2)[0]
        dispmin = torch.zeros(disp_score.size())
        dispmax = torch.zeros(disp_score.size())
        for i in range(4):
            dispmin[i, :, :, :] = disp_score_min[i].repeat(
                1, 1, disp_score.shape[2], disp_score.shape[3]
            )
            dispmax[i, :, :, :] = disp_score_max[i].repeat(
                1, 1, disp_score.shape[2], disp_score.shape[3]
            )
        disp_score = (disp_score - dispmin) / (dispmax - dispmin) * 255

        return disp_score
