# Copyright (c) SenseTime. All Rights Reserved.
import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import torchvision.transforms
from pysot.core.config import cfg
from pysot.models import ModelBuilder
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker
from pytorch_grad_cam import AblationCAM, GradCAM, ScoreCAM
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch import Tensor
from typing import Callable, Optional, TypeVar


T = TypeVar('T', bound=Tensor)
PostProcessFn = Callable[[T],T]

class SiamRPNTracker(SiameseTracker):
    # cam: Optional[BaseCAM]
    x_crop_post_processing: Optional[PostProcessFn]
    z_crop_post_processing: Optional[PostProcessFn]

    def __init__(self, model: ModelBuilder):
        super(SiamRPNTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()
        #
        # cam_targets = [self.model.backbone.layer4]
        # self.cam = AblationCAM(
        #     self.model.backbone,
        #     target_layers=cam_targets,
        #     use_cuda=True,
        # )
        #
        self.x_crop_post_processing = None
        self.z_crop_post_processing = None

    def generate_anchor(self, score_size: int) -> npt.NDArray[np.float32]:
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor:np.ndarray = anchors.anchors # type: ignore
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta: Tensor, anchor: npt.NDArray[np.float32]) -> Tensor:
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score: Tensor) -> Tensor:
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def get_z_crop(self, img: np.ndarray, bbox: list):
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        return z_crop

    def get_x_crop(self, img: np.ndarray):
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        return x_crop, scale_z

    def get_res_from_x_crop(self, img: np.ndarray, x_crop: Tensor, scale_z):
        if self.x_crop_post_processing is not None:
            x_crop = self.x_crop_post_processing(x_crop)
        
        # print("------------------------")
        # print(f"img.shape = {img.shape}")        
        # print(f"x_crop.shape = {x_crop.shape}")    
        # cv2.imwrite("grfxc_origin.jpg", img)
        # x = x_crop[0].permute(1,2,0).cpu().detach().numpy()
        # cv2.imwrite("grfxc_x_crop.jpg", x)
        # print(x_crop.device)
        # torch.ones_like(x_crop).to(torch.device(x_crop.device)) + x_crop
        # exit(0)
        # img.shape: (1024, 1024, 3)
        # x_crop.shape: torch.Size([1, 3, 255, 255])    

        outputs = self.model.track(x_crop)

        score = self._convert_score(outputs["cls"])
        pred_bbox = self._convert_bbox(outputs["loc"], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)
        
        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))
        
        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        self.score = score
        
        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
            "bbox": bbox,
            "best_score": best_score,
        }

    def feed_model_template(self, z_crop: Tensor):
        if self.z_crop_post_processing is not None:
            z_crop = self.z_crop_post_processing(z_crop)
        self.model.template(z_crop)

    def init(self, img: np.ndarray, bbox):
        z_crop = self.get_z_crop(img, bbox)
        self.feed_model_template(z_crop)

    def track(self, img: np.ndarray):
        x_crop, scale_z = self.get_x_crop(img)
        res = self.get_res_from_x_crop(img, x_crop, scale_z)
        return res

