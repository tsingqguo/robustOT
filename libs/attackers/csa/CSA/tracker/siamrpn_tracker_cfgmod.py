import cv2
import numpy as np
import os
from attackers.csa.CSA.attack_utils import (
    add_pulse_noise,
    adv_attack_search,
    adv_attack_template,
    adv_attack_template_S,
)
from attackers.csa.CSA.base_model import (
    Base_L2_500_Search,
    Base_L2_500_Template,
)
from attackers.csa.CSA.data_utils import tensor2img
from pysot.tracker.siamrpn_tracker_cfgmod import SiamRPNTracker, SeqImg
from typing import Optional


class CSA_SiamRPNTracker:
    raw: SiamRPNTracker

    def __init__(self, tracker: SiamRPNTracker) -> None:
        self.raw = tracker

    def save_img(
        self, tensor_clean, tensor_adv, save_path: str, frame_id: int
    ):
        os.makedirs(save_path, exist_ok=True)
        ## clean x_crop
        img_clean = tensor2img(tensor_clean)
        cv2.imwrite(
            os.path.join(save_path, "%04d_clean.jpg" % frame_id), img_clean
        )
        ## adv x_crop
        img_adv = tensor2img(tensor_adv)
        cv2.imwrite(
            os.path.join(save_path, "%04d_adv.jpg" % frame_id), img_adv
        )
        ## diff
        tensor_diff = (tensor_adv - tensor_clean) * 10
        # print(torch.mean(torch.abs(tensor_diff)))
        tensor_diff += 127.0
        img_diff = tensor2img(tensor_diff)
        cv2.imwrite(
            os.path.join(save_path, "%04d_diff.jpg" % frame_id), img_diff
        )

    def init_adv(
        self,
        img: SeqImg,
        bbox,
        GAN: Base_L2_500_Template,
        dummy: Optional[SiamRPNTracker],
        save_path: Optional[str] = None,
        name: Optional[str] = None,
    ):
        z_crop = self.raw.get_z_crop(img, bbox, dummy)
        """Adversarial Attack"""
        z_crop_adv = adv_attack_template(z_crop, GAN)
        self.raw.feed_model_template(z_crop_adv)
        """save"""
        if save_path is not None and name is not None:
            os.makedirs(save_path, exist_ok=True)
            z_crop_img = tensor2img(z_crop)
            cv2.imwrite(
                os.path.join(save_path, name + "_clean.jpg"), z_crop_img
            )
            z_crop_adv_img = tensor2img(z_crop_adv)
            cv2.imwrite(
                os.path.join(save_path, name + "_adv.jpg"), z_crop_adv_img
            )
            diff = z_crop_adv - z_crop
            diff_img = tensor2img(diff)
            cv2.imwrite(os.path.join(save_path, name + "_diff.jpg"), diff_img)

    def init_adv_S(
        self,
        img: SeqImg,
        bbox,
        GAN: Base_L2_500_Search,
        dummy: Optional[SiamRPNTracker],
        save_path: Optional[str] = None,
        name: Optional[str] = None,
    ):
        z_crop = self.raw.get_z_crop(img, bbox, dummy)
        """Adversarial Attack"""
        z_crop_adv = adv_attack_template_S(
            z_crop,
            GAN,
            target_sz=(
                z_crop.shape[-2],
                z_crop.shape[-1],
            ),
        )
        self.raw.feed_model_template(z_crop_adv)
        """save"""
        if save_path is not None and name is not None:
            z_crop_img = tensor2img(z_crop)
            cv2.imwrite(
                os.path.join(save_path, name + "_clean.jpg"), z_crop_img
            )
            z_crop_adv_img = tensor2img(z_crop_adv)
            cv2.imwrite(
                os.path.join(save_path, name + "_adv.jpg"), z_crop_adv_img
            )
            diff = z_crop_adv - z_crop
            diff_img = tensor2img(diff)
            cv2.imwrite(os.path.join(save_path, name + "_diff.jpg"), diff_img)

    def track_adv(
        self,
        img: SeqImg,
        GAN: Base_L2_500_Search,
        dummy: Optional[SiamRPNTracker],
        save_path: Optional[str] = None,
        frame_id: Optional[int] = None,
    ):
        x_crop, scale_z = self.raw.get_x_crop(img, dummy)
        import time

        t0 = time.time()
        """Adversarial Attack"""
        x_crop_adv = adv_attack_search(
            x_crop,
            GAN,
            search_sz=(
                x_crop.shape[-2],
                x_crop.shape[-1],
            ),
        )
        ta = time.time() - t0
        """predict"""
        output_dict = self.raw.get_res_from_x_crop(
            img, x_crop_adv, scale_z, dummy
        )
        if save_path is not None and frame_id is not None:
            """save"""
            self.save_img(x_crop, x_crop_adv, save_path, frame_id)
        return output_dict, ta

    def track_impulse(
        self,
        img: SeqImg,
        dummy: Optional[SiamRPNTracker],
        prob,
        save_path: Optional[str],
        frame_id: Optional[int],
    ):
        x_crop, scale_z = self.raw.get_x_crop(img, dummy)
        """impulse Attack"""
        x_crop_adv = add_pulse_noise(x_crop, prob)
        """predict"""
        output_dict = self.raw.get_res_from_x_crop(
            img, x_crop_adv, scale_z, dummy
        )
        if save_path is not None and frame_id is not None:
            """save"""
            self.save_img(x_crop, x_crop_adv, save_path, frame_id)
        return output_dict

    """supplementary material"""

    def track_supp(
        self,
        img,
        GAN: Base_L2_500_Search,
        dummy: Optional[SiamRPNTracker],
        save_path: str,
        frame_id: int,
    ):
        x_crop, scale_z = self.raw.get_x_crop(img, dummy)
        """save clean region and heatmap"""
        x_crop_img = tensor2img(x_crop)
        cv2.imwrite(
            os.path.join(save_path, "ori_search_%d.jpg" % frame_id), x_crop_img
        )
        """original heatmap"""
        outputs_clean = self.raw.model.track(x_crop)
        score = self.raw._convert_score(outputs_clean["cls"])  # (25x25x5,)
        heatmap_clean = 255.0 * np.max(
            score.reshape(5, 25, 25), axis=0
        )  # [0,1]
        heatmap_clean = cv2.resize(
            heatmap_clean, (255, 255), interpolation=cv2.INTER_CUBIC
        )
        heatmap_clean = cv2.applyColorMap(
            heatmap_clean.clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        cv2.imwrite(
            os.path.join(save_path, "heatmap_clean_%d.jpg" % frame_id),
            heatmap_clean,
        )
        """Adversarial Attack"""
        x_crop_adv = adv_attack_search(
            x_crop,
            GAN,
            search_sz=(
                x_crop.shape[-2],
                x_crop.shape[-1],
            ),
        )
        output_dict = self.raw.get_res_from_x_crop(
            img, x_crop_adv, scale_z, dummy
        )
        """save adv region and heatmap"""
        x_crop_img_adv = tensor2img(x_crop_adv)
        cv2.imwrite(
            os.path.join(save_path, "adv_search_%d.jpg" % frame_id),
            x_crop_img_adv,
        )
        score_adv = self.raw.score
        heatmap_adv = 255.0 * np.max(
            score_adv.reshape(5, 25, 25), axis=0
        )  # [0,1]
        heatmap_adv = cv2.resize(
            heatmap_adv, (255, 255), interpolation=cv2.INTER_CUBIC
        )
        heatmap_adv = cv2.applyColorMap(
            heatmap_adv.clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        cv2.imwrite(
            os.path.join(save_path, "heatmap_adv_%d.jpg" % frame_id),
            heatmap_adv,
        )
        return output_dict
