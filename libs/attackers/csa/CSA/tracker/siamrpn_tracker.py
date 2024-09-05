import cv2
import numpy as np
import os
from attackers.csa.CSA.attack_utils_nocfg import (
    add_pulse_noise,
    adv_attack_search,
    adv_attack_template,
    adv_attack_template_S,
)
from attackers.csa.CSA.data_utils import tensor2img
from pysot.tracker.siamrpn_tracker import SiamRPNTracker


class CSA_SiamRPNTracker:
    raw: SiamRPNTracker

    def __init__(self, tracker: SiamRPNTracker) -> None:
        self.raw = tracker

    def save_img(self, tensor_clean, tensor_adv, save_path, frame_id):
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

    def init_adv(self, img, bbox, GAN, save_path=None, name=None):
        z_crop = self.raw.get_z_crop(img, bbox)
        """Adversarial Attack"""
        z_crop_adv = adv_attack_template(z_crop, GAN)
        self.raw.feed_model_template(z_crop_adv)
        """save"""
        if save_path != None and name != None:
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

    def init_adv_S(self, img, bbox, GAN, save_path=None, name=None):
        z_crop = self.raw.get_z_crop(img, bbox)
        """Adversarial Attack"""
        z_crop_adv = adv_attack_template_S(z_crop, GAN)
        self.raw.feed_model_template(z_crop_adv)
        """save"""
        if save_path != None and name != None:
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

    def track_adv(self, img, GAN, save_path=None, frame_id=None):
        x_crop, scale = self.raw.get_x_crop(img)
        """Adversarial Attack"""
        x_crop_adv = adv_attack_search(x_crop, GAN)
        """predict"""
        output_dict = self.raw.get_res_from_x_crop(img, x_crop_adv, scale)
        if save_path != None and frame_id != None:
            """save"""
            self.save_img(x_crop, x_crop_adv, save_path, frame_id)
        return output_dict

    def track_impulse(self, img, prob, save_path, frame_id):
        x_crop, scale = self.raw.get_x_crop(img)
        """impulse Attack"""
        x_crop_adv = add_pulse_noise(x_crop, prob)
        """predict"""
        output_dict = self.raw.get_res_from_x_crop(img, x_crop_adv, scale)
        if save_path != None and frame_id != None:
            """save"""
            self.save_img(x_crop, x_crop_adv, save_path, frame_id)
        return output_dict

    """supplementary material"""

    def track_supp(self, img, GAN, save_path, frame_id):
        x_crop, scale = self.raw.get_x_crop(img)
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
        x_crop_adv = adv_attack_search(x_crop, GAN)
        output_dict = self.raw.get_res_from_x_crop(img, x_crop_adv, scale)
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
