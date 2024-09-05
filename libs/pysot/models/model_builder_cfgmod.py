import torch.nn.functional as F
from pyotp.config.pysot import CFG
from pysot.models import (
    MB_M_Input,
    MB_M_Output,
    ModelBuilder as _ModelBuilder,
    _MB_Track_R,
)
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck
from torch import Tensor


class ModelBuilder(_ModelBuilder):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(CFG.backbone.type, **CFG.backbone.kwargs)

        # build adjust layer
        if CFG.adjust.adjust:
            self.neck = get_neck(CFG.adjust.type, **CFG.adjust.kwargs)

        # build rpn head
        self.rpn_head = get_rpn_head(CFG.rpn.type, **CFG.rpn.kwargs)

        # build mask head
        if CFG.mask.mask:
            self.mask_head = get_mask_head(CFG.mask.type, **CFG.mask.kwargs)

            if CFG.refine.refine:
                self.refine_head = get_refine_head(CFG.refine.type)

    def template(self, z: Tensor) -> None:
        zf = self.backbone(z)
        if CFG.mask.mask:
            zf = zf[-1]
        if CFG.adjust.adjust:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x: Tensor) -> _MB_Track_R:
        xf = self.backbone(x)
        if CFG.mask.mask:
            self.xf = xf[:-1]
            xf = xf[-1]
        if CFG.adjust.adjust:
            xf = self.neck(xf)
        cls, loc = self.rpn_head(self.zf, xf)
        if CFG.mask.mask:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        else:
            mask = None
        return {
            "cls": cls,
            "loc": loc,
            "mask": mask,
        }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls: Tensor) -> Tensor:
        """
        log_softmax for RPN output's `cls`
        """
        B, anchors, H, W = cls.shape
        cls = cls.view(B, 2, anchors // 2, H, W)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data: MB_M_Input) -> MB_M_Output:
        """only used in training"""
        template = data["template"].cuda()
        search = data["search"].cuda()
        label_cls = data["label_cls"].cuda()
        label_loc = data["label_loc"].cuda()
        label_loc_weight = data["label_loc_weight"].cuda()

        # print('[debug] template:', template.shape)
        # print('[debug] search:', search.shape)

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        
        if CFG.mask.mask:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if CFG.adjust.adjust:
            zf = self.neck(zf)
            xf = self.neck(xf)
        cls, loc = self.rpn_head.forward(zf, xf)

        # print(f"上 [[ cls.shape: {cls.shape} ]]")

        # print('[debug]')
        # print('[debug] loc:', loc.shape)
        # print('[debug] label_loc:', label_loc.shape)
        # exit(1)

        # get loss
        cls_loss = select_cross_entropy_loss(self.log_softmax(cls), label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)
        # print(f"下 [[ cls.shape: {cls.shape} ]]")

        outputs: MB_M_Output = {
            "total_loss": (
                CFG.train.cls_weight * cls_loss
                + CFG.train.loc_weight * loc_loss
            ),
            "cls_loss": cls_loss,
            "loc_loss": loc_loss,
            "mask_loss": None,
            "_cls": cls,
            "_loc": loc,
        }

        if CFG.mask.mask:
            # TODO
            raise NotImplementedError("mask loss not implemented yet")
            mask, self.mask_corr_feature = self.mask_head(zf, xf)
            mask_loss = None
            outputs["total_loss"] += CFG.train.mask_weight * mask_loss
            outputs["mask_loss"] = mask_loss
        return outputs
