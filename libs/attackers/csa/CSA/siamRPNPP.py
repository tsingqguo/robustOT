# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import os
# import torch.nn.functional as F
# from pysot.core.config import cfg
# from pysot.models import ModelBuilder as _ModelBuilder
# from pysot.models.model_builder import ModelBuilder
# from pysot.utils.model_load import load_pretrain
# from pyotp import ENV

'''Capsule SiamRPN++(We can use it as one component in higher-level task)'''
class SiamRPNPP():
    ...
    # model: _ModelBuilder

    # def __init__(self,dataset=''):
    #     exp_root = os.path.join(
    #         'CSA',
    #         'exps',
    #         'siamrpn_r50_l234_dwxcorr'
    #     )
    #     if 'OTB' in dataset:
    #         exp_root += '_otb'
    #     elif 'LT' in dataset:
    #         exp_root += '_lt'
    #     else:
    #         pass
    #     cfg_file = os.path.join(ENV.experiments_path, exp_root, 'config.yaml')
    #     snapshot = os.path.join(ENV.experiments_path, exp_root, 'model.pth')
    #     # load config
    #     cfg.merge_from_file(cfg_file)
    #     # create model
    #     # FIXME: dyn model_builder
    #     self.model = ModelBuilder()# A Neural Network.(a torch.nn.Module)
    #     # load model
    #     self.model = load_pretrain(self.model, snapshot).cuda().eval()

    # def get_heat_map(self, X_crop, softmax=False):
    #     score_map = self.model.track(X_crop)['cls']#(N,2x5,25,25)
    #     score_map = score_map.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)#(5HWN,2)
    #     if softmax:
    #         score_map = F.softmax(score_map, dim=1).data[:, 1]#(5HWN,)
    #     return score_map
    # def get_cls_reg(self, X_crop, softmax=False):
    #     outputs = self.model.track(X_crop)#(N,2x5,25,25)
    #     score_map = outputs['cls'].permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)#(5HWN,2)
    #     reg_res = outputs['loc'].permute(1, 2, 3, 0).contiguous().view(4, -1)
    #     if softmax:
    #         score_map = F.softmax(score_map, dim=1).data[:, 1]#(5HWN,)
    #     return score_map, reg_res

