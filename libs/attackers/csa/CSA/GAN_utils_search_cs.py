import os
from attackers.csa.CSA.base_model import Base_L2_500_Search
from attackers.csa.CSA.options.test_options import TestOptions
from pix2pix.models import create_model

from pyotp.env import ENV

opt = TestOptions("cs").parse([])

# modify some config
"""Attack Search Regions"""
# only cooling
# opt.model = 'G_search_L2_500'
# cooling + shrinking
opt.model = "G_search_L2_500_regress"

opt.netG = "unet_256"

# create and initialize model
"""create perturbation generator"""
# create a model given opt.model and other options
GAN: Base_L2_500_Search = create_model(opt)  # type: ignore

GAN.load_path_overwrite = os.path.join(
    ENV.experiments_path,
    "CSA",
    "checkpoints",
    opt.model,
    "1_net_G.pth",
)
GAN.setup(opt)  # regular setup: load and print networks; create schedulers
GAN.eval()
