import os
from pyotp.utils.config import Config


class Env(Config):
    project_path: str
    experiments_path: str
    dset_root_testing: str
    dset_root_training: str


ENV = Env()
ENV.project_path = os.environ.get("PYOTP_PATH", "")
ENV.experiments_path = os.environ.get("PYOTP_EXP", "")
ENV.dset_root_testing = os.environ.get("TEST_DSET_PATH", "")
ENV.dset_root_training = os.environ.get("TRAIN_DSET_PATH", "")
