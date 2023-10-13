import sys
from dataclasses import dataclass

import numpy as np

# ====================
import pix2pix

sys.path.append(pix2pix.__path__[0])  # type: ignore
from pix2pix.models.base_model import BaseModel

# ====================

from msot.tools.test.utils.process import (
    Allow,
    Input,
    ProcessTemplate,
    Processor,
    ProceesorAttrs,
    ProceesorConfig,
    ValidInput,
    ValidOutput,
)
from msot.trackers.types import ScaledCrop

from robustot.libs.CSA.utils import (
    adv_attack_search,
    adv_attack_template,
    adv_attack_template_S,
)
from robustot.libs.CSA.models import get_test_model
from robustot.libs.CSA.options import get_test_options
from robustot.libs.CSA.types import AttackOn, AttackType


@dataclass
class CSAConfig(ProceesorConfig):
    attack_on: AttackOn
    attack_type: AttackType
    checkpoint_dir: str
    allow_access = Allow.TRACKER | Allow.TARGET
    valid_input = ValidInput.SCALED_CROP
    valid_output = ValidOutput.SCALED_CROP | ValidOutput.NONE


@dataclass
class CSAAttrs(ProceesorAttrs):
    ...


class CSAProcessor(Processor[CSAAttrs, CSAConfig]):
    gan: BaseModel

    def __init__(
        self,
        name: str,
        config: CSAConfig,
    ) -> None:
        super().__init__(name, config, CSAAttrs)

        options = get_test_options(
            self.config.attack_on,
            self.config.attack_type,
        )
        self.gan = get_test_model(options, self.config.checkpoint_dir)

    def process(self, input: ScaledCrop) -> ScaledCrop | None:
        target_size = (
            self.process_target.crop_size,
            self.process_target.crop_size,
        )

        output: ScaledCrop | None = None

        if type(self.process_target) is ProcessTemplate:
            if AttackOn.TEMPLATE in self.config.attack_on:
                if AttackOn.SEARCH in self.config.attack_on:
                    z_crop = adv_attack_template_S(
                        input.crop, self.gan, target_size
                    )
                else:
                    z_crop = adv_attack_template(input.crop, self.gan)
                output = ScaledCrop(z_crop, input.size)
            else:
                ...
        else:
            if AttackOn.SEARCH in self.config.attack_on:
                x_crop = adv_attack_search(input.crop, self.gan, target_size)
                output = ScaledCrop(x_crop, input.size)
            else:
                ...

        return output
