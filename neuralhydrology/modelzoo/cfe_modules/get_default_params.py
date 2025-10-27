import torch

from neuralhydrology.modelzoo.cfe_modules.cfe_dataclasses import CFEParams
from neuralhydrology.utils.config import Config


def get_default_params(cfg: Config, additional_features: torch.Tensor, device: str) -> CFEParams:
    """
    See lines 67--71 of dcfe.py. Note that additional things should be lifted from the config:
     - hourly
     - dcfe_soil_scheme
     - dcfe_partition_scheme
    """
    pass
