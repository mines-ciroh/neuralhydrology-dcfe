import torch

from neuralhydrology.modelzoo.cfe_modules.cfe_dataclasses import BasinCharacteristics, CFEParams, SoilParams
from neuralhydrology.utils.config import Config


def get_default_params(cfg: Config, additional_features: torch.Tensor, device: str) -> CFEParams:
    """
    Initialize from default parameters. Right now these are coming from the Araki study.
    """
    soil_params = SoilParams(
        depth=additional_features["depth"],
        bb=additional_features["bb"],
        satdk=additional_features["satdk"],
        satpsi=additional_features["satpsi"],
        slop=additional_features["slop"],
        smcmax=additional_features["smcmax"],
        wltsmc=additional_features["wltsmc"],
        D=additional_features["D"],
        mult=additional_features["mult"],
    )
    basin_characteristics = BasinCharacteristics(
        catchment_area_km2=additional_features["catchment_area_km2"],
        refkdt=additional_features["refkdt"],
        max_gw_storage=additional_features["max_gw_storage"],
        expon=additional_features["expon"],
        Cgw=additional_features["Cgw"],
        alpha_fc=additional_features["alpha_fc"],
        K_nash=additional_features["K_nash"],
        K_lf=additional_features["K_lf"],
        nash_storage=additional_features["nash_storage"],
        giuh_ordinates=additional_features["giuh_ordinates"],
    )
    cfe_params = CFEParams(
        soil_params=soil_params,
        basin_characteristics=basin_characteristics,
        hourly=cfg.dcfe_hourly,
        soil_scheme=cfg.dcfe_soil_scheme,
        partition_scheme=cfg.dcfe_partition_scheme,
    )
    return cfe_params
