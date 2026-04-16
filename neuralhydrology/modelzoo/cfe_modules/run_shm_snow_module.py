import torch

from neuralhydrology.modelzoo.cfe_modules.cfe_dataclasses import Flux, SnowStates


def run_shm_snow_module(
    flux: Flux, 
    snow_reservoir: SnowStates,
) -> Flux:
    """
    update snowmelt and snow storage based on the degree-day method. 

    Parameters
    ----------
    flux : Flux
        The flux dataclass containing the current fluxes, including the predicted snowmelt flux for
        the current timestep (flux.timestep_snowmelt_m) and the snowfall input for the current timestep (flux.timestep_snowfall_input_m).
    snow_reservoir : SnowStates
        The snow reservoir dataclass containing the current snow storage (snow_reservoir.storage_m).
    
    Returns
    -------
    Flux
        Updated flux dataclass with the adjusted snowmelt flux after considering snow storage.
    SnowStates
        Updated snow reservoir dataclass with the new snow storage after accounting for snowfall and snowmelt
    """
    snowmelt_adjustment = torch.minimum(snow_reservoir.storage_m, flux.timestep_snowmelt_m) # snowmelt cannot be more than the snow storage
    snow_reservoir.storage_m = snow_reservoir.storage_m - snowmelt_adjustment + flux.timestep_snowfall_input_m # update snow storage with snowfall and snowmelt
    
    flux.timestep_snowmelt_m = snowmelt_adjustment # update snowmelt flux to be the adjusted snowmelt after considering snow storage
    
    # This module directly goes into partition scheme, so count snowmelt as part of "rainfall" to integrate to CFE.
    flux.timestep_rainfall_input_m = flux.timestep_rainfall_input_m + flux.timestep_snowmelt_m # add snowmelt to rainfall input for partitioning in the next step.
    return flux, snow_reservoir
