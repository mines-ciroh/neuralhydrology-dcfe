import torch

from neuralhydrology.modelzoo.cfe_modules.cfe_dataclasses import Snow_CFEParams, Flux


def get_and_calculate_rainfall_snow_snowmelt(
    conceptual_forcing_timestep, flux: Flux, cfe_params: Snow_CFEParams,
) -> Flux:
    """
    calculate snowfall, rainfall, and snowmelt for the current timestep based on the input forcing and the snow parameters in cfe_params.
    Inputs:
        conceptual_forcing_timestep: torch.Tensor shape (batch_size, n_features) where 
            n_features = 3 for hourly data: [rainfall_mm_per_timestep, temp_C, shortwave_radiation_W_per_m2]
            n_features = 4 for daily data: [rainfall_mm_per_timestep, min_temp_C, max_temp_C, shortwave_radiation_W_per_m2]
        flux (Flux): Flux dataclass containing model fluxes.
        cfe_params (Snow_CFEParams): Snow_CFEParams dataclass containing basin characteristics, soil parameters, and snow parameters.
    
    Returns:
        flux:
            - timestep_rainfall_input_m (torch.Tensor): Updated rainfall input for the timestep [m/timestep].
            - timestep_snowfall_input_m (torch.Tensor): Calculated snowfall input for the timestep [m/timestep].
            - timestep_snowmelt_m (torch.Tensor): Calculated snowmelt for the timestep [m/timestep].
    """
    if cfe_params.hourly:
        mean_temp = conceptual_forcing_timestep[:, 1]
        raise ValueError(
            f"Hourly data is not currently supported in the Snow-CFE module due to dd param."
        )
    else:
        mean_temp = (conceptual_forcing_timestep[:, 1] + conceptual_forcing_timestep[:, 2]) / 2.0
    
    temp_mask = mean_temp < 0
    snow_melt = mean_temp * cfe_params.snow_params.dd  # simple degree-day snowmelt model
    snow_melt[temp_mask] = torch.zeros_like(snow_melt[temp_mask]) # no melt if mean temp is below 0 C

    liquid_rainfall = conceptual_forcing_timestep[:, 0].clone()
    liquid_rainfall[temp_mask] = torch.zeros_like(liquid_rainfall[temp_mask]) # no liquid rainfall if mean temp is below 0 C
    snowfall = conceptual_forcing_timestep[:, 0].clone()
    snowfall[~temp_mask] = torch.zeros_like(snowfall[~temp_mask]) # no snowfall if mean temp is above or equal to 0 C
    
    flux.timestep_rainfall_input_m = liquid_rainfall / 1000.0  # convert from mm/timestep to m/timestep
    flux.timestep_snowfall_input_m = snowfall / 1000.0  # convert from mm/timestep to m/timestep
    flux.timestep_snowmelt_m = snow_melt / 1000.0  # convert from mm/timestep to m/timestep

    return flux
