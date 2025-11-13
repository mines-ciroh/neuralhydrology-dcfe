import torch

from neuralhydrology.modelzoo.cfe_modules.cfe_dataclasses import CFEParams, Flux


def get_and_calculate_input_rainfall_and_ET(
    conceptual_forcing_timestep, flux: Flux, cfe_params: CFEParams, constants: any
) -> Flux:
    """
    calculate PET from shortwave radiation and mean temperature using jensen_evaporation_2016 "https://github.com/pyet-org/pyet/blob/master/pyet/radiation.py"
    Inputs:
        conceptual_forcing_timestep: torch.Tensor shape (batch_size, n_features) where 
            n_features = 3 for hourly data: [rainfall_mm_per_timestep, temp_C, shortwave_radiation_W_per_m2]
            n_features = 4 for daily data: [rainfall_mm_per_timestep, min_temp_C, max_temp_C, shortwave_radiation_W_per_m2]
        flux (Flux): Flux dataclass containing model fluxes.
        cfe_params (CFEParams): CFEParams dataclass containing basin characteristics including hourly
        constants (any): constants dictionary containing time step size information.
    
    Returns:
        flux:
            - timestep_rainfall_input_m (torch.Tensor): Updated rainfall input for the timestep [m/timestep].
            - potential_et_m_per_timestep (torch.Tensor): Updated potential evapotranspiration for the timestep [m/timestep].
            - reduced_potential_et_m_per_timestep (torch.Tensor): Updated reduced potential evapotranspiration for the timestep [m/timestep].
    """
    expected_feats = 3 if cfe_params.hourly else 4
    if conceptual_forcing_timestep.shape[1] != expected_feats:
        raise ValueError(
            f"Expected {expected_feats} features for {'hourly' if cfe_params.hourly else 'daily'} data, "
            f"but got {conceptual_forcing_timestep.shape[1]}."
        )

    # convert and store rainfall mm/timestep to m/timestep
    flux.timestep_rainfall_input_m = conceptual_forcing_timestep[:, 0] / 1000.0

    if cfe_params.hourly:
        mean_temp = conceptual_forcing_timestep[:, 1]
        shortRad = (
            conceptual_forcing_timestep[:, 2] * constants['time']['step_size'] / 1000000
        )  # convert shortwave radiation [W/m^2] to [MJ/m^2 hour]
    else:
        mean_temp = (conceptual_forcing_timestep[:, 1] + conceptual_forcing_timestep[:, 2]) / 2.0
        shortRad = (
            conceptual_forcing_timestep[:, 3] * constants['time']['step_size'] / 1000000
        )  # convert shortwave radiation [W/m^2] to [MJ/m^2 day]

    lambd = 2.501 - 0.002361 * mean_temp  # using mean temp
    pet_m_per_timestep_calc = (0.025 * shortRad * (mean_temp - (-3.0)) / lambd) / 1000  # convert pet [mm/hr] to [m/hr]
    pet_m_per_timestep_mask = pet_m_per_timestep_calc < 0  # make mask for negative PET
    flux.potential_et_m_per_timestep = torch.where(pet_m_per_timestep_mask, 0, pet_m_per_timestep_calc)  # clip negative PET to 0
    flux.reduced_potential_et_m_per_timestep = flux.potential_et_m_per_timestep.clone()

    return flux
