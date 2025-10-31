import torch

from neuralhydrology.modelzoo.cfe_modules.cfe_dataclasses import Flux


def calculate_evaporation_from_rainfall(flux: Flux) -> Flux:
    # INITIALIZE
    rainfall_mask = flux.timestep_rainfall_input_m > 0.0
    # check if any basins in the batch receive rainfall this timestep.
    if torch.any(rainfall_mask):
        rainfall = flux.timestep_rainfall_input_m[rainfall_mask]
        pet = flux.potential_et_m_per_timestep[rainfall_mask]

        # UPDATE
        # If rainfall exceeds PET, actual ET = PET.
        # Otherwise, actual ET equals rainfall.
        actual_et_from_rain = torch.where(rainfall >= pet, pet, rainfall)

        reduced_rainfall = torch.where(rainfall >= pet, rainfall - actual_et_from_rain, torch.zeros_like(rainfall))

        # FINALIZE
        flux.actual_et_from_rain_m_per_timestep[rainfall_mask] = actual_et_from_rain
        flux.timestep_rainfall_input_m[rainfall_mask] = reduced_rainfall
        # adjust pet based on evaporation from rainfall.
        flux.reduced_potential_et_m_per_timestep[rainfall_mask] = pet - actual_et_from_rain
        # Track volume from rainfall.
        flux.actual_et_m_per_timestep += flux.actual_et_from_rain_m_per_timestep

    return flux
