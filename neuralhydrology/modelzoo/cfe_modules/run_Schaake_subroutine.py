from typing import Tuple

import torch

from neuralhydrology.modelzoo.cfe_modules.cfe_dataclasses import CFEParams, Flux, SoilStates


def run_Schaake_subroutine(
    flux: Flux,
    constants,
    cfe_params: CFEParams,
    soil_reservoir: SoilStates,
    soil_config,
) -> Tuple[Flux, SoilStates]:
    """
    Note: There's no ice process.
    """
    soil_reservoir.update(cfe_params, soil_config, constants)

    # Compute masks
    rainfall_mask = flux.timestep_rainfall_input_m > 0
    soil_noDeficit_mask = soil_reservoir.storage_deficit_m < 0  # mark ones w/o deficit
    soil_noDeficit_rain_mask = rainfall_mask & soil_noDeficit_mask
    soil_deficit_rain_mask = rainfall_mask & ~soil_noDeficit_mask

    if torch.any(rainfall_mask):
        flux.surface_runoff_depth_m[soil_noDeficit_rain_mask] = flux.timestep_rainfall_input_m[
            soil_noDeficit_rain_mask
        ]  # did not put in infiltration_depth_m as they are 0 in this case.

        Schaake_parenthetical_term = 1 - torch.exp(
            -soil_reservoir.Schaake_adjusted_magic_constant_by_soil_type[soil_deficit_rain_mask] * constants["time"]["days"]
        )

        Ic = soil_reservoir.storage_deficit_m[soil_deficit_rain_mask] * Schaake_parenthetical_term
        Px = flux.timestep_rainfall_input_m[soil_deficit_rain_mask]
        flux.infiltration_depth_m[soil_deficit_rain_mask] = Px * (Ic / (Px + Ic))

        soil_excess_mask = (
            flux.timestep_rainfall_input_m - flux.infiltration_depth_m > 0
        )  # mask for if rainfall is more than infilt depth
        combined_soil_excess_mask = soil_deficit_rain_mask & soil_excess_mask  # mask for above chunk + excess rain
        combined_soil_noExcess_mask = soil_deficit_rain_mask & ~soil_excess_mask  # mask for above chunk + no excess rain

        flux.surface_runoff_depth_m[combined_soil_excess_mask] = (flux.timestep_rainfall_input_m - flux.infiltration_depth_m)[
            combined_soil_excess_mask
        ]
        # not written else surface_runoff_depth_m = 0, since initialized at 0

        flux.infiltration_depth_m[combined_soil_noExcess_mask] = (flux.timestep_rainfall_input_m - flux.surface_runoff_depth_m)[
            combined_soil_noExcess_mask
        ]
        # not written, if no rainfall, surface_runoff_depth_m = 0 and infiltration_depth_m = 0 since initialized at 0

    return flux, soil_reservoir
