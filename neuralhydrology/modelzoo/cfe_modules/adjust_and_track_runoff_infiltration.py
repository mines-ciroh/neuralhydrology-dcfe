import torch

from neuralhydrology.modelzoo.cfe_modules.cfe_dataclasses import Flux, SoilStates


def adjust_and_track_runoff_infiltration(flux: Flux, soil_reservoir: SoilStates) -> tuple[Flux, SoilStates]:
    """
    Calculates saturation excess overland flow (SOF)
    If the infiltration depth exceeds the soil storage deficit, additional SOF occurs
    and soil get saturated.
    This should be run after calculate_infiltration_excess_overland_flow, then,
    infiltration_depth_m and surface_runoff_depth_m get finalized
    Args:
        flux: Flux dataclass containing infiltration and runoff depths
        soil_reservoir: SoilStates dataclass containing soil reservoir states
    Returns:
        Updated flux:
            - infiltration_depth_m
            - surface_runoff_depth_m
        Updated soil_reservoir:
            - storage_m
            - storage_deficit_m
    """
    # INITIALIZE
    excess_infil_mask = soil_reservoir.storage_deficit_m < flux.infiltration_depth_m

    if torch.any(excess_infil_mask):
        diff = (flux.infiltration_depth_m - soil_reservoir.storage_deficit_m)[excess_infil_mask]

        # Adjusting the surface runoff and infiltration depths for the specific basins
        flux.surface_runoff_depth_m[excess_infil_mask] = flux.surface_runoff_depth_m[excess_infil_mask] + diff
        flux.infiltration_depth_m[excess_infil_mask] = soil_reservoir.storage_deficit_m[excess_infil_mask]

        # This was missing from original implementation, added by Ziyu 11/11/24 from c code line 142
        soil_reservoir.storage_m[excess_infil_mask] = soil_reservoir.storage_max_m[excess_infil_mask]
        # Setting the soil reservoir storage deficit to zero for the specific basins
        soil_reservoir.storage_deficit_m[excess_infil_mask] = 0.0

    return flux, soil_reservoir
