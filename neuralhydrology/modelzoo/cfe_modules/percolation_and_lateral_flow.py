import torch

from neuralhydrology.modelzoo.cfe_modules.cfe_dataclasses import Flux, GroundwaterStates


# TODO: Check storage_deficit_m against original code.
def percolation_and_lateral_flow(flux: Flux, gw_reservoir: GroundwaterStates) -> tuple[Flux, GroundwaterStates]:
    """Handle percolation to groundwater and lateral flow.
    Args:
        flux (Flux): Flux dataclass containing flux variables.
        gw_reservoir (GroundwaterStates): GroundwaterStates dataclass containing groundwater state variables
    Returns:
        flux:
            - flux_perc_m (torch.Tensor): Updated percolation flux [m/timestep].
            - surface_runoff_depth_m (torch.Tensor): Updated surface runoff depth [m/timestep].
        gw_reservoir:
            - storage_m (torch.Tensor): Updated groundwater storage [m/timestep].
    """

    ### calculate_groundwater_storage_deficit
    storage_deficit_m = gw_reservoir.storage_max_m - gw_reservoir.storage_m
    ### adjust_precolation_to_gw
    overflow_mask = flux.flux_perc_m > storage_deficit_m

    # When the groundwater storage is full, the overflowing amount goes to direct runoff
    if torch.any(overflow_mask):
        amount_of_overflow = flux.flux_perc_m[overflow_mask] - storage_deficit_m[overflow_mask]

        # Overflow goes to surface runoff. Not sure where this is from.
        flux.surface_runoff_depth_m[overflow_mask] = flux.surface_runoff_depth_m[overflow_mask] + amount_of_overflow

        # Reduce the infiltration (maximum possible flux_perc_m is equal to gw_reservoir_storage_deficit_m)
        flux.flux_perc_m[overflow_mask] = storage_deficit_m[overflow_mask]

        # Saturate the Groundwater storage. I believe storage + flux_prec_m saturated = storage max
        gw_reservoir.storage_m[overflow_mask] = gw_reservoir.storage_max_m[overflow_mask]

        storage_deficit_m[overflow_mask] = 0.0

    # If no overflow, all percolation flux goes to storage
    no_overflow_mask = ~overflow_mask
    if torch.any(no_overflow_mask):
        gw_reservoir.storage_m[no_overflow_mask] = gw_reservoir.storage_m[no_overflow_mask] + flux.flux_perc_m[no_overflow_mask]
        # storage_deficit_m[no_overflow_mask] -= flux.flux_perc_m[
        #    no_overflow_mask
        # ]  # NOTE (11/10/2025). This line is not in original dcfe code. (11/11/2025) After experimenting, does not impact results.

    return flux, gw_reservoir
