import torch

from neuralhydrology.modelzoo.cfe_modules.cfe_dataclasses import Flux, GroundwaterStates


def percolation_and_lateral_flow(flux: Flux, gw_reservoir: GroundwaterStates) -> tuple[Flux, GroundwaterStates]:
    ### calculate_groundwater_storage_deficit
    gw_reservoir.storage_deficit_m = gw_reservoir.storage_max_m - gw_reservoir.storage_m
    ### adjust_precolation_to_gw
    overflow_mask = flux.flux_perc_m > gw_reservoir.storage_deficit_m

    # When the groundwater storage is full, the overflowing amount goes to direct runoff
    if torch.any(overflow_mask):
        amount_of_overflow = flux.flux_perc_m[overflow_mask] - gw_reservoir.storage_deficit_m[overflow_mask]

        # Overflow goes to surface runoff. Not sure where this is from.
        flux.surface_runoff_depth_m[overflow_mask] += amount_of_overflow

        # Reduce the infiltration (maximum possible flux_perc_m is equal to gw_reservoir_storage_deficit_m)
        flux.flux_perc_m[overflow_mask] = gw_reservoir.storage_deficit_m[overflow_mask]

        # Saturate the Groundwater storage. I believe storage + flux_prec_m saturated = storage max
        gw_reservoir.storage_m[overflow_mask] = gw_reservoir.storage_max_m[overflow_mask]

        gw_reservoir.storage_deficit_m[overflow_mask] = 0.0

    # If no overflow, all percolation flux goes to storage
    no_overflow_mask = ~overflow_mask
    if torch.any(no_overflow_mask):
        gw_reservoir.storage_m[no_overflow_mask] += flux.flux_perc_m[no_overflow_mask]
        gw_reservoir.storage_deficit_m[no_overflow_mask] -= flux.flux_perc_m[no_overflow_mask]

    return flux, gw_reservoir
