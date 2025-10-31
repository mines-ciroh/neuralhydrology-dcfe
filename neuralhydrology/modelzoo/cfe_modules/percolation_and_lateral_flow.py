from neuralhydrology.modelzoo.cfe_modules.cfe_dataclasses import Flux, GroundwaterStates
import torch


def percolation_and_lateral_flow(flux: Flux, gw_reservoir: GroundwaterStates) -> tuple[Flux, GroundwaterStates]:

    ### calculate_groundwater_storage_deficit
    gw_reservoir.storage_deficit_m = gw_reservoir.storage_max_m - gw_reservoir.storage_m
    ### adjust_precolation_to_gw
    overflow_mask = flux.flux_perc_m > gw_reservoir.storage_deficit_m

    # When the groundwater storage is full, the overflowing amount goes to direct runoff
    if torch.any(overflow_mask):

    return flux, gw_reservoir
