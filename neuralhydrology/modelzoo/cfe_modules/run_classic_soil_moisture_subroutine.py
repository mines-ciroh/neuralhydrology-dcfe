import torch

from neuralhydrology.modelzoo.cfe_modules.cfe_dataclasses import Flux, SoilStates


def run_classic_soil_moisture_subroutine(flux: Flux, soil_reservoir: SoilStates) -> tuple[Flux, SoilStates]:
    """
    Here and elsewhere, it might be helpful to list here what fields of flux and soil_reservoir are updated by this function.
    """
    mask_perc_soil = flux.flux_perc_m > soil_reservoir.storage_deficit_m
    if torch.any(mask_perc_soil):
        diff_perc_soil = flux.flux_perc_m[mask_perc_soil] - soil_reservoir.storage_deficit_m[mask_perc_soil]
        flux.infiltration_depth_m[mask_perc_soil] = soil_reservoir.storage_deficit_m[mask_perc_soil]
        # not added: lines 170 & 171 send flow back to giuh in self.vol & correct over-prediction of infilt
        flux.surface_runoff_depth_m[mask_perc_soil] = flux.surface_runoff_depth_m[mask_perc_soil] + diff_perc_soil
        soil_reservoir.storage_deficit_m[mask_perc_soil] = 0.0

    # Assumes we don't have a single outlet exponential gw storage...
    # Add infiltration flux and calculate the reservoir flux
    # this is adjusted for ET already (not sure where this is from)
    soil_reservoir.storage_m = soil_reservoir.storage_m + flux.infiltration_depth_m

    ## do soil_conceptual_reservoir_flux_calc
    # Calculate primary flux
    storage_above_threshold_primary = soil_reservoir.storage_m - soil_reservoir.storage_threshold_primary_m
    primary_flux_mask = storage_above_threshold_primary > 0.0

    if torch.any(primary_flux_mask):
        storage_diff_primary = (
            soil_reservoir.storage_max_m[primary_flux_mask] - soil_reservoir.storage_threshold_primary_m[primary_flux_mask]
        )
        storage_ratio_primary = storage_above_threshold_primary[primary_flux_mask] / storage_diff_primary
        storage_power_primary = torch.pow(
            storage_ratio_primary, soil_reservoir.exponent_primary
        )  # "exponent primary" is scalar for now but can try to init as tensor
        flux.primary_flux_m[primary_flux_mask] = soil_reservoir.coeff_primary[primary_flux_mask] * storage_power_primary

        # a mask for when primary_flux > storage_above_primary
        primary_above_mask = flux.primary_flux_m > storage_above_threshold_primary

        # if primary_flux_m > storage_above_threshold_m then
        flux.primary_flux_m[primary_flux_mask & primary_above_mask] = storage_above_threshold_primary[
            primary_flux_mask & primary_above_mask
        ].clone()

    # Calculate secondary flux
    storage_above_threshold_secondary = soil_reservoir.storage_m - soil_reservoir.storage_threshold_secondary_m
    secondary_flux_mask = storage_above_threshold_secondary > 0.0
    if torch.any(secondary_flux_mask):
        storage_diff_secondary = (
            soil_reservoir.storage_max_m[secondary_flux_mask] - soil_reservoir.storage_threshold_secondary_m[secondary_flux_mask]
        )
        storage_ratio_secondary = storage_above_threshold_secondary[secondary_flux_mask] / storage_diff_secondary
        storage_power_secondary = torch.pow(
            storage_ratio_secondary, soil_reservoir.exponent_secondary
        )  # "exponent_secondary" is also a scalar for now
        flux.secondary_flux_m[secondary_flux_mask] = soil_reservoir.coeff_secondary[secondary_flux_mask] * storage_power_secondary
        # crate a mask for when secondary_flux > storage_above_secondary - primary_flux_m
        secondary_above_mask = flux.secondary_flux_m > (storage_above_threshold_secondary - flux.primary_flux_m)

        # if above is true then
        flux.secondary_flux_m[secondary_flux_mask & secondary_above_mask] = (
            storage_above_threshold_secondary - flux.primary_flux_m
        )[secondary_flux_mask & secondary_above_mask]

    return flux, soil_reservoir
