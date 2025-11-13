import torch

from neuralhydrology.modelzoo.cfe_modules.cfe_dataclasses import Flux, SoilStates


def calculate_evaporation_from_soil(flux: Flux, soil_reservoir: SoilStates):
    """ Calculate evaporation from soil moisture with classic soil moisture scheme.
    Args:
        flux (Flux): Flux dataclass containing flux variables.
        soil_reservoir (SoilStates): SoilStates dataclass containing soil state variables.

    Returns:
        flux:
            - actual_et_from_soil_m_per_timestep (torch.Tensor): Actual evapotranspiration from soil [m/timestep].
            - actual_et_m_per_timestep (torch.Tensor): Total actual evapotranspiration [m/timestep].
            - reduced_potential_et_m_per_timestep (torch.Tensor): Reduced potential evapotranspiration [m/timestep].
        soil_reservoir:
            - storage_m (torch.Tensor): Updated soil moisture storage [m/timestep].       
    """

    # INITIALIZE
    soil_wilting_mask = soil_reservoir.storage_m > soil_reservoir.wilting_point_m
    reduced_pet_mask = flux.reduced_potential_et_m_per_timestep > 0.0
    # check if any basins in the batch have soil moisture above wilting point.
    if torch.any(soil_wilting_mask) and torch.any(reduced_pet_mask):
        # currently only handle the classic soil scheme.
        reduced_pet = flux.reduced_potential_et_m_per_timestep[soil_wilting_mask & reduced_pet_mask]
        storage_threshold_prim = soil_reservoir.storage_threshold_primary_m[soil_wilting_mask & reduced_pet_mask]
        actual_et_soil = flux.actual_et_from_soil_m_per_timestep[soil_wilting_mask & reduced_pet_mask]
        soil_storage = soil_reservoir.storage_m[soil_wilting_mask & reduced_pet_mask]
        wilting_point = soil_reservoir.wilting_point_m[soil_wilting_mask & reduced_pet_mask]

        storage_threshold_mask = soil_storage >= storage_threshold_prim

        # UPDATE
        # Ziyu (11/13/2025): NH-dCFE code did not have soil_storage[storage_threshold_mask] - wilting_point[storage_threshold_mask]; 
        # only soil_storage[storage_threshold_mask]
        # if soil_storage > storage_threshold_primary_m:
        #actual_et_soil[storage_threshold_mask] = torch.min(
        #    reduced_pet[storage_threshold_mask], soil_storage[storage_threshold_mask] - wilting_point[storage_threshold_mask]
        #)
        actual_et_soil[storage_threshold_mask] = torch.min(
            reduced_pet[storage_threshold_mask], soil_storage[storage_threshold_mask]
        )

        # If soil_storage < storage_threshold_primary_m:
        Budyko_numerator = soil_storage[~storage_threshold_mask] - wilting_point[~storage_threshold_mask]
        Budyko_denominator = storage_threshold_prim[~storage_threshold_mask] - wilting_point[~storage_threshold_mask]
        Budyko_ratio = Budyko_numerator / Budyko_denominator
        actual_et_soil[~storage_threshold_mask] = torch.min(
            reduced_pet[~storage_threshold_mask] * Budyko_ratio, soil_storage[~storage_threshold_mask]
        )

        # FINALIZE
        flux.actual_et_from_soil_m_per_timestep[soil_wilting_mask & reduced_pet_mask] = actual_et_soil
        soil_reservoir.storage_m -= flux.actual_et_from_soil_m_per_timestep
        flux.reduced_potential_et_m_per_timestep -= flux.actual_et_from_soil_m_per_timestep
        flux.actual_et_m_per_timestep += flux.actual_et_from_soil_m_per_timestep

    return flux, soil_reservoir
