from neuralhydrology.modelzoo.cfe_modules.cfe_dataclasses import Flux, SoilStates


def adjust_from_soil_outflux(flux: Flux, soil_reservoir: SoilStates) -> tuple[Flux, SoilStates]:
    """ Module to adjust the soil outflux for CFE model at every timestep, 
    update only implemented for classic soil scheme

    Args:
        flux (Flux): Flux dataclass containing fluxes
        soil_reservoir (SoilStates): SoilStates dataclass containing soil states
    Returns:
        flux:
            - flux_perc_m: percolation flux in m, equal to primary flux
            - flux_lat_m: lateral flux in m, equal to secondary flux
        soil_reservoir:
            - storage_m: updated soil storage in m for classic soil scheme only
    """
    flux.flux_perc_m = flux.primary_flux_m  # percolation_flux
    flux.flux_lat_m = flux.secondary_flux_m  # lateral_flux

    # only implemented update for classic soil scheme, old code includes option for ode
    soil_reservoir.storage_m -= flux.flux_perc_m + flux.flux_lat_m

    return flux, soil_reservoir
