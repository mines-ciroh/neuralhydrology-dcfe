from neuralhydrology.modelzoo.cfe_modules.cfe_dataclasses import Flux, SoilStates


def adjust_from_soil_outflux(flux: Flux, constants, soil_reservoir: SoilStates) -> tuple[Flux, SoilStates]:
    flux.flux_perc_m = flux.primary_flux_m  # percolation_flux
    flux.flux_lat_m = flux.secondary_flux_m  # lateral_flux

    # only implemented update for classic soil scheme.
    soil_reservoir.storage_m -= flux.flux_perc_m + flux.flux_lat_m

    return flux, soil_reservoir
