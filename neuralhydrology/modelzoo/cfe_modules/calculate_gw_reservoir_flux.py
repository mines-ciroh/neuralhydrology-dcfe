import torch

from neuralhydrology.modelzoo.cfe_modules.cfe_dataclasses import CFEParams, Flux, GroundwaterStates


def calculate_gw_reservoir_flux(
    timestep_conceptual_forcing, flux: Flux, cfe_params: CFEParams, gw_reservoir: GroundwaterStates
) -> tuple[Flux, GroundwaterStates]:
    """
    This calculates the flux from a linear, or nonlinear
        conceptual reservoir with one or two outlets, or from an exponential nonlinear conceptual reservoir with only one outlet.
        In the non-exponential instance, each outlet can have its own activation storage threshold.  Flow from the second outlet is turned off by setting the discharge coeff. to 0.0.
    """
    device = timestep_conceptual_forcing.device
    batch_size = timestep_conceptual_forcing.shape[0]

    flux_exponential = torch.exp(
        gw_reservoir.exponent_primary * gw_reservoir.storage_m / gw_reservoir.storage_max_m
    ) - torch.ones((batch_size), dtype=torch.float32, device=device)

    flux.primary_flux_from_gw_m = torch.minimum(cfe_params.basin_characteristics.Cgw * flux_exponential, gw_reservoir.storage_m)

    flux.from_deep_gw_to_chan_m = flux.primary_flux_from_gw_m + flux.secondary_flux_from_gw_m
    # there's no 2nd flux since exponential
    # DM: Can we remove the secondary flux entirely?

    gw_reservoir.storage_m -= flux.from_deep_gw_to_chan_m

    return flux, gw_reservoir
