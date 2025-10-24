import torch
from adjust_and_track_runoff_infiltration import adjust_and_track_runoff_infiltration
from adjust_from_soil_outflux import adjust_from_soil_outflux
from calculate_convolutional_integral_for_GIUH import calculate_convolutional_integral_for_GIUH
from calculate_evaporation_from_rainfall import calculate_evaporation_from_rainfall
from calculate_evaporation_from_soil import calculate_evaporation_from_soil
from calculate_gw_reservoir_flux import calculate_gw_reservoir_flux
from cfe_dataclasses import CONSTANTS, CFEParams, Flux, GroundwaterStates, RoutingInfo, SoilStates
from get_and_calculate_input_rainfall_and_ET import get_and_calculate_input_rainfall_and_ET
from percolation_and_lateral_flow import percolation_and_lateral_flow
from run_classic_soil_moisture_subroutine import run_classic_soil_moisture_subroutine
from run_nash_cascade import run_nash_cascade
from run_Schaake_subroutine import run_Schaake_subroutine
from timestep_basin_constants import timestep_basin_constants

## NB: Let's ensure titles of subroutines are super self-explanatory.


def timestep_cfe(
    x_conceptual_timestep: torch.Tensor,
    cfe_params: CFEParams,
    timestep_params: None,  # for consistency changed timestep_parameters --> timestep_params
    gw_reservoir: GroundwaterStates,
    soil_reservoir: SoilStates,
    routing_info: RoutingInfo,
    constants=CONSTANTS,
):
    ## INITIALIZE
    cfe_params, gw_reservoir, soil_reservoir = timestep_basin_constants(
        conceptual_forcing_timestep=x_conceptual_timestep,
        gw_reservoir=gw_reservoir,
        soil_reservoir=soil_reservoir,
        cfe_params=cfe_params,
        # constants=constants,
        timestep_params=timestep_params,
    )

    flux = Flux(
        device=x_conceptual_timestep.device, batch_size=x_conceptual_timestep.shape[0]
    )  # no longer need function initialize_flux_timestep as this is handled by the __init__ method of the Flux class.

    ## UPDATES
    flux = get_and_calculate_input_rainfall_and_ET(
        conceptual_forcing_timestep=x_conceptual_timestep, flux=flux, cfe_params=cfe_params
    )  # hourly is now in cfe_params. What shall we do with constants?

    flux = calculate_evaporation_from_rainfall(flux=flux)

    flux, soil_reservoir = calculate_evaporation_from_soil(flux=flux, constants=constants, soil_reservoir=soil_reservoir)

    # infiltration partitioning.
    if cfe_params.dcfe_partition_scheme == "Schaake":
        flux, soil_reservoir = run_Schaake_subroutine(
            flux=flux, constants=constants, cfe_params=cfe_params, soil_reservoir=soil_reservoir
        )
    else:
        raise NotImplementedError(f"Partition scheme {cfe_params.dcfe_partition_scheme} not implemented.")

    flux, soil_reservoir = adjust_and_track_runoff_infiltration(flux=flux, soil_reservoir=soil_reservoir)

    # soil moisture reservoir.
    if cfe_params.dcfe_soil_scheme == "classic":
        flux, soil_reservoir = run_classic_soil_moisture_subroutine(flux=flux, soil_reservoir=soil_reservoir)
    else:
        raise NotImplementedError(f"Soil scheme {cfe_params.dcfe_soil_scheme} not implemented.")

    flux, soil_reservoir = adjust_from_soil_outflux(flux=flux, constants=constants, soil_reservoir=soil_reservoir)

    flux, gw_reservoir = percolation_and_lateral_flow(flux=flux, gw_reservoir=gw_reservoir)

    flux, gw_reservoir = calculate_gw_reservoir_flux(
        timestep_conceptual_forcing=x_conceptual_timestep, flux=flux, cfe_params=cfe_params, gw_reservoir=gw_reservoir
    )

    # surface runoff routing
    flux, routing_info = calculate_convolutional_integral_for_GIUH(flux=flux, routing_info=routing_info, cfe_params=cfe_params)

    # lateral flow routing
    flux, routing_info, cfe_params = run_nash_cascade(flux=flux, routing_info=routing_info, cfe_params=cfe_params)

    ### FINALIZE
    flux.Qout_m = flux.giuh_runoff_m + flux.nash_lateral_runoff_m + flux.from_deep_gw_to_chan_m

    return cfe_params, gw_reservoir, soil_reservoir, routing_info, flux
