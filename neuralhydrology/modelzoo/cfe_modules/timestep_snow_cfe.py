import torch
from neuralhydrology.modelzoo.cfe_modules.calculate_PET_from_rainfall import calculate_PET_from_rainfall
from neuralhydrology.modelzoo.cfe_modules.run_shm_snow_module import run_shm_snow_module
from neuralhydrology.modelzoo.cfe_modules.adjust_and_track_runoff_infiltration import adjust_and_track_runoff_infiltration
from neuralhydrology.modelzoo.cfe_modules.adjust_from_soil_outflux import adjust_from_soil_outflux
from neuralhydrology.modelzoo.cfe_modules.calculate_convolutional_integral_for_GIUH import calculate_convolutional_integral_for_GIUH
from neuralhydrology.modelzoo.cfe_modules.calculate_evaporation_from_rainfall import calculate_evaporation_from_rainfall
from neuralhydrology.modelzoo.cfe_modules.calculate_evaporation_from_soil import calculate_evaporation_from_soil
from neuralhydrology.modelzoo.cfe_modules.calculate_gw_reservoir_flux import calculate_gw_reservoir_flux
from neuralhydrology.modelzoo.cfe_modules.cfe_dataclasses import CFEParams, Flux, GroundwaterStates, RoutingInfo, Snow_CFEParams, SnowStates, SoilStates
from neuralhydrology.modelzoo.cfe_modules.get_and_calculate_rainfall_snow_snowmelt import get_and_calculate_rainfall_snow_snowmelt
from neuralhydrology.modelzoo.cfe_modules.percolation_and_lateral_flow import percolation_and_lateral_flow
from neuralhydrology.modelzoo.cfe_modules.run_classic_soil_moisture_subroutine import run_classic_soil_moisture_subroutine
from neuralhydrology.modelzoo.cfe_modules.run_nash_cascade import run_nash_cascade
from neuralhydrology.modelzoo.cfe_modules.run_Schaake_subroutine import run_Schaake_subroutine

## NB: Let's ensure titles of subroutines are super self-explanatory.


def timestep_snow_cfe(
    x_conceptual_timestep: torch.Tensor,
    cfe_params: CFEParams,
    snow_cfe_params: Snow_CFEParams,
    timestep_params: dict | None,  # for consistency changed timestep_parameters --> timestep_params
    snow_reservoir: SnowStates,
    gw_reservoir: GroundwaterStates,
    soil_reservoir: SoilStates,
    soil_config,
    routing_info: RoutingInfo,
    constants,
)->tuple[Flux, SnowStates, GroundwaterStates, SoilStates, RoutingInfo, Flux]:  # enumerate what this returns. If cfe_params is not modified by this function, do not return it.
    ## INITIALIZE
    # timestep basin constants
    
    if timestep_params is not None:
        # dynamic parameters change every timestep, so update all dependent states in place
        cfe_params.update(timestep_params)
        snow_cfe_params.update(timestep_params)
        gw_reservoir.update(cfe_params)
        soil_config.update(cfe_params)
        soil_reservoir.update(cfe_params, soil_config, constants)
        

    flux = Flux(
        device=x_conceptual_timestep.device, batch_size=x_conceptual_timestep.shape[0]
    )  # no longer need function initialize_flux_timestep as this is handled by the __init__ method of the Flux class.

    ## UPDATES
    flux = get_and_calculate_rainfall_snow_snowmelt(
        conceptual_forcing_timestep=x_conceptual_timestep, flux=flux, cfe_params=cfe_params, snow_cfe_params=snow_cfe_params
    )  # hourly is now in cfe_params. What shall we do with constants?

    flux = calculate_PET_from_rainfall(conceptual_forcing_timestep=x_conceptual_timestep,flux=flux, cfe_params=cfe_params, constants=constants) # calculate PET from rainfall before adjusting for snowmelt and snowfall, as snowmelt and snowfall can reduce the amount of rainfall available for ET. This order is important for accurately calculating the evaporation from rainfall in the next step.
    
    flux = calculate_evaporation_from_rainfall(flux=flux)

    flux, soil_reservoir = calculate_evaporation_from_soil(flux=flux,  soil_reservoir=soil_reservoir)
    
    flux, snow_reservoir = run_shm_snow_module(flux=flux, snow_reservoir=snow_reservoir) # update snowmelt and snow storage based on the degree-day method. This will adjust the snowmelt flux based on the available snow storage and update the snow storage based on the snowfall input and the adjusted snowmelt.
    
    # infiltration partitioning.
    if cfe_params.dcfe_partition_scheme == "Schaake":
        flux, soil_reservoir = run_Schaake_subroutine(
            flux=flux, constants=constants, cfe_params=cfe_params, soil_reservoir=soil_reservoir, soil_config=soil_config
        )
    else:
        raise NotImplementedError(f"Partition scheme {cfe_params.dcfe_partition_scheme} not implemented.")

    flux, soil_reservoir = adjust_and_track_runoff_infiltration(flux=flux, soil_reservoir=soil_reservoir)

    # soil moisture reservoir.
    if cfe_params.dcfe_soil_scheme == "classic":
        flux, soil_reservoir = run_classic_soil_moisture_subroutine(flux=flux, soil_reservoir=soil_reservoir)
    else:
        raise NotImplementedError(f"Soil scheme {cfe_params.dcfe_soil_scheme} not implemented.")

    flux, soil_reservoir = adjust_from_soil_outflux(flux=flux, soil_reservoir=soil_reservoir)

    flux, gw_reservoir = percolation_and_lateral_flow(flux=flux, gw_reservoir=gw_reservoir)

    flux, gw_reservoir = calculate_gw_reservoir_flux(
        timestep_conceptual_forcing=x_conceptual_timestep, flux=flux, cfe_params=cfe_params, gw_reservoir=gw_reservoir
    )

    # surface runoff routing
    flux, routing_info = calculate_convolutional_integral_for_GIUH(flux=flux, routing_info=routing_info, cfe_params=cfe_params)

    # lateral flow routing
    flux, cfe_params = run_nash_cascade(flux=flux, routing_info=routing_info, cfe_params=cfe_params)

    ### FINALIZE
    flux.Qout_m = flux.giuh_runoff_m + flux.nash_lateral_runoff_m + flux.from_deep_gw_to_chan_m

    return cfe_params, snow_reservoir, gw_reservoir, soil_reservoir, routing_info, flux
