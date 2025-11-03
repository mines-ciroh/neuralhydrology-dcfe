from neuralhydrology.modelzoo.cfe_modules.cfe_dataclasses import CFEParams, Flux, RoutingInfo


def run_nash_cascade(flux: Flux, routing_info: RoutingInfo, cfe_params: CFEParams) -> tuple[Flux, RoutingInfo, CFEParams]:
    nash_storage_timestep = cfe_params.basin_characteristics.nash_storage.clone()

    # Calculate discharge from each Nash storage
    Q = cfe_params.basin_characteristics.K_nash.unsqueeze(1) * nash_storage_timestep

    # Update Nash storage with discharge.
    nash_storage_timestep -= Q

    # First storage gets lateral flow outflux from soil storage
    nash_storage_timestep[:, 0] += flux.flux_lat_m

    # Remaining storage gets discharge from upper Nash storage.
    if routing_info.num_reservoirs > 1:
        nash_storage_timestep[:, 1:] += Q[:, :-1]

    # Update the state
    cfe_params.basin_characteristics.nash_storage = nash_storage_timestep

    # Final discharge from Nash cascade is from the lowermost Nash storage.
    flux.nash_lateral_runoff_m = Q[:, -1]

    return flux, routing_info, cfe_params
