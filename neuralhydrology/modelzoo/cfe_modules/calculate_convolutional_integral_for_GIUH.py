from neuralhydrology.modelzoo.cfe_modules.cfe_dataclasses import CFEParams, Flux, RoutingInfo


def calculate_convolutional_integral_for_GIUH(
    flux: Flux, routing_info: RoutingInfo, cfe_params: CFEParams
) -> tuple[Flux, RoutingInfo]:
    # DM: @Ziyu can you explain why we set the last element to zero?
    routing_info.runoff_queue_m_per_timestep[:, routing_info.num_ordinates] = 0.0

    # Add incoming surface runoff to queue
    routing_info.runoff_queue_m_per_timestep[:, :-1] += (
        cfe_params.basin_characteristics.giuh_ordinates * flux.surface_runoff_depth_m.expand(routing_info.num_ordinates, -1).T
    )

    # Take top one in runoff queue as runoff to channel
    flux.giuh_runoff_m = routing_info.runoff_queue_m_per_timestep[:, 0].clone()

    # Shift the queue down by one
    routing_info.runoff_queue_m_per_timestep[:, :-1] = routing_info.runoff_queue_m_per_timestep[:, 1:].clone()

    return flux, routing_info
