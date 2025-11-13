from neuralhydrology.modelzoo.cfe_modules.cfe_dataclasses import CFEParams, Flux, RoutingInfo


def calculate_convolutional_integral_for_GIUH(
    flux: Flux, routing_info: RoutingInfo, cfe_params: CFEParams
) -> tuple[Flux, RoutingInfo]:
    """Calculate the convolutional integral for GIUH runoff routing.
    
    Args:
        flux (Flux): Flux dataclass containing surface runoff depth and GIUH runoff.
        routing_info (RoutingInfo): RoutingInfo dataclass containing runoff queue and number of ordinates
        cfe_params (CFEParams): CFEParams dataclass containing basin characteristics including GIUH ordinates.
    Returns:
        flux:
            - giuh_runoff_m (torch.Tensor): GIUH runoff depth [m] routed to channel.
        routing_info:
            - runoff_queue_m_per_timestep (torch.Tensor): Updated runoff queue after GIUH routing
    """
    # INITIALIZE
    # set the last element in the runoff queue to zero (runoff_queue[:-1] were pushed forward in last timestep)
    routing_info.runoff_queue_m_per_timestep[:, routing_info.num_ordinates] = 0.0
    
    # UPDATE
    # Add incoming surface runoff to queue
    routing_info.runoff_queue_m_per_timestep[:, :-1] += (
        cfe_params.basin_characteristics.giuh_ordinates * flux.surface_runoff_depth_m.expand(routing_info.num_ordinates, -1).T
    )
    # FINALIZE
    # Take top one in runoff queue as runoff to channel
    flux.giuh_runoff_m = routing_info.runoff_queue_m_per_timestep[:, 0].clone()

    # Shift the queue down by one to prepare for next timestep
    routing_info.runoff_queue_m_per_timestep[:, :-1] = routing_info.runoff_queue_m_per_timestep[:, 1:].clone()

    return flux, routing_info
