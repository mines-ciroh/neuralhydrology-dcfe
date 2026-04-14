from typing import Dict, Union

import torch

from neuralhydrology.modelzoo.baseconceptualmodel import BaseConceptualModel
from neuralhydrology.modelzoo.cfe_modules.cfe_dataclasses import (
    INITIAL_STATES_SNOW,
    SNOW_CFE_PARAMETERS_RANGES,
    GroundwaterStates,
    Snow_CFEParams,
    RoutingInfo,
    SoilConfig,
    SoilStates,
    SnowStates,
    get_constants,
)
from neuralhydrology.modelzoo.cfe_modules.get_default_params import get_default_params
from neuralhydrology.modelzoo.cfe_modules.timestep_snow_cfe import timestep_snow_cfe
from neuralhydrology.utils.config import Config


class SnowDCFE(BaseConceptualModel):
    """
    Fully differentiable implementation of CFE based upon  https://github.com/NWC-CUAHSI-Summer-Institute/ngen-aridity/blob/main/Project%20Manuscript_LongForm.pdf
    ten parameters are now differentiable, and controlled by an LSTM, not an MLP.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.constants = get_constants(cfg.dcfe_hourly)

    def forward(
        self, x_conceptual: torch.Tensor, lstm_out: torch.Tensor, additional_features: torch.Tensor
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        ## INITIALIZE
        device = x_conceptual.device
        batch_size = x_conceptual.shape[0]
        self.cfe_params = get_default_params(self.cfg, additional_features, device)  # fetch default params for basins in batch.
        self.snow_cfe_params = Snow_CFEParams(device=device, batch_size=batch_size)  # initialize snow CFE parameters
        dynamic_parameters = self._get_dynamic_parameters_conceptual(
            lstm_out=lstm_out
        )  # convert lstm output to appropriate range for each param.

        # initialize structures to store the information
        states, out = self._initialize_information(conceptual_inputs=x_conceptual)

        # initialize model states/reservoirs.
        constants = get_constants(self.cfg.dcfe_hourly)
        snow_reservoir = SnowStates(device=device, batch_size=batch_size)
        gw_reservoir = GroundwaterStates(device=device, batch_size=batch_size, cfe_params=self.cfe_params)
        soil_config = SoilConfig(cfe_params=self.cfe_params, device=device, batch_size=batch_size, constants=self.constants)
        soil_reservoir = SoilStates(
            device=device,
            batch_size=batch_size,
            cfe_params=self.cfe_params,
            soil_config=soil_config,
            constants=self.constants,
        )
        routing_info = RoutingInfo(device=device, batch_size=batch_size, cfe_params=self.cfe_params)

        # TODO: want to refactor code so this type of dynamic parameter update is universal for conceptual
        conceptual_param = self._form_conceptual_input_param(dynamic_parameters)

        ## Spinup CFE module. Do not track gradients.
        with torch.no_grad():
            for j in range(0, self.cfg.spin_up_period):
                # grab the parameters for this timestep.
                timestep_conceptual_param = {}
                for k in dynamic_parameters.keys():
                    timestep_conceptual_param[k] = conceptual_param[k][:, j]

                self.cfe_params, snow_reservoir, gw_reservoir, soil_reservoir, routing_info, flux = timestep_snow_cfe(
                    x_conceptual_timestep=x_conceptual[:, j, :],
                    cfe_params=self.cfe_params,
                    snow_cfe_params=self.snow_cfe_params,
                    timestep_params=timestep_conceptual_param,
                    snow_reservoir=snow_reservoir,
                    gw_reservoir=gw_reservoir,
                    soil_reservoir=soil_reservoir,
                    soil_config=soil_config,
                    routing_info=routing_info,
                    constants=constants,
                )

                ##FINALIZE
                states, out = self._store_timestep_information(j, flux, snow_reservoir,gw_reservoir, soil_reservoir, states, out)

        # now run dCFE for prediction. Gradients are tracked.
        for i in range(self.cfg.spin_up_period, lstm_out.shape[1]):
            # grab the parameters for this timestep.
            timestep_conceptual_param = {}
            for k in dynamic_parameters.keys():
                timestep_conceptual_param[k] = conceptual_param[k][:, i]

            self.cfe_params, snow_reservoir, gw_reservoir, soil_reservoir, routing_info, flux = timestep_snow_cfe(
                x_conceptual_timestep=x_conceptual[:, i, :],
                cfe_params=self.cfe_params,
                snow_cfe_params=self.snow_cfe_params,
                timestep_params=timestep_conceptual_param,
                snow_reservoir=snow_reservoir,
                gw_reservoir=gw_reservoir,
                soil_reservoir=soil_reservoir,
                soil_config=soil_config,
                routing_info=routing_info,
                constants=constants,
            )

            ## FINALIZE
            states, out = self._store_timestep_information(i, flux, snow_reservoir, gw_reservoir, soil_reservoir, states, out)

        return {"y_hat": out, "parameters": conceptual_param, "internal_states": states}

    def _store_timestep_information(self, timestep_idx, flux, snow_reservoir,gw_reservoir, soil_reservoir, states, out):
        out[:, timestep_idx, 0] = flux.Qout_m * 1000
        states["gw_reservoir_storage_m"][:, timestep_idx] = gw_reservoir.storage_m
        states["soil_reservoir_storage_m"][:, timestep_idx] = soil_reservoir.storage_m
        states["snow_storage_m"][:, timestep_idx] = snow_reservoir.storage_m
        states["flux_giuh_runoff_m"][:, timestep_idx] = flux.giuh_runoff_m
        states["flux_nash_lateral_runoff_m"][:, timestep_idx] = flux.nash_lateral_runoff_m
        states["flux_from_deep_gw_to_chan_m"][:, timestep_idx] = flux.from_deep_gw_to_chan_m
        states["surface_runoff_depth_m"][:, timestep_idx] = flux.surface_runoff_depth_m
        states["actual_et_m_per_timestep"][:, timestep_idx] = flux.actual_et_m_per_timestep
        return states, out

    @property
    def parameter_ranges(self):
        return SNOW_CFE_PARAMETERS_RANGES

    @property
    def initial_states(self):
        return INITIAL_STATES_SNOW
