from typing import Dict, Union

import torch

from neuralhydrology.modelzoo.baseconceptualmodel import BaseConceptualModel
from neuralhydrology.modelzoo.cfe_modules.cfe_dataclasses import (
    INITIAL_STATES,
    PARAMETER_RANGES,
    Flux,
    GroundwaterStates,
    RoutingInfo,
    SoilConfig,
    SoilStates,
    get_constants,
)
from neuralhydrology.modelzoo.cfe_modules.get_default_params import get_default_params
from neuralhydrology.modelzoo.cfe_modules.timestep_cfe import timestep_cfe
from neuralhydrology.utils.config import Config


class DCFE(BaseConceptualModel):
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

        dynamic_parameters = self._get_dynamic_parameters_conceptual(
            lstm_out=lstm_out
        )  # convert lstm output to appropriate range for each param.

        # initialize structures to store the information
        states, out = self._initialize_information(conceptual_inputs=x_conceptual, lstm_out=lstm_out)

        # initialize model states/reservoirs.
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
        flux = Flux(device=device, batch_size=batch_size)

        ## Spinup CFE module. Do not track gradients.
        with torch.no_grad():
            for j in range(0, self.cfg.spin_up):
                if self.cfg.dcfe_predict_config == "dynamic":
                    self.cfe_params.update(dynamic_parameters, j)

                gw_reservoir, soil_reservoir, routing_info, flux = timestep_cfe(
                    x_conceptual_timestep=x_conceptual[:, j, :],
                    cfe_params=self.cfe_params,
                    gw_reservoir=gw_reservoir,
                    soil_reservoir=soil_reservoir,
                    routing_info=routing_info,
                    flux=flux,
                )

                ##FINALIZE
                states, out = self._store_timestep_information(j, flux, gw_reservoir, soil_reservoir, states, out)

        # now run dCFE for prediction. Gradients are tracked.
        for k in range(self.cfg.spin_up, lstm_out.shape[1]):
            if self.cfg.dcfe_predict_config == "dynamic":
                self.cfe_params.update(dynamic_parameters, k)

            ## UPDATE
            gw_reservoir, soil_reservoir, routing_info, flux = timestep_cfe(
                x_conceptual_timestep=x_conceptual[:, k, :],
                cfe_params=self.cfe_params,
                gw_reservoir=gw_reservoir,
                soil_reservoir=soil_reservoir,
                routing_info=routing_info,
                flux=flux,
            )

            ## FINALIZE
            states, out = self._store_timestep_information(k, flux, gw_reservoir, soil_reservoir, routing_info, states, out)

        return {"y_hat": out, "parameters": dynamic_parameters, "internal_states": states}

    def _store_timestep_information(self, timestep_idx, flux, gw_reservoir, soil_reservoir, routing_info, states, out):
        out[:, timestep_idx, 0] = flux.Qout_m * 1000
        states["gw_reservoir_storage_m"][:, timestep_idx] = gw_reservoir.storage_m
        states["soil_reservoir_storage_m"][:, timestep_idx] = soil_reservoir.storage_m
        states["first_nash_storage"][:, timestep_idx] = self.cfe_params.basin_characteristics.nash_storage[:, 0]
        return states, out

    @property
    def parameter_ranges(self):
        return PARAMETER_RANGES

    @property
    def initial_states(self):
        return INITIAL_STATES
