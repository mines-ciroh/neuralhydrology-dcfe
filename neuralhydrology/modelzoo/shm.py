from typing import Dict, Union

import torch

from neuralhydrology.modelzoo.baseconceptualmodel import BaseConceptualModel
from neuralhydrology.utils.config import Config


class SHM(BaseConceptualModel):
    """Modified version of SHM [#]_ hydrological model with dynamic parameterization.

    The SHM receives the dynamic parameterization given by a deep learning model. This class has two properties which
    define the initial conditions of the internal states of the model (buckets) and the ranges in which the model
    parameters are allowed to vary during optimization.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    References
    ----------
    .. [#] Ehret, U., van Pruijssen, R., Bortoli, M., Loritz, R., Azmi, E., and Zehe, E.: Adaptive clustering: reducing
        the computational costs of distributed (hydrological) modelling by exploiting time-variable similarity among
        model elements, Hydrology and Earth System Sciences, 24, 4389–4411, https://doi.org/10.5194/hess-24-4389-2020,
        2020.
    """

    def __init__(self, cfg: Config):
        super(SHM, self).__init__(cfg=cfg)

    def forward(
        self, x_conceptual: torch.Tensor, lstm_out: torch.Tensor
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Performs forward pass on the SHM model.

        In this forward pass, all elements of the batch are processed in  parallel.

        Parameters
        ----------
        x_conceptual: torch.Tensor
            Tensor of size [batch_size, time_steps, n_inputs]. The batch_size is associated with a certain basin and a
            certain prediction period. The time_steps refer to the number of time steps (e.g. days) that our conceptual
            model is going to be run for. The n_inputs refer to the dynamic forcings used to run the conceptual model
            (e.g. Precipitation, Temperature...)

        lstm_out: torch.Tensor
            Tensor of size [batch_size, time_steps, n_parameters]. The tensor comes from the data-driven model  and will
            be used to obtained the dynamic parameterization of the conceptual model

        Returns
        -------
        Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
            - y_hat: torch.Tensor
                Simulated outflow
            - parameters: Dict[str, torch.Tensor]
                Dynamic parameterization of the conceptual model
            - internal_states: Dict[str, torch.Tensor]]
                Time-evolution of the internal states of the conceptual model
        """
        ## INITIALIZE
        device = x_conceptual.device
        batch_size = x_conceptual.shape[0]

        # get model parameters
        dynamic_parameters = self._get_dynamic_parameters_conceptual(lstm_out=lstm_out)
        conceptual_parameters = self._form_conceptual_input_param(dynamic_parameters)

        # initialize structures to store the information
        states, out = self._initialize_information(conceptual_inputs=x_conceptual)

        # initialize model reservoirs
        ss, sf, su, si, sb = self.initialize_states(batch_size, device)

        # spin up SHM model. Do not track gradients
        with torch.no_grad():
            for j in range(0, self.cfg.spin_up_period):
                x_conceptual_timestep = x_conceptual[:, j, :]
                timestep_params = {}
                for k in conceptual_parameters.keys():
                    """
                    dynamic parameters is Dict[str, torch.Tensor] where torch.Tensor is of shape [batch_size, timesteps].
                    This reshapes to Dict[str, torch.Tensor] where torch.Tensor is of shape [batch_size].
                    So, just parameters for a specific timestep
                    """
                    timestep_params[k] = conceptual_parameters[k][:, j]

                ss, sf, su, si, sb, timestep_out = self.timestep_shm(
                    ss, sf, su, si, sb, timestep_params, x_conceptual_timestep, device
                )

        # Now run model for prediction while traking gradients
        for j in range(self.cfg.spin_up_period, lstm_out.shape[1]):
            x_conceptual_timestep = x_conceptual[:, j, :]
            timestep_params = {}
            for k in conceptual_parameters.keys():
                """
                dynamic parameters is Dict[str, torch.Tensor] where torch.Tensor is of shape [batch_size, timesteps].
                This reshapes to Dict[str, torch.Tensor] where torch.Tensor is of shape [batch_size].
                So, just parameters for a specific timestep
                """
                timestep_params[k] = conceptual_parameters[k][:, j]

            ss, sf, su, si, sb, timestep_out = self.timestep_shm(
                ss, sf, su, si, sb, timestep_params, x_conceptual_timestep, device
            )

            # Store time evolution of the internal states
            states["ss"][:, j] = ss
            states["sf"][:, j] = sf
            states["su"][:, j] = su
            states["si"][:, j] = si
            states["sb"][:, j] = sb
            out[:, j, 0] = timestep_out

        return {"y_hat": out, "parameters": timestep_params, "internal_states": states}

    @property
    def initial_states(self):
        return {"ss": 0.0, "sf": 1.0, "su": 5.0, "si": 10.0, "sb": 15.0}

    @property
    def parameter_ranges(self):
        return {
            "dd": [0.0, 10.0],
            "f_thr": [10.0, 60.0],
            "sumax": [20.0, 700.0],
            "beta": [1.0, 6.0],
            "perc": [0.0, 1.0],
            "kf": [1.0, 20.0],
            "ki": [1.0, 100.0],
            "kb": [10.0, 1000.0],
        }

    def initialize_states(self, batch_size, device):
        # initialize reservoirs
        ss = self.initial_states["ss"] * torch.ones(size=(batch_size,), device=device, dtype=torch.float32)
        sf = self.initial_states["sf"] * torch.ones(size=(batch_size,), device=device, dtype=torch.float32)
        su = self.initial_states["su"] * torch.ones(size=(batch_size,), device=device, dtype=torch.float32)
        si = self.initial_states["si"] * torch.ones(size=(batch_size,), device=device, dtype=torch.float32)
        sb = self.initial_states["sb"] * torch.ones(size=(batch_size,), device=device, dtype=torch.float32)

        return ss, sf, su, si, sb

    def timestep_shm(self, ss, sf, su, si, sb, timestep_params, x_conceptual_timestep, device):
        """
        ss, sf, su, si, sb are all reservoirs in SHM. They should be torch tensors of size batch_size.
        timestep_params is a dict of params. It should NOT have a time dimension.
        """
        # auxiliary vectors
        t_mean = (x_conceptual_timestep[:, 2] + x_conceptual_timestep[:, 3]) / 2
        temp_mask = t_mean < 0
        snow_melt = t_mean * timestep_params["dd"]
        snow_melt[temp_mask] = torch.zeros_like(snow_melt[temp_mask])
        klu = torch.tensor(0.90, device=device, dtype=torch.float32)

        # liquid precipitation:
        liquid_p = x_conceptual_timestep[:, 0].clone()
        liquid_p[temp_mask] = torch.zeros_like(liquid_p[temp_mask])

        # solid precipitation (snow):
        snow = x_conceptual_timestep[:, 0].clone()
        snow[~temp_mask] = torch.zeros_like(snow[~temp_mask])

        # permanent wilting point use in ET:
        pwp = 0.8 * timestep_params["sumax"]
        # pwp = torch.tensor(0.8, dtype=torch.float32, device=x_conceptual.device) * parameters["sumax"]

        # Snow module --------------------------
        qs_out = torch.minimum(ss, snow_melt)
        ss = ss - qs_out + snow
        qsp_out = qs_out + liquid_p

        # Split snowmelt+rainfall into inflow to fastflow reservoir and unsaturated reservoir ------
        qf_in = torch.maximum(torch.tensor(0.0), qsp_out - timestep_params["f_thr"])
        qu_in = torch.minimum(qsp_out, timestep_params["f_thr"])

        # Fastflow module ----------------------
        sf = sf + qf_in
        qf_out = sf / timestep_params["kf"]
        sf = sf - qf_out

        # Unsaturated zone----------------------
        psi = (su / timestep_params["sumax"]) ** timestep_params["beta"]
        su_temp = su + qu_in * (1 - psi)
        su = torch.minimum(su_temp, timestep_params["sumax"])
        qu_out = qu_in * psi + torch.maximum(torch.tensor(0.0), su_temp - timestep_params["sumax"])  # [mm]

        # Evapotranspiration -------------------
        ktetha = su / timestep_params["sumax"]
        et_mask = su <= pwp
        ktetha[~et_mask] = torch.ones_like(ktetha[~et_mask])
        ret = x_conceptual_timestep[:, 1] * klu * ktetha  # [mm]
        su = torch.maximum(torch.tensor(0.0), su - ret)  # [mm]

        # Interflow reservoir ------------------
        qi_in = qu_out * timestep_params["perc"]  # [mm]
        si = si + qi_in  # [mm]
        qi_out = si / timestep_params["ki"]  # [mm]
        si = si - qi_out  # [mm]

        # Baseflow reservoir -------------------
        qb_in = qu_out * (1.0 - timestep_params["perc"])  # [mm]
        sb = sb + qb_in  # [mm]
        qb_out = sb / timestep_params["kb"]  # [mm]
        sb = sb - qb_out

        # total outflow
        timestep_out = qf_out + qi_out + qb_out  # [mm]

        return ss, sf, su, si, sb, timestep_out
