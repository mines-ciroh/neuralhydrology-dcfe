from typing import Dict, Union
from neuralhydrology.modelzoo.cfe_modules import get_pet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint
from neuralhydrology.modelzoo.baseconceptualmodel import BaseConceptualModel
from neuralhydrology.utils.config import Config


class SHMNeuralODE(BaseConceptualModel):
    """Modified version of SHM [#]_ hydrological model with dynamic parameterization.

    The SHM receives the dynamic parameterization given by a deep learning model. This class has two properties which
    define the initial conditions of the internal states of the model (buckets) and the ranges in which the model
    parameters are allowed to vary during optimization.
    
    Conceptual parameters:
        - dd: snowmelt degree-day factor [mm/°C/day]
        - f_thr: threshold for fastflow reservoir [mm]
        - sumax: maximum storage of the unsaturated reservoir [mm]
        - beta: shape parameter [-]
        - perc: interflow-baseflow partitioning parameter [%]
        - kf: fastflow reservoir retention constant [day]
        - ki: interflow reservoir retention constant [day]
        - kb: baseflow reservoir retention constant [day]

    Parameters
    ----------
    cfg : Config
        The run configuration.

    References
    ----------
        [#] Ehret, U., van Pruijssen, R., Bortoli, M., Loritz, R., Azmi, E., and Zehe, E.: Adaptive clustering: reducing
        the computational costs of distributed (hydrological) modelling by exploiting time-variable similarity among
        model elements, Hydrology and Earth System Sciences, 24, 4389–4411, https://doi.org/10.5194/hess-24-4389-2020,
        2020.
        [#] Acuña Espinoza, E., Loritz, R., Álvarez Chaves, M., Bäuerle, N., and Ehret, U.: To Bucket or not to Bucket?
        Analyzing the performance and interpretability of hybrid hydrological models with dynamic parameterization,
        Hydrology and Earth System Sciences, 28, 2705–2719, https://doi.org/10.5194/hess-28-2705-2024, 2024.
    """

    def __init__(self, cfg: Config):
        super(SHMNeuralODE, self).__init__(cfg=cfg)
        
    def _compute_q(self, trajectory, conceptual_parameters, start_idx, end_idx):
        # trajectory: [time_steps, batch_size, 5]
        
        sf = trajectory[:, :, 1]   
        si = trajectory[:, :, 3]
        sb = trajectory[:, :, 4]
        
        # Slice the parameters to match the current time segment
        kf = conceptual_parameters["kf"][:, start_idx:end_idx]  
        ki = conceptual_parameters["ki"][:, start_idx:end_idx]
        kb = conceptual_parameters["kb"][:, start_idx:end_idx]
        
        # Now kf.T exactly matches the time_steps of sf
        qf_out = sf / kf.T
        qi_out = si / ki.T
        qb_out = sb / kb.T
        
        return qf_out + qi_out + qb_out
    
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
            In order of n_inputs:
                0: Precipitation [mm/day]
                1: PET [mm/day] (for CAMELS_GB); use SRAD [W/m2] for CAMELS_US and convert to PET outside the model
                2: Tmin [°C]
                3: Tmax [°C]

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
        states, q_out = self._initialize_information(conceptual_inputs=x_conceptual)

        # initialize model reservoirs
        ss, sf, su, si, sb = self.initialize_states(batch_size, device)
        
        # initialize torchdiffeq ODEFunc
        ode_func = self.ODEFunc(cfg=self.cfg,
                                    x_conceptual=x_conceptual,
                                    conceptual_parameters=conceptual_parameters,
                                    device=device)

        # SPIN UP PERIOD (No gradient tracking)
        with torch.no_grad():
            # spinup period
            t_spinup = torch.arange(start=0,
                                    end=self.cfg.spin_up_period,
                                    device=device,
                                    dtype=torch.float32)
            # initial reservoir states
            y0_spinup = torch.stack(tensors=[ss, sf, su, si, sb], dim=-1)  # [batch_size, 5]
            
            # time evolution of storage states
            trajectory_spinup = odeint(func=ode_func,
                                       y0=y0_spinup,
                                       t=t_spinup,
                                       rtol=1e-3,
                                       atol=1e-4)  # [spin_up_period, batch_size, 5]

            # Store spin-up states and spin-up q_out
            states["ss"][:, :self.cfg.spin_up_period] = trajectory_spinup[:, :, 0].T
            states["sf"][:, :self.cfg.spin_up_period] = trajectory_spinup[:, :, 1].T
            states["su"][:, :self.cfg.spin_up_period] = trajectory_spinup[:, :, 2].T
            states["si"][:, :self.cfg.spin_up_period] = trajectory_spinup[:, :, 3].T
            states["sb"][:, :self.cfg.spin_up_period] = trajectory_spinup[:, :, 4].T
            
            q_out[:, :self.cfg.spin_up_period, 0] = self._compute_q(trajectory=trajectory_spinup,
                                                                    conceptual_parameters=conceptual_parameters,
                                                                    start_idx=0,
                                                                    end_idx=self.cfg.spin_up_period).T

        # PREDICTION PERIOD
        y0_prediction = trajectory_spinup[-1].detach()  # [batch_size, 5]
        t_prediction = torch.arange(self.cfg.spin_up_period, lstm_out.shape[1], device=device, dtype=torch.float32)
        
        # When tracking gradients for training we should use adjoint method to ensure memory doesn't explode.
        trajectory_prediction = odeint_adjoint(func=ode_func,
                                       y0=y0_prediction,
                                       t=t_prediction,
                                       rtol=1e-3,
                                       atol=1e-4)  # [pred_period, batch_size, 5]

        # store prediction states
        states["ss"][:, self.cfg.spin_up_period:] = trajectory_prediction[:, :, 0].T
        states["sf"][:, self.cfg.spin_up_period:] = trajectory_prediction[:, :, 1].T
        states["su"][:, self.cfg.spin_up_period:] = trajectory_prediction[:, :, 2].T
        states["si"][:, self.cfg.spin_up_period:] = trajectory_prediction[:, :, 3].T
        states["sb"][:, self.cfg.spin_up_period:] = trajectory_prediction[:, :, 4].T
        q_out[:, self.cfg.spin_up_period:, 0] = self._compute_q(trajectory=trajectory_prediction,
                                                                conceptual_parameters=conceptual_parameters,
                                                                start_idx=self.cfg.spin_up_period,
                                                                end_idx=lstm_out.shape[1]).T
        
        return {"y_hat": q_out, "parameters": conceptual_parameters, "internal_states": states}
    
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

    class ODEFunc(nn.Module):
        def __init__(self, cfg: Config, x_conceptual, conceptual_parameters, device):
            super().__init__()
            # store forcings and parameters for lookup at arbitrary t
            self.x_conceptual = x_conceptual
            self.conceptual_parameters = conceptual_parameters
            self.device = device
            self.sharpness = 10.0
            self.cfg = cfg

        def _interp_conceptual_parameters(self, t_low, t_high, alpha):
                return {
                    k: (1 - alpha) * v[:, t_low] + alpha * v[:, t_high]
                    for k, v in self.conceptual_parameters.items()
                }
                
        def _snow_reservoir(self, ss_0, precip, T_mean, params_t):
                """Snow accumulation and melt (smoothed for ODE solvers).
                
                Returns
                -------
                dss : increment to snow reservoir
                qsp_out : total liquid input (snowmelt + rainfall) [mm/day]
                """
                cold = torch.sigmoid(-self.sharpness * T_mean)  # ~1 when T_mean < 0°C
                warm = 1 - cold                                  # ~1 when T_mean > 0°C

                liquid_p  = precip * warm
                snow      = precip * cold
                snow_melt = T_mean * params_t["dd"] * warm       # naturally 0 when cold

                # Smooth minimum identity: min(a, b) = a - max(0, a - b)
                # This replaces: torch.minimum(ss_0, snow_melt)
                qs_out  = ss_0 - F.softplus(ss_0 - snow_melt, beta=self.sharpness)
                
                dss     = snow - qs_out
                qsp_out = qs_out + liquid_p                      # total liquid input to soil

                return dss, qsp_out

        def _fastflow_reservoir(self, sf_0, qf_in, params_t):
            """Fastflow reservoir — linear decay.
            
            Returns
            -------
            dsf : increment to fastflow reservoir
            qf_out : fastflow outflow [mm/day]
            """
            sf_temp = sf_0 + qf_in
            qf_out  = sf_temp / params_t["kf"]
            dsf     = qf_in - qf_out

            return dsf, qf_out

        def _unsaturated_zone(self, su_0, qu_in, params_t):
            """Unsaturated zone with capacity cap and overflow (smoothed for ODE solvers).
            
            Returns
            -------
            su_after_cap : unsaturated storage after capacity cap [mm]
            qu_out : outflow from unsaturated zone [mm/day]
            """
            psi     = (su_0 / params_t["sumax"]) ** params_t["beta"]
            su_temp = su_0 + qu_in * (1 - psi)
            
            # Calculate overflow using softplus (smooth approximation of max(0, x))
            # This replaces: torch.maximum(0.0, su_temp - sumax)
            overflow = F.softplus(su_temp - params_t["sumax"], beta=self.sharpness)
            
            # Smooth minimum identity: min(a, b) = a - max(0, a - b)
            # This replaces: torch.minimum(su_temp, sumax)
            su_after_cap = su_temp - overflow
            
            # Outflow is the fractional outflow plus the smoothed overflow
            qu_out = qu_in * psi + overflow

            return su_after_cap, qu_out

        def _evapotranspiration(self, su_after_cap, su_0, T_mean, x_t, params_t):
            """Evapotranspiration with soft wilting point switch.
            
            Returns
            -------
            dsu : increment to unsaturated reservoir
            """
            permanent_wilting_point = 0.8 * params_t["sumax"]
            klu                     = torch.tensor(0.90, device=self.device, dtype=torch.float32)

            # soft switch replaces non-differentiable et_mask
            above_pwp = torch.sigmoid(self.sharpness * (su_after_cap - permanent_wilting_point))
            ktetha    = above_pwp + (1 - above_pwp) * (su_after_cap / params_t["sumax"])

            if self.cfg.dataset == "camels_us":
                pet = get_pet.daily_pet_jensen2016(T_avg=T_mean, S_rad=x_t[:, 1]).to(self.device)
            else:
                pet = x_t[:, 1]

            ret     = pet * klu * ktetha
            su_next = F.softplus(su_after_cap - ret, beta=self.sharpness)
            dsu     = su_next - su_0

            return dsu

        def _interflow_reservoir(self, si_0, qi_in, params_t):
            """Interflow reservoir — linear decay.
            
            Returns
            -------
            dsi : increment to interflow reservoir
            qi_out : interflow outflow [mm/day]
            """
            si_temp = si_0 + qi_in
            qi_out  = si_temp / params_t["ki"]
            dsi     = qi_in - qi_out

            return dsi, qi_out

        def _baseflow_reservoir(self, sb_0, qb_in, params_t):
            """Baseflow reservoir — linear decay.
            
            Returns
            -------
            dsb : increment to baseflow reservoir
            qb_out : baseflow outflow [mm/day]
            """
            sb_temp = sb_0 + qb_in
            qb_out  = sb_temp / params_t["kb"]
            dsb     = qb_in - qb_out

            return dsb, qb_out

        def forward(self, t, state):
            """Computes dS/dt for the SHM model at time t.

            Parameters
            ----------
            t : torch.Tensor
                Scalar time value provided by the ODE solver.
            state : torch.Tensor
                Current reservoir states of shape [batch_size, 5].
                Dimension order: [ss, sf, su, si, sb]

            Returns
            -------
            torch.Tensor
                Increments dS/dt of shape [batch_size, 5].
                Dimension order: [dss, dsf, dsu, dsi, dsb]
            """
            # Unpack state vector
            ss_0 = torch.clamp(state[:, 0], min=0.0)
            sf_0 = torch.clamp(state[:, 1], min=0.0)
            su_0 = torch.clamp(state[:, 2], min=0.0)
            si_0 = torch.clamp(state[:, 3], min=0.0)
            sb_0 = torch.clamp(state[:, 4], min=0.0)

            # Interpolate forcings and parameters at time t
            t_max  = self.x_conceptual.shape[1] - 1    # last valid index
            t_low  = min(int(t), t_max - 1)            # never index the last element as t_low
            t_high = t_low + 1                         # always valid since t_low <= T_max - 1
            alpha  = t - int(t)                        # fractional part always from original t

            x_t = (1 - alpha) * self.x_conceptual[:, t_low, :] + alpha * self.x_conceptual[:, t_high, :]
            params_t = self._interp_conceptual_parameters(t_low, t_high, alpha)

            # Unpack forcings
            precip = x_t[:, 0]
            T_mean = (x_t[:, 2] + x_t[:, 3]) / 2

            # Process all derivatives
            dss, qsp_out                = self._snow_reservoir(ss_0, precip, T_mean, params_t)
            qf_in                       = F.softplus(qsp_out - params_t["f_thr"], beta=self.sharpness)
            qu_in                       = qsp_out - F.softplus(qsp_out - params_t["f_thr"], beta=self.sharpness)
            dsf, _                      = self._fastflow_reservoir(sf_0, qf_in, params_t)
            su_after_cap, qu_out        = self._unsaturated_zone(su_0, qu_in, params_t)
            dsu                         = self._evapotranspiration(su_after_cap, su_0, T_mean, x_t, params_t)
            dsi, _                      = self._interflow_reservoir(si_0, qu_out * params_t["perc"], params_t)
            dsb, _                      = self._baseflow_reservoir(sb_0, qu_out * (1 - params_t["perc"]), params_t)

            return torch.stack([dss, dsf, dsu, dsi, dsb], dim=-1)