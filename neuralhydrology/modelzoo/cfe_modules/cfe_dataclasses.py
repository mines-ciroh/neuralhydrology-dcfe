from dataclasses import dataclass

import torch

# To Discuss:
# 1. Are the types correct.
# 2. Note that hourly, dcfe_soil_scheme, and dcfe_partition_scheme are now in CFEParams.
# 3. Should we declare default values for any of these parameters?


@dataclass
class SoilParams:
    depth: float
    bb: float
    satdk: float
    satpsi: float
    slop: float
    smcmax: float
    wltsmc: float
    D: float
    mult: float


@dataclass
class BasinCharacteristics:
    catchment_area_km2: float
    refkdt: float
    max_gw_storage: float
    expon: float
    Cgw: float
    alpha_fc: float
    K_nash: float
    K_lf: float
    nash_storage: int  # is this the right type?
    giuh_ordinates: list[int]


@dataclass
class CFEParams:
    soil_params: SoilParams
    basin_characteristics: BasinCharacteristics
    hourly: bool = False
    dcfe_soil_scheme: str = "classic"
    dcfe_partition_scheme: str = "Schaake"


class Flux:
    def __init__(self, device, batch_size):
        self.device = device
        self.batch_size = batch_size

        # initialize all fluxes to zero
        zero_tensor = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        self.surface_runoff_depth_m = zero_tensor.clone()
        self.infilt_excess_m = zero_tensor.clone()
        self.infiltration_depth_m = zero_tensor.clone()
        self.infilt_depth_m = zero_tensor.clone()
        self.actual_et_from_rain_m_per_timestep = zero_tensor.clone()
        self.actual_et_from_soil_m_per_timestep = zero_tensor.clone()
        self.actual_et_m_per_timestep = zero_tensor.clone()
        self.reduced_potential_et_m_per_timestep = zero_tensor.clone()
        self.primary_flux_m = zero_tensor.clone()
        self.secondary_flux_m = zero_tensor.clone()
        self.primary_flux_from_gw_m = zero_tensor.clone()
        self.secondary_flux_from_gw_m = zero_tensor.clone()
        self.giuh_runoff_m = zero_tensor.clone()
        self.nash_lateral_runoff_m = zero_tensor.clone()
        self.from_deep_gw_to_chan_m = zero_tensor.clone()
        self.tension_water_m = zero_tensor.clone()
        self.Qout_m = zero_tensor.clone()

    def reset_fluxes(self):
        for flux_name, _ in vars(self).items():
            if flux_name not in ["device", "batch_size"]:
                setattr(self, flux_name, torch.zeros(self.batch_size, dtype=torch.float32, device=self.device))


class GroundwaterStates:
    def __init__(self, device: str, batch_size: int, cfe_params: CFEParams):
        self.device = device
        self.batch_size = batch_size
        # suggest we add the logic from lines 145--150 of CFE_modules.py here to initialize groundwater states. Also, are there any additional states?


class SoilStates:
    def __init__(self, device: str, batch_size: int, cfe_params: CFEParams, soil_config: SoilConfig):
        self.device = device
        self.batch_size = batch_size
        self.soil_moisture_content = torch.zeros(batch_size, dtype=torch.float32, device=device)
        # suggest we add the logic from lines 155--160 of CFE_modules.py here to initialize soil states.


class SoilConfig:
    # @Ziyu let's discuss the function of this guy. Perhaps we can eliminate it?
    pass


@dataclass
class RoutingInfo:
    device: str = "cpu"
    batch_size: int
    num_ordinates: int
    num_reservoirs: int = 2  # do we even need to worry about non-default values?

    @property
    def runoff_queue_per_timestep(self) -> torch.Tensor:
        return torch.ones((self.batch_size, self.num_ordinates + 1), dtype=torch.float32, device=self.device)


TIME = {
    "step_size": 3600 if hourly else 3600 * 24,  # num of [seconds]
    "hrs": (3600 if hourly else 3600 * 24) / 3600,  # num of [hours]
    "days": ((3600 if hourly else 3600 * 24) / 3600) / 24,  # time step in [days]
}

PHYSICS_CONSTANTS = {
    "atm_press_Pa": 101325.0,  # [Pa]
    "unit_weight_water_N_per_m3": 9810.0,  # [N/m3]
}

CONSTANTS = {"time": TIME, "physics": PHYSICS_CONSTANTS}  # note we've moved cfe_scheme to cfe_params
