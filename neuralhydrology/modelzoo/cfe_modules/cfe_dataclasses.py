from dataclasses import dataclass
from typing import Any, Dict

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
    nash_storage: list[int]  # is this the right type?
    giuh_ordinates: list[int]


class CFEParams:
    def __init__(
        self,
        soil_params: SoilParams,
        basin_characteristics: BasinCharacteristics,
        hourly: bool = False,
        dcfe_soil_scheme: str = "classic",
        dcfe_partition_scheme: str = "Schaake",
    ):
        self.soil_params = soil_params
        self.basin_characteristics = basin_characteristics
        self.hourly = hourly
        self.dcfe_soil_scheme = dcfe_soil_scheme
        self.dcfe_partition_scheme = dcfe_partition_scheme

    def update(self, timestep_params: torch.Tensor, time_step: int):
        """
        This mimics the functionality of timestep_basin_constants in the original CFE_modules.py.
        """
        self.soil_params.bb = timestep_params["bb"][:, time_step]
        self.soil_params.satdk = timestep_params["satdk"][:, time_step]
        self.soil_params.slop = timestep_params["slop"][:, time_step]
        self.soil_params.smcmax = timestep_params["smcmax"][:, time_step]
        self.soil_params.satpsi = timestep_params["satpsi"][:, time_step]
        self.basin_characteristics.Cgw = timestep_params["Cgw"][:, time_step]
        self.basin_characteristics.max_gw_storage = timestep_params["max_gw_storage"][:, time_step]
        self.basin_characteristics.expon = timestep_params["expon"][:, time_step]
        self.basin_characteristics.K_lf = timestep_params["K_lf"][:, time_step]
        self.basin_characteristics.K_nash = timestep_params["K_nash"][:, time_step]


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
        self.timestep_rainfall_input_m = zero_tensor.clone()
        self.potential_et_m_per_timestep = zero_tensor.clone()
        self.Qout_m = zero_tensor.clone()

    def reset_fluxes(self):
        for flux_name, _ in vars(self).items():
            if flux_name not in ["device", "batch_size"]:
                setattr(self, flux_name, torch.zeros(self.batch_size, dtype=torch.float32, device=self.device))


class GroundwaterStates:
    def __init__(self, device: str, batch_size: int, cfe_params: CFEParams):
        self.device = device
        self.batch_size = batch_size
        ones_tensor = torch.ones(batch_size, dtype=torch.float32, device=device)
        self.storage_m = 0.05 * ones_tensor.clone()
        self.storage_max_m = cfe_params.basin_characteristics.max_gw_storage
        self.coeff_primary = cfe_params.basin_characteristics.Cgw
        self.exponent_primary = cfe_params.basin_characteristics.expon
        assert self.storage_max_m.shape[0] == self.batch_size
        assert self.coeff_primary.shape[0] == self.batch_size
        assert self.exponent_primary.shape[0] == self.batch_size


class SoilConfig:
    def __init__(self, cfe_params: CFEParams, device: str, batch_size: int, constants: Dict[str, Any]):
        self.cfe_params = cfe_params
        self.device = device
        self.batch_size = batch_size
        self.constants = constants

        # Compute
        trigger_z_m = 0.5 * torch.ones(batch_size, dtype=torch.float32, device=device)  # assuming uniform depth of 0.5 m
        field_capacity_atm_press_fraction = cfe_params.basin_characteristics.alpha_fc
        # Soil outflux calculation, Eq. 3
        H_water_table_m = (
            field_capacity_atm_press_fraction
            * constants["physics"]["atm_press_Pa"]
            / constants["physics"]["unit_weight_water_N_per_m3"]
        )
        Omega = H_water_table_m - trigger_z_m

        # upper & lower limit of the integral in Eq. 4
        lower_lim = torch.pow(Omega, (1.0 - 1.0 / cfe_params.soil_params.bb)) / (1.0 - 1.0 / cfe_params.soil_params.bb)
        upper_lim = torch.pow(Omega + cfe_params.soil_params.D, (1.0 - 1.0 / cfe_params.soil_params.bb)) / (
            1.0 - 1.0 / cfe_params.soil_params.bb
        )

        # integral & power term in Eq. 4 and 5
        storage_thresh_pow_term = torch.pow(1.0 / cfe_params.soil_params.satpsi, (-1.0 / cfe_params.soil_params.bb))
        lim_diff = upper_lim - lower_lim

        # FINALIZE
        self.field_capacity_storage_threshold_m = cfe_params.soil_params.smcmax * storage_thresh_pow_term * lim_diff
        self.lateral_flow_threshold_storage_m = self.field_capacity_storage_threshold_m.clone()


class SoilStates:  # new soil_reservoir class
    def __init__(self, device: str, batch_size: int, cfe_params: CFEParams, soil_config: SoilConfig, constants: Dict[str, Any]):
        self.device = device
        self.batch_size = batch_size
        self.soil_moisture_content = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self.wilting_point_m = cfe_params.soil_params.wltsmc * cfe_params.soil_params.D
        self.storage_max_m = cfe_params.soil_params.smcmax * cfe_params.soil_params.D
        self.exponent_primary = 1.0  # Why hardcoded?
        self.storage_threshold_primary_m = soil_config.field_capacity_threshold_primary_m
        self.coeff_primary = cfe_params.soil_params.satdk * cfe_params.slop * constants.time.step_size
        self.coeff_secondary = cfe_params.basin_characteristics.K_lf
        self.exponent_secondary = 1.0
        self.storage_threshold_secondary_m = soil_config.lateral_flow_threshold_storage_m
        self.storage_m = 0.05 * torch.ones(batch_size, dtype=torch.float32, device=device)  # initialize soil storage
        self.storage_deficit_m = cfe_params.soil_params.smcmax * cfe_params.soil_params.D - self.storage_m
        self.Schaake_adjusted_magic_constant_by_soil_type = (
            cfe_params.basin_characteristics.refkdt * cfe_params.soil_params.satdk / 2.0e-6
        )

    def update(self, cfe_params: CFEParams):
        self.storage_deficit_m = cfe_params.soil_params.smcmax * cfe_params.soil_params.D - self.storage_m
        self.Schaake_adjusted_magic_constant_by_soil_type = (
            cfe_params.basin_characteristics.refkdt * cfe_params.soil_params.satdk / 2.0e-6
        )


class RoutingInfo:
    def __init__(self, device: str, batch_size: int, cfe_params: CFEParams):
        self.device = device
        self.batch_size = batch_size
        self.num_ordinates = cfe_params.basin_characteristics.giuh_ordinates.shape[1]
        self.num_reservoirs = cfe_params.basin_characteristics.nash_storage.shape[1]
        self.runoff_queue_m_per_timestep = torch.ones((batch_size, self.num_ordinates + 1), dtype=torch.float32, device=device)

    @property
    def runoff_queue_per_timestep(self) -> torch.Tensor:
        return self.runoff_queue_m_per_timestep


PARAMETER_RANGES = {
    "satdk": [0.0, 0.000726],  # Saturated hydraulic conductivity [m/hr]
    "Cgw": [0.0000018, 0.0018],  # Primary groundwater reservoir constant [m/hr]
    "bb": [0, 21.94],  # exponent on Clapp-Hornberg functin [-]
    "smcmax": [0.20554, 1],  # Max soil moisture content [m3/hr3]
    "slop": [0, 1],  # slope coefficient [-]
    "max_gw_storage": [0.01, 0.25],  # [m]
    "expon": [1, 8],  # A primary groundwater nonlinear reservoir exponential constant [-]
    "K_lf": [0, 1],  # Lateral flow coefficient
    "K_nash": [0, 1],  # Nash cascade discharge coefficient
    "satpsi": [0.05, 0.95],
}

INITIAL_STATES = {
    "gw_reservoir_storage_m": 0.5,
    "soil_reservoir_storage_m": 0.6,
    "first_nash_storage": 0.0,
}


def get_constants(hourly: bool):
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
    return CONSTANTS
