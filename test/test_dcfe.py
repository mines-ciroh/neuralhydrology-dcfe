"""Test for checking forward process of CFE model matches reference implementation"""

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from neuralhydrology.modelzoo.cfe_modules.cfe_dataclasses import (
    GroundwaterStates,
    RoutingInfo,
    SoilConfig,
    SoilStates,
    get_constants,
)
from neuralhydrology.modelzoo.cfe_modules.get_default_params import get_default_params
from neuralhydrology.modelzoo.cfe_modules.timestep_cfe import timestep_cfe
from test import Fixture


def test_cfe(get_config: Fixture[Callable[[str], dict]]):
    config = get_config("cfe")

    # we only need to test for a single data set, input/output setting and model specifications
    config.update_config({"data_dir": config.data_dir / "cfe_data"})

    test_data_dir = Path(config.data_dir)
    device = config.device
    batch_size = 2

    forcingFile = pd.read_csv(test_data_dir / "cat58_01Dec2015.csv")
    forcingsTest = torch.zeros(batch_size, forcingFile.shape[0], 3)  # [ "batch_size", time_step, num_forcings]

    forcingsTest[:, :, 0] = torch.tensor(forcingFile.loc[:, "precip_rate"]) * 1000 * 3600  # m/s to mm/hr
    forcingsTest[:, :, 1] = torch.tensor(forcingFile.loc[:, "TMP_2maboveground"]) - 273.15  # K to C
    forcingsTest[:, :, 2] = torch.tensor(forcingFile.loc[:, "DSWRF_surface"])  # same unit as camels

    # empty vector to store output
    Discharge = torch.zeros(forcingsTest.shape[1], 8)

    cfe_features = {
        "catchment_area_km2": 15.617167355002097 * torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size),
        "refkdt": 3.8266861353378374 * torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size),
        "max_gw_storage": 16 * torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size),
        "Cgw": 0.01 * torch.tensor(1.0).repeat(batch_size),
        "expon": 6.0 * torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size),
        "alpha_fc": 0.33 * torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size),
        "K_nash": 0.03 * torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size),
        "K_lf": 0.01 * torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size),
        "nash_storage": torch.zeros((batch_size, 2), dtype=torch.float32, device=device),
        "giuh_ordinates": torch.tensor(
            [[0.1, 0.35, 0.2, 0.14, 0.1, 0.06, 0.05], [0.1, 0.35, 0.2, 0.14, 0.1, 0.06, 0.05]], dtype=torch.float32, device=device
        ),
        "depth": 2.0
        * torch.tensor(1.0, dtype=torch.float32, device=device).repeat(
            batch_size
        ),  # not sure where they got these values, they don't match CAMELS, [m]
        "bb": 4.05
        * torch.tensor(1.0, dtype=torch.float32, device=device).repeat(
            batch_size
        ),  # exponent on Clapp-Hornberger function, part of calibration
        "satdk": 0.00000338 * torch.tensor(1.0).repeat(batch_size),
        "satpsi": 0.355 * torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size),
        "slop": 1
        * torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size),  # slope coefficient, part of calibration
        "smcmax": 0.439
        * torch.tensor(1.0, dtype=torch.float32, device=device).repeat(
            batch_size
        ),  # maximum soil moisture content [m3/m3], part of calibration
        "wltsmc": 0.066 * torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size),
        "D": 2.0 * torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size),
        "mult": 1000.0 * torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size),
    }

    cfe_params = get_default_params(config, cfe_features, device)
    constants = get_constants(config.dcfe_hourly)

    # initialize model states/reservoirs.
    gw_reservoir = GroundwaterStates(device=device, batch_size=batch_size, cfe_params=cfe_params)
    soil_config = SoilConfig(cfe_params=cfe_params, device=device, batch_size=batch_size, constants=constants)
    soil_reservoir = SoilStates(
        device=device,
        batch_size=batch_size,
        cfe_params=cfe_params,
        soil_config=soil_config,
        constants=constants,
    )
    routing_info = RoutingInfo(device=device, batch_size=batch_size, cfe_params=cfe_params)

    # T-Shirt specific settings
    gw_reservoir.storage_m = gw_reservoir.storage_max_m * 0.03125
    soil_reservoir.storage_m = 0.667 * torch.tensor(1.0, dtype=torch.float32, device=device).repeat(batch_size)

    for i in range(forcingsTest.shape[1]):
        cfe_params, gw_reservoir, soil_reservoir, routing_info, flux = timestep_cfe(
            x_conceptual_timestep=forcingsTest[:, i, :],
            cfe_params=cfe_params,
            timestep_params=None,
            gw_reservoir=gw_reservoir,
            soil_reservoir=soil_reservoir,
            soil_config=soil_config,
            routing_info=routing_info,
            constants=constants,
        )

        Discharge[i, 0] = flux.Qout_m[0]  # total discharge
        Discharge[i, 1] = flux.giuh_runoff_m[0]
        Discharge[i, 2] = flux.nash_lateral_runoff_m[0]
        Discharge[i, 3] = flux.from_deep_gw_to_chan_m[0]
        Discharge[i, 4] = flux.surface_runoff_depth_m[0]
        Discharge[i, 5] = flux.actual_et_m_per_timestep[0]  # total ET
        Discharge[i, 6] = soil_reservoir.storage_m[0]  # soil water storage
        Discharge[i, 7] = gw_reservoir.storage_m[0]  # gw storage

    # grab observations from actual CFE
    compareFile = pd.read_csv(test_data_dir / "cat58_test_compare.csv")

    tot_discharge = compareFile["Total Discharge"]
    tot_discharge_sim = (
        Discharge[:, 0].numpy()
        * cfe_params.basin_characteristics.catchment_area_km2[0].numpy()
        * 1000000.0
        / constants["time"]["step_size"]
    )
    rmse_tot_discharge = np.sqrt(((tot_discharge[490:550] - tot_discharge_sim[490:550]) ** 2).mean())
    # check for consistency in model outputs
    assert rmse_tot_discharge < 1.2

    flow = compareFile["Flow"]
    flow_sim = Discharge[:, 0].numpy()
    rmse_flow = np.sqrt(((flow[490:550] - flow_sim[490:550]) ** 2).mean())
    assert rmse_flow < 3e-4

    direct_runoff = compareFile["Direct Runoff"]
    direct_runoff_sim = Discharge[:, 4].numpy()
    rmse_direct_runoff = np.sqrt(((direct_runoff[490:550] - direct_runoff_sim[490:550]) ** 2).mean())
    assert rmse_direct_runoff < 1e-3

    lateral_flow = compareFile["Lateral Flow"]
    lateral_flow_sim = Discharge[:, 2].numpy()
    rmse_lateral_flow = np.sqrt(((lateral_flow[490:550] - lateral_flow_sim[490:550]) ** 2).mean())
    assert rmse_lateral_flow < 3e-4

    base_flow = compareFile["Base Flow"]
    base_flow_sim = Discharge[:, 3].numpy()
    rmse_base_flow = np.sqrt(((base_flow[490:550] - base_flow_sim[490:550]) ** 2).mean())
    assert rmse_base_flow < 5e-4

    giuh_runoff = compareFile["GIUH Runoff"]
    giuh_runoff_sim = Discharge[:, 1].numpy()
    rmse_giuh_runoff = np.sqrt(((giuh_runoff[490:550] - giuh_runoff_sim[490:550]) ** 2).mean())
    assert rmse_giuh_runoff < 3e-4
