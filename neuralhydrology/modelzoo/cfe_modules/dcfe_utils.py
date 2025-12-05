import json
import re
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from neuralhydrology.datautils import utils
from neuralhydrology.utils.config import Config

def get_dcfe_params(cfg):
    """This function reads the config file, grabs HydroShare params needed for CFE, and returns a basin-index dataframe
    with the parameters for each basin in the training list.

    These parameters are a combo of default CFE parameters and calibrated parameters from the JSON files.

    Args:
        cfg: configuration
        device: ??

    Returns:
        df: dataframe inexed by basin ids with 2 columns, soil_params and basinCharacteristics, which are each dicts of parameters
    """
    cfe_param_dir = cfg.conceptual_dir / "CFE_Config_Cver_from_Luciana"
    calibrated_params_dir = cfg.conceptual_dir / "CFE_Calibrated_Config" / "runs"

    # --- get all the basin ids as strings ---
    basins = utils.load_basin_file(getattr(cfg, "train_basin_file"))

    col_keys = KEYS["soil"] + KEYS["basin_characteristics"]
    df = pd.DataFrame(index=basins, columns=col_keys)

    # --- iterate thru each basin id to get params ---
    for basin_id in basins:
        cfe_param_file_path = cfe_param_dir / (basin_id + "_bmi_config_cfe_pass.txt")
        with open(cfe_param_file_path, "r") as f:
            content = f.read()
        f.close()
        pattern = r"([\w.]+)\s*=\s*([0-9.eE+-]+(?:,\s*[0-9.eE+-]+)*)"
        matches_list = re.findall(pattern, content)
        matches = {}
        for match in matches_list:
            try:
                matches[match[0]] = float(match[1])
            except:
                matches[match[0]] = [float(x) for x in match[1].split(",")]

        soil_params = {}
        for k in KEYS["soil"]:
            match_key = "b" if k == "bb" else k
            if k == "D":
                soil_params[k] = torch.tensor(2.0, dtype=torch.float32)
            elif k == "mult":
                soil_params[k] = torch.tensor(1.0, dtype=torch.float32)
            else:
                soil_params[k] = torch.tensor(matches[f"soil_params.{match_key}"], dtype=torch.float32)

        basinCharacteristics = {}
        for k in KEYS["basin_characteristics"]:
            if k == "catchment_area_km2":
                basinCharacteristics[k] = torch.tensor(111.11, dtype=torch.float32)
            else:
                basinCharacteristics[k] = torch.tensor(matches[k], dtype=torch.float32)

        # --- Update parameters from JSON file in calibrated_params_dir ---
        # TODO: Maybe add an option to use the default parameters, instead of the calibrated ones?
        json_file_path = calibrated_params_dir / f"cat_{basin_id}_testrun_results.json"
        if json_file_path.exists():
            with open(json_file_path, "r") as file:
                data = json.load(file)
                best_params = data.get("best_params", {})

                # --- Update soil parameters ---
                for k in KEYS["soil"]:
                    temp_value = best_params.get(k, soil_params[k])
                    soil_params[k] = (
                        temp_value.clone().detach()
                        if isinstance(temp_value, torch.Tensor)
                        else torch.tensor(temp_value, dtype=torch.float32)
                    )

                # --- Update basin characteristics ---
                for k in KEYS["basin_characteristics"]:
                    lookup_key = "scheme" if k == "refkdt" else k
                    temp_value = best_params.get(lookup_key, basinCharacteristics[k])
                    basinCharacteristics[k] = (
                        temp_value.clone().detach()
                        if isinstance(temp_value, torch.Tensor)
                        else torch.tensor(temp_value, dtype=torch.float32)
                    )
        else:
            print(f"[warn] JSON file not found for basin {basin_id}, using default parameters.")

        # --- Unpack into DataFrame row ---
        for item in KEYS["soil"]:
            df.at[basin_id, item] = soil_params[item]

        for item in KEYS["basin_characteristics"]:
            df.at[basin_id, item] = basinCharacteristics[item]

    return df

KEYS = {
    "basin_characteristics": [
        "catchment_area_km2",
        "refkdt",
        "max_gw_storage",
        "expon",
        "Cgw",
        "alpha_fc",
        "K_nash",
        "K_lf",
        "nash_storage",
        "giuh_ordinates",
    ],
    "soil": [
        "depth",
        "bb",
        "satdk",
        "satpsi",
        "slop",
        "smcmax",
        "wltsmc",
        "D",
        "mult",
    ],
    "study_calibrated_params": [
        "bb",
        "smcmax",
        "satdk",
        "slop",
        "max_gw_storage",
        "expon",
        "Cgw",
        "K_lf",
        "K_nash",
        "refkdt",
    ],
}