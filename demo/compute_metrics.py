import numpy as np
import pandas as pd
import xarray as xr
from neuralhydrology.evaluation.metrics import nse, rmse, kge
from load_data import load_shm_data, load_lstm_data, load_dcfe_data

def compute_metrics_all_models(basin: str, epoch: int, shm_data: pd.DataFrame, dcfe_data: pd.DataFrame, lstm_data: dict) -> pd.DataFrame:
    # TODO: There's something wrong with our data structures when passed through NH's metrics function
    # I might need to change the data to DataArrays.
    basin_id  = basin.split(":")[0]
    epoch_key = f"epoch{epoch}"
    
    model_names = ["SHM Dynamic", "SHM Oracle", "SHM Operational", "dCFE Dynamic", "dCFE Oracle", "dCFE Operational", "LSTM"]
    
    obs                  = shm_data["shm_dynamic"][epoch_key]["predictions"][basin_id]['1D']['xr']['QObs(mm/d)_obs'].squeeze()
    
    shm_sim_dynamic      = shm_data["shm_dynamic"][epoch_key]["predictions"][basin_id]['1D']['xr']['QObs(mm/d)_sim'].squeeze()
    shm_sim_oracle       = shm_data["shm_oracle"][epoch_key]["predictions"][basin_id]['1D']['xr']['QObs(mm/d)_sim'].squeeze()
    shm_sim_operational  = shm_data["shm_operational"][epoch_key]["predictions"][basin_id]['1D']['xr']['QObs(mm/d)_sim'].squeeze()
    
    dcfe_sim_dynamic     = dcfe_data["dcfe_dynamic"][epoch_key]["predictions"][basin_id]['1D']['xr']['QObs(mm/d)_sim'].squeeze()
    dcfe_sim_oracle      = dcfe_data["dcfe_oracle"][epoch_key]["predictions"][basin_id]['1D']['xr']['QObs(mm/d)_sim'].squeeze()
    dcfe_sim_operational = dcfe_data["dcfe_operational"][epoch_key]["predictions"][basin_id]['1D']['xr']['QObs(mm/d)_sim'].squeeze()
    
    # LSTM has a different data structure than hybrid models. Need to convert to DataArray and rename dimensions..
    lstm_unstacked = lstm_data[epoch_key][basin_id]['1D']['xr']['QObs(mm/d)_sim']
    lstm_sim = xr.DataArray(np.hstack([lstm_unstacked[0], lstm_unstacked[365], lstm_unstacked[365*2], lstm_unstacked[365*3], lstm_unstacked[365*4]])).squeeze()
    lstm_sim = lstm_sim.rename({"dim_0": "datetime"})
    
    models_to_evaluate = [shm_sim_dynamic, 
                          shm_sim_oracle, 
                          shm_sim_operational, 
                          dcfe_sim_dynamic, 
                          dcfe_sim_oracle, 
                          dcfe_sim_operational, 
                          lstm_sim]
    nse_scores = []
    rmse_scores = []
    kge_scores = []
    
    for idx, model in enumerate(models_to_evaluate):
        # four decimal places for readability in the GUI
        nse_scores.append(round(nse(obs, model), 4))
        rmse_scores.append(round(rmse(obs, model), 4))
        kge_scores.append(round(kge(obs, model), 4))

    return pd.DataFrame({
        "Model": model_names,
        "NSE":   nse_scores,
        "RMSE":   rmse_scores,
        "KGE":   kge_scores,
    })

if __name__ == "__main__":
    # Debugging metric computations
    shm_data = load_shm_data()
    lstm_data = load_lstm_data()
    dcfe_data = load_dcfe_data()
    basins = [
        '02177000: CHATTOOGA RIVER NEAR CLAYTON, GA',
        '02349900: TURKEY CREEK AT BYROMVILLE, GA',
    ]

    metrics_df = compute_metrics_all_models(basins[0], 10, shm_data, dcfe_data, lstm_data)
    print(metrics_df)