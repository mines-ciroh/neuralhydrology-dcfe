import numpy as np
import pandas as pd
import plotly.graph_objects as go

from load_data import load_shm_data, load_lstm_data, load_dcfe_data

def plot_streamflow_results(basin: str,
                            epoch: int,
                            shm_data:pd.DataFrame,
                            dcfe_data:pd.DataFrame,
                            lstm_data:dict) -> go.Figure:
    basin_id  = basin.split(":")[0]
    epoch_key = f"epoch{epoch}"

    obs                  = shm_data["shm_dynamic"][epoch_key]["predictions"][basin_id]['1D']['xr']['QObs(mm/d)_obs']
    shm_sim_dynamic      = shm_data["shm_dynamic"][epoch_key]["predictions"][basin_id]['1D']['xr']['QObs(mm/d)_sim']
    shm_sim_oracle       = shm_data["shm_oracle"][epoch_key]["predictions"][basin_id]['1D']['xr']['QObs(mm/d)_sim']
    shm_sim_operational  = shm_data["shm_operational"][epoch_key]["predictions"][basin_id]['1D']['xr']['QObs(mm/d)_sim']
    dcfe_sim_dynamic     = dcfe_data["dcfe_dynamic"][epoch_key]["predictions"][basin_id]['1D']['xr']['QObs(mm/d)_sim']
    dcfe_sim_oracle      = dcfe_data["dcfe_oracle"][epoch_key]["predictions"][basin_id]['1D']['xr']['QObs(mm/d)_sim']
    dcfe_sim_operational = dcfe_data["dcfe_operational"][epoch_key]["predictions"][basin_id]['1D']['xr']['QObs(mm/d)_sim']

    # LSTM has a different data structure than hybrid models.
    lstm_unstacked = lstm_data[epoch_key][basin_id]['1D']['xr']['QObs(mm/d)_sim']
    lstm_sim = np.hstack([lstm_unstacked[0], lstm_unstacked[365], lstm_unstacked[365*2], lstm_unstacked[365*3], lstm_unstacked[365*4]])

    dates = obs['datetime']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=obs,                  name="Observed"))
    fig.add_trace(go.Scatter(x=dates, y=shm_sim_dynamic,      name="SHM Dynamic"))
    fig.add_trace(go.Scatter(x=dates, y=shm_sim_oracle,       name="SHM Oracle"))
    fig.add_trace(go.Scatter(x=dates, y=shm_sim_operational,  name="SHM Operational"))
    fig.add_trace(go.Scatter(x=dates, y=dcfe_sim_dynamic,     name="dCFE Dynamic"))
    fig.add_trace(go.Scatter(x=dates, y=dcfe_sim_oracle,      name="dCFE Oracle"))
    fig.add_trace(go.Scatter(x=dates, y=dcfe_sim_operational, name="dCFE Operational"))
    fig.add_trace(go.Scatter(x=dates, y=lstm_sim,             name="LSTM"))
    fig.update_layout(
        template="plotly_dark",
        title=f"Basin {basin}",
        xaxis_title="Date",
        yaxis_title="Streamflow (mm/day)",
        autosize=True,
        height=400,
        xaxis=dict(zeroline=False),
        yaxis=dict(zeroline=False),
        margin=dict(l=60, r=20, t=50, b=50),
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top"),
    )
    return fig