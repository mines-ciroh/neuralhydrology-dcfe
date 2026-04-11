"""
Defines the page for comparing streamflow predictions across models and basins.
"""
import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from theme import NORD, theme
from load_data import load_SHM_data, load_LSTM_data, load_dCFE_data

basins = [
    '02177000: CHATTOOGA RIVER NEAR CLAYTON, GA',
    '02349900: TURKEY CREEK AT BYROMVILLE, GA',
]

SHM_data = load_SHM_data()
dCFE_data = load_dCFE_data() # blank for now
LSTM_data = load_LSTM_data() # blank for now

def plot_streamflow_results(basin: str, epoch: int) -> go.Figure:
    basin_id  = basin.split(":")[0]
    epoch_key = f"epoch{epoch}"

    obs             = SHM_data["SHM_Dynamic"][epoch_key]["predictions"][basin_id]['1D']['xr']['QObs(mm/d)_obs']
    sim_dynamic     = SHM_data["SHM_Dynamic"][epoch_key]["predictions"][basin_id]['1D']['xr']['QObs(mm/d)_sim']
    sim_oracle      = SHM_data["SHM_Oracle"][epoch_key]["predictions"][basin_id]['1D']['xr']['QObs(mm/d)_sim']
    sim_operational = SHM_data["SHM_Operational"][epoch_key]["predictions"][basin_id]['1D']['xr']['QObs(mm/d)_sim']
    dates = obs['datetime']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=obs,             name="Observed",        line=dict(color=NORD["cyan"],   width=1.5)))
    fig.add_trace(go.Scatter(x=dates, y=sim_dynamic,     name="SHM Dynamic",     line=dict(color=NORD["green"],  width=1.2)))
    fig.add_trace(go.Scatter(x=dates, y=sim_oracle,      name="SHM Oracle",      line=dict(color=NORD["yellow"], width=1.2)))
    fig.add_trace(go.Scatter(x=dates, y=sim_operational, name="SHM Operational", line=dict(color=NORD["blue"],   width=1.2)))

    fig.update_layout(
        title=f"Basin {basin}",
        xaxis_title="Date",
        yaxis_title="Streamflow (mm/day)",
        autosize=True,
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor=NORD["border"], gridwidth=0.5, zeroline=False),
        yaxis=dict(gridcolor=NORD["border"], gridwidth=0.5, zeroline=False),
        margin=dict(l=60, r=20, t=50, b=50),
        legend=dict(
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
            bgcolor=NORD["panel2"],
            bordercolor=NORD["border"],
            borderwidth=1,
            font=dict(color=NORD["text"], size=13),
        ),
        font=dict(color=NORD["text"], size=13),
    )
    return fig


def get_metrics(basin: str, epoch: int) -> pd.DataFrame:
    # TODO: compute from real run outputs
    return pd.DataFrame({
        "Model": ["dCFE", "SHM", "LSTM"],
        "NSE":   [None, None, None],
        "MSE":   [None, None, None],
    })


with gr.Blocks() as page:
    gr.Markdown("## Streamflow Comparison")
    gr.Markdown("Compare dCFE, SHM, and LSTM (baseline) predictions across basins.")

    with gr.Row():
        basin_dd = gr.Dropdown(choices=basins, label="Basin", value=basins[0])
        epoch_dd = gr.Dropdown(choices=list(range(1, 11)), label="Model Epoch", value=10)

    plot_out    = gr.Plot(label="Streamflow Comparison", value=plot_streamflow_results(basins[0], 10))
    metrics_out = gr.Dataframe(label="Performance Metrics (TODO)")

    for inp in [basin_dd, epoch_dd]:
        inp.change(fn=plot_streamflow_results, inputs=[basin_dd, epoch_dd], outputs=plot_out)
        inp.change(fn=get_metrics,             inputs=[basin_dd, epoch_dd], outputs=metrics_out)