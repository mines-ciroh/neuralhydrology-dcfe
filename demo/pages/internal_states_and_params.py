# pages/internal_states_and_params.py
import gradio as gr
import plotly.graph_objects as go

from theme import NORD, theme
from load_data import load_SHM_data, load_dCFE_data

basins = [
    '02177000: CHATTOOGA RIVER NEAR CLAYTON, GA',
    '02349900: TURKEY CREEK AT BYROMVILLE, GA',
]

SHM_data  = load_SHM_data()
dCFE_data = load_dCFE_data()

def plot_SHM_internal_states(basin: str, epoch: int) -> go.Figure:
    basin_id  = basin.split(":")[0]
    epoch_key = f"epoch{epoch}"
    fig = go.Figure()
    # TODO
    return fig


def plot_SHM_parameters(basin: str, epoch: int) -> go.Figure:
    basin_id  = basin.split(":")[0]
    epoch_key = f"epoch{epoch}"
    fig = go.Figure()
    # TODO
    return fig


def plot_dCFE_internal_states(basin: str, epoch: int) -> go.Figure:
    basin_id  = basin.split(":")[0]
    epoch_key = f"epoch{epoch}"
    fig = go.Figure()
    # TODO
    return fig


def plot_dCFE_parameters(basin: str, epoch: int) -> go.Figure:
    basin_id  = basin.split(":")[0]
    epoch_key = f"epoch{epoch}"
    fig = go.Figure()
    # TODO
    return fig


with gr.Blocks() as page:
    gr.Markdown("## Internal States & Parameters")
    gr.Markdown("Explore internal storage states and LSTM-predicted input parameters to the given model.")

    with gr.Row():
        basin_dd = gr.Dropdown(choices=basins, label="Basin", value=basins[0])
        epoch_dd = gr.Dropdown(choices=list(range(1, 11)), label="Model Epoch", value=10)

    with gr.Tab("SHM"):
        shm_states_out = gr.Plot(label="SHM Internal States")
        shm_params_out = gr.Plot(label="SHM Parameters")

    with gr.Tab("dCFE"):
        dcfe_states_out = gr.Plot(label="dCFE Internal States")
        dcfe_params_out = gr.Plot(label="dCFE Parameters")

    for inp in [basin_dd, epoch_dd]:
        inp.change(fn=plot_SHM_internal_states, inputs=[basin_dd, epoch_dd], outputs=shm_states_out)
        inp.change(fn=plot_SHM_parameters,      inputs=[basin_dd, epoch_dd], outputs=shm_params_out)
        inp.change(fn=plot_dCFE_internal_states, inputs=[basin_dd, epoch_dd], outputs=dcfe_states_out)
        inp.change(fn=plot_dCFE_parameters,      inputs=[basin_dd, epoch_dd], outputs=dcfe_params_out)