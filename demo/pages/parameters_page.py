# pages/parameters_page.py
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plots.internal_states_plots import plot_shm_internal_states, plot_dcfe_internal_states
from plots.parameters_plots import plot_shm_parameters, plot_dcfe_parameters

# TODO: move plotting functions to internal_states_plots.py and parameters_plots.py for better organization and readability
basins = [
    '02177000: CHATTOOGA RIVER NEAR CLAYTON, GA',
    '02349900: TURKEY CREEK AT BYROMVILLE, GA',
]

def page(shm_data: pd.DataFrame, dcfe_data: pd.DataFrame):
    with gr.Blocks() as layout:
        gr.Markdown("# Explore LSTM-predicted input parameters to the hybrid models (SHM and dCFE).")
        gr.Markdown("### The hybrid models are spun-up before the prediction period so that they can reach physically meaningful inputs.")
        gr.Markdown("- Spin-up periods are shown with dashed lines, prediction periods with solid lines.")
        
        with gr.Row():
            basin_dd = gr.Dropdown(choices=basins, label="Basin", value=basins[0])
            epoch_dd = gr.Dropdown(choices=list(range(1, 11)), label="Model Epoch", value=10)

        def update_shm_params(basin, epoch):
            return plot_shm_parameters(basin, epoch, shm_data)

        def update_dcfe_params(basin, epoch):
            return plot_dcfe_parameters(basin, epoch, dcfe_data)

        with gr.Tab("SHM"):
            shm_params_out = gr.Plot(label="SHM Parameters", value=plot_shm_parameters(basins[0], 10, shm_data))

        with gr.Tab("dCFE"):
            dcfe_params_out = gr.Plot(label="dCFE Parameters", value=plot_dcfe_parameters(basins[0], 10, dcfe_data))

        for inp in [basin_dd, epoch_dd]:
            inp.change(fn=update_shm_params,  inputs=[basin_dd, epoch_dd], outputs=shm_params_out)
            inp.change(fn=update_dcfe_params, inputs=[basin_dd, epoch_dd], outputs=dcfe_params_out)
    
    return layout