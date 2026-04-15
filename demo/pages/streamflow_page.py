#pages/streamflow_page.py
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from plots.streamflow_plot import plot_streamflow_results
from compute_metrics import compute_metrics_all_models
basins = [
    '02177000: CHATTOOGA RIVER NEAR CLAYTON, GA',
    '02349900: TURKEY CREEK AT BYROMVILLE, GA',
]

def page(shm_data: pd.DataFrame, dcfe_data: pd.DataFrame, lstm_data: dict) -> gr.Blocks:
    with gr.Blocks() as layout:
        gr.Markdown("# Compare dCFE, SHM, and LSTM (baseline) streamflow predictions across basins.")

        with gr.Row():
            basin_dd = gr.Dropdown(choices=basins, label="Basin", value=basins[0])
            epoch_dd = gr.Dropdown(choices=list(range(1, 11)), label="Model Epoch", value=10)

        plot_out = gr.Plot(
            label="Streamflow Comparison",
            value=plot_streamflow_results(basins[0], 10, shm_data, dcfe_data, lstm_data)
        )
        
        # Wrapper function to handle the static data injections
        def update_plot(basin, epoch):
            return plot_streamflow_results(basin, epoch, shm_data, dcfe_data, lstm_data)
        def update_metrics(basin, epoch):
            return compute_metrics_all_models(basin, epoch, shm_data, dcfe_data, lstm_data)

        metrics_out = gr.Dataframe(label="Performance Metrics", value=compute_metrics_all_models(basins[0], 10, shm_data, dcfe_data, lstm_data))

        for inp in [basin_dd, epoch_dd]:
            inp.change(
                fn=update_plot,
                inputs=[basin_dd, epoch_dd],
                outputs=plot_out
            )
            
            inp.change(fn=update_metrics, inputs=[basin_dd, epoch_dd], outputs=metrics_out)

    return layout