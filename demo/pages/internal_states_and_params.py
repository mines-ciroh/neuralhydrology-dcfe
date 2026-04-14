# pages/internal_states_and_params.py
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    
    all_output_dynamic     = SHM_data["SHM_Dynamic"][epoch_key]["all_output"]
    all_output_operational = SHM_data["SHM_Operational"][epoch_key]["all_output"]
    all_output_oracle      = SHM_data["SHM_Oracle"][epoch_key]["all_output"]

    dynamic     = all_output_dynamic[basin_id]     # basin level
    operational = all_output_operational[basin_id]
    oracle      = all_output_oracle[basin_id]
    
    state_dates = all_output_dynamic["datetime"] # just use one of the three since they should be the same
    state_keys  = list(dynamic["spinup_internal_states"].keys())

    fig = go.Figure()

    # One subplot row per state key
    fig = make_subplots(
        rows=len(state_keys), cols=1,
        shared_xaxes=True,
        subplot_titles=state_keys,
        vertical_spacing=0.05,
    )

    for i, state in enumerate(state_keys, start=1):
        show_legend = i == 1  # only show legend labels on first subplot

        # Dynamic
        fig.add_trace(go.Scatter(
            x=state_dates, y=dynamic["spinup_internal_states"][state],
            name="Dynamic Spinup", line=dict(color=NORD["blue"], dash="dash", width=1),
            opacity=0.5, legendgroup="dynamic", showlegend=show_legend,
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=state_dates, y=dynamic["prediction_internal_states"][state],
            name="Dynamic Prediction", line=dict(color=NORD["blue"], width=1.5),
            legendgroup="dynamic_pred", showlegend=show_legend,
        ), row=i, col=1)

        # Operational
        fig.add_trace(go.Scatter(
            x=state_dates, y=operational["spinup_internal_states"][state],
            name="Operational Spinup", line=dict(color=NORD["yellow"], dash="dash", width=1),
            opacity=0.5, legendgroup="operational", showlegend=show_legend,
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=state_dates, y=operational["prediction_internal_states"][state],
            name="Operational Prediction", line=dict(color=NORD["yellow"], width=1.5),
            legendgroup="operational_pred", showlegend=show_legend,
        ), row=i, col=1)

        # Oracle
        fig.add_trace(go.Scatter(
            x=state_dates, y=oracle["spinup_internal_states"][state],
            name="Oracle Spinup", line=dict(color=NORD["green"], dash="dash", width=1),
            opacity=0.5, legendgroup="oracle", showlegend=show_legend,
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=state_dates, y=oracle["prediction_internal_states"][state],
            name="Oracle Prediction", line=dict(color=NORD["green"], width=1.5),
            legendgroup="oracle_pred", showlegend=show_legend,
        ), row=i, col=1)

        # y-axis label per row
        fig.update_yaxes(
            title_text=f"{state} [mm]",
            gridcolor=NORD["border"], gridwidth=0.5, zeroline=False,
            row=i, col=1,
        )

    fig.update_xaxes(gridcolor=NORD["border"], gridwidth=0.5, zeroline=False)
    fig.update_layout(
        title=f"Basin {basin}",
        height=250 * len(state_keys),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=NORD["text"], size=13),
        legend=dict(
            orientation="h",
            x=0.5, y=-0.03,
            xanchor="center", yanchor="top",
            bgcolor=NORD["panel2"],
            bordercolor=NORD["border"],
            borderwidth=1,
            font=dict(color=NORD["text"], size=12),
        ),
        margin=dict(l=60, r=20, t=50, b=80),
    )

    return fig

def plot_SHM_parameters(basin: str, epoch: int) -> go.Figure:
    basin_id  = basin.split(":")[0]
    epoch_key = f"epoch{epoch}"

    all_output_dynamic     = SHM_data["SHM_Dynamic"][epoch_key]["all_output"]
    all_output_operational = SHM_data["SHM_Operational"][epoch_key]["all_output"]
    all_output_oracle      = SHM_data["SHM_Oracle"][epoch_key]["all_output"]

    dynamic     = all_output_dynamic[basin_id]     # basin level
    operational = all_output_operational[basin_id]
    oracle      = all_output_oracle[basin_id]

    param_dates     = all_output_dynamic["datetime"]
    param_keys      = list(dynamic["spinup_parameters"].keys())
    param_unit_keys = ['[mm/C/day]', '[mm]', '[mm]', '[-]', '[%]', '[day]', '[day]', '[day]']

    fig = make_subplots(
        rows=len(param_keys), cols=1,
        shared_xaxes=True,
        subplot_titles=param_keys,
        vertical_spacing=0.05,
    )

    for i, param in enumerate(param_keys, start=1):
        show_legend = i == 1
        unit = param_unit_keys[i - 1] if i - 1 < len(param_unit_keys) else ""

        # Dynamic
        # spinup period
        fig.add_trace(go.Scatter(
            x=param_dates, y=dynamic["spinup_parameters"][param],
            name="Dynamic Spinup", line=dict(color=NORD["blue"], dash="dash", width=1),
            opacity=0.5, legendgroup="dynamic", showlegend=show_legend,
        ), row=i, col=1)
        # prediction period
        fig.add_trace(go.Scatter(
            x=param_dates, y=dynamic["prediction_parameters"][param],
            name="Dynamic Prediction", line=dict(color=NORD["blue"], width=1.5),
            legendgroup="dynamic_pred", showlegend=show_legend,
        ), row=i, col=1)

        # Operational
        # spinup period
        fig.add_trace(go.Scatter(
            x=param_dates, y=operational["spinup_parameters"][param],
            name="Operational Spinup", line=dict(color=NORD["yellow"], dash="dash", width=1),
            opacity=0.5, legendgroup="operational", showlegend=show_legend,
        ), row=i, col=1)
        # prediction period
        fig.add_trace(go.Scatter(
            x=param_dates, y=operational["prediction_parameters"][param],
            name="Operational Prediction", line=dict(color=NORD["yellow"], width=1.5),
            legendgroup="operational_pred", showlegend=show_legend,
        ), row=i, col=1)

        # Oracle
        # spinup period
        fig.add_trace(go.Scatter(
            x=param_dates, y=oracle["spinup_parameters"][param],
            name="Oracle Spinup", line=dict(color=NORD["green"], dash="dash", width=1),
            opacity=0.5, legendgroup="oracle", showlegend=show_legend,
        ), row=i, col=1)
        # prediction period
        fig.add_trace(go.Scatter(
            x=param_dates, y=oracle["prediction_parameters"][param],
            name="Oracle Prediction", line=dict(color=NORD["green"], width=1.5),
            legendgroup="oracle_pred", showlegend=show_legend,
        ), row=i, col=1)

        fig.update_yaxes(
            title_text=f"{param} {unit}",
            gridcolor=NORD["border"], gridwidth=0.5, zeroline=False,
            row=i, col=1,
        )

    fig.update_xaxes(gridcolor=NORD["border"], gridwidth=0.5, zeroline=False)
    fig.update_layout(
        title=f"Basin {basin}",
        height=250 * len(param_keys),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=NORD["text"], size=13),
        legend=dict(
            orientation="h",
            x=0.5, y=-0.03,
            xanchor="center", yanchor="top",
            bgcolor=NORD["panel2"],
            bordercolor=NORD["border"],
            borderwidth=1,
            font=dict(color=NORD["text"], size=12),
        ),
        margin=dict(l=60, r=20, t=50, b=80),
    )

    return fig

def plot_dCFE_internal_states(basin: str, epoch: int) -> go.Figure:
    basin_id  = basin.split(":")[0]
    epoch_key = f"epoch{epoch}"

    all_output_dynamic     = dCFE_data["dCFE_Dynamic"][epoch_key]["all_output"]
    all_output_operational = dCFE_data["dCFE_Operational"][epoch_key]["all_output"]
    all_output_oracle      = dCFE_data["dCFE_Oracle"][epoch_key]["all_output"]

    dynamic     = all_output_dynamic[basin_id]
    operational = all_output_operational[basin_id]
    oracle      = all_output_oracle[basin_id]

    state_dates = all_output_dynamic["datetime"]
    state_keys  = list(dynamic["spinup_internal_states"].keys())

    fig = make_subplots(
        rows=len(state_keys), cols=1,
        shared_xaxes=True,
        subplot_titles=state_keys,
        vertical_spacing=0.05,
    )

    for i, state in enumerate(state_keys, start=1):
        show_legend = i == 1

        # Dynamic
        fig.add_trace(go.Scatter(
            x=state_dates, y=dynamic["spinup_internal_states"][state],
            name="Dynamic Spinup", line=dict(color=NORD["blue"], dash="dash", width=1),
            opacity=0.5, legendgroup="dynamic", showlegend=show_legend,
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=state_dates, y=dynamic["prediction_internal_states"][state],
            name="Dynamic Prediction", line=dict(color=NORD["blue"], width=1.5),
            legendgroup="dynamic_pred", showlegend=show_legend,
        ), row=i, col=1)

        # Oracle
        fig.add_trace(go.Scatter(
            x=state_dates, y=oracle["spinup_internal_states"][state],
            name="Oracle Spinup", line=dict(color=NORD["yellow"], dash="dash", width=1),
            legendgroup="oracle", showlegend=show_legend,
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=state_dates, y=oracle["prediction_internal_states"][state],
            name="Oracle Prediction", line=dict(color=NORD["yellow"], width=1.5),
            opacity=0.5, legendgroup="oracle_pred", showlegend=show_legend,
        ), row=i, col=1)

        # Operational
        fig.add_trace(go.Scatter(
            x=state_dates, y=operational["spinup_internal_states"][state],
            name="Operational Spinup", line=dict(color=NORD["green"], dash="dash", width=1),
            opacity=0.5, legendgroup="operational", showlegend=show_legend,
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=state_dates, y=operational["prediction_internal_states"][state],
            name="Operational Prediction", line=dict(color=NORD["green"], width=1.5),
            opacity=0.5, legendgroup="operational_pred", showlegend=show_legend,
        ), row=i, col=1)

        fig.update_yaxes(
            title_text=f"{state} [mm]",
            gridcolor=NORD["border"], gridwidth=0.5, zeroline=False,
            row=i, col=1,
        )

    fig.update_xaxes(gridcolor=NORD["border"], gridwidth=0.5, zeroline=False)
    fig.update_layout(
        title=f"Basin {basin}",
        height=250 * len(state_keys),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=NORD["text"], size=13),
        legend=dict(
            orientation="h",
            x=0.5, y=-0.03,
            xanchor="center", yanchor="top",
            bgcolor=NORD["panel2"],
            bordercolor=NORD["border"],
            borderwidth=1,
            font=dict(color=NORD["text"], size=12),
        ),
        margin=dict(l=60, r=20, t=50, b=80),
    )

    return fig


def plot_dCFE_parameters(basin: str, epoch: int) -> go.Figure:
    basin_id  = basin.split(":")[0]
    epoch_key = f"epoch{epoch}"

    all_output_dynamic     = dCFE_data["dCFE_Dynamic"][epoch_key]["all_output"]
    all_output_operational = dCFE_data["dCFE_Operational"][epoch_key]["all_output"]
    all_output_oracle      = dCFE_data["dCFE_Oracle"][epoch_key]["all_output"]

    dynamic     = all_output_dynamic[basin_id]
    operational = all_output_operational[basin_id]
    oracle      = all_output_oracle[basin_id]

    param_dates = all_output_dynamic["datetime"]
    param_keys  = list(dynamic["spinup_parameters"].keys())

    fig = make_subplots(
        rows=len(param_keys), cols=1,
        shared_xaxes=True,
        subplot_titles=param_keys,
        vertical_spacing=0.05,
    )

    for i, param in enumerate(param_keys, start=1):
        show_legend = i == 1

        # Dynamic
        fig.add_trace(go.Scatter(
            x=param_dates, y=dynamic["spinup_parameters"][param],
            name="Dynamic Spinup", line=dict(color=NORD["blue"], dash="dash", width=1),
            opacity=0.5, legendgroup="dynamic", showlegend=show_legend,
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=param_dates, y=dynamic["prediction_parameters"][param],
            name="Dynamic Prediction", line=dict(color=NORD["blue"], width=1.5),
            legendgroup="dynamic_pred", showlegend=show_legend,
        ), row=i, col=1)

        # Oracle
        fig.add_trace(go.Scatter(
            x=param_dates, y=oracle["spinup_parameters"][param],
            name="Oracle Spinup", line=dict(color=NORD["yellow"], dash="dash", width=1),
            legendgroup="oracle", showlegend=show_legend,
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=param_dates, y=oracle["prediction_parameters"][param],
            name="Oracle Prediction", line=dict(color=NORD["yellow"], width=1.5),
            opacity=0.5, legendgroup="oracle_pred", showlegend=show_legend,
        ), row=i, col=1)

        # Operational
        fig.add_trace(go.Scatter(
            x=param_dates, y=operational["spinup_parameters"][param],
            name="Operational Spinup", line=dict(color=NORD["green"], dash="dash", width=1),
            opacity=0.5, legendgroup="operational", showlegend=show_legend,
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=param_dates, y=operational["prediction_parameters"][param],
            name="Operational Prediction", line=dict(color=NORD["green"], width=1.5),
            opacity=0.5, legendgroup="operational_pred", showlegend=show_legend,
        ), row=i, col=1)

        fig.update_yaxes(
            title_text=param,
            gridcolor=NORD["border"], gridwidth=0.5, zeroline=False,
            row=i, col=1,
        )

    fig.update_xaxes(gridcolor=NORD["border"], gridwidth=0.5, zeroline=False)
    fig.update_layout(
        title=f"Basin {basin}",
        height=250 * len(param_keys),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=NORD["text"], size=13),
        legend=dict(
            orientation="h",
            x=0.5, y=-0.03,
            xanchor="center", yanchor="top",
            bgcolor=NORD["panel2"],
            bordercolor=NORD["border"],
            borderwidth=1,
            font=dict(color=NORD["text"], size=12),
        ),
        margin=dict(l=60, r=20, t=50, b=80),
    )

    return fig

with gr.Blocks() as page:
    gr.Markdown("## Internal States & Parameters")
    gr.Markdown("Explore internal storage states and LSTM-predicted input parameters to the given model.")

    with gr.Row():
        basin_dd = gr.Dropdown(choices=basins, label="Basin", value=basins[0])
        epoch_dd = gr.Dropdown(choices=list(range(1, 11)), label="Model Epoch", value=10)

    with gr.Tab("SHM"):
        gr.Markdown("### We spin-up the SHM model for a predefined period so that the internal states and parameters can reach physically meaningful values before the prediction period.")
        gr.Markdown("- Spin-up period is shown with dashed lines, prediction period with solid lines.")
        gr.Markdown("- Internal States: Basin storage states, used for streamflow predictions.")
        gr.Markdown("- Parameters: LSTM-predicted physical input parameters to the SHM model.")
        gr.Markdown("### Click the dropdown to expand and view SHM internal states and parameters over time.")
        
        with gr.Accordion("Internal States", open=False):
            shm_states_out = gr.Plot(label="SHM Internal States", value=plot_SHM_internal_states(basins[0], 10))
        with gr.Accordion("Parameters", open=False):
            shm_params_out = gr.Plot(label="SHM Parameters", value=plot_SHM_parameters(basins[0], 10))


    with gr.Tab("dCFE"):
        gr.Markdown("### We spin-up the dCFE model for a predefined period so that the internal states and parameters can reach physically meaningful values before the prediction period.")
        gr.Markdown("- Spin-up period is shown with dashed lines, prediction period with solid lines.")
        gr.Markdown("- Internal States: Basin storage states, used for streamflow predictions.")
        gr.Markdown("- Parameters: LSTM-predicted physical input parameters to the dCFE model.")
        gr.Markdown("### Click the dropdown to expand and view dCFE internal states and parameters over time.")
        
        with gr.Accordion("Internal States", open=False):
            dcfe_states_out = gr.Plot(label="dCFE Internal States", value=plot_dCFE_internal_states(basins[0], 10))
        with gr.Accordion("Parameters", open=False):
            dcfe_params_out = gr.Plot(label="dCFE Parameters", value=plot_dCFE_parameters(basins[0], 10))

    for inp in [basin_dd, epoch_dd]:
        inp.change(fn=plot_SHM_internal_states, inputs=[basin_dd, epoch_dd], outputs=shm_states_out)
        inp.change(fn=plot_SHM_parameters,      inputs=[basin_dd, epoch_dd], outputs=shm_params_out)
        inp.change(fn=plot_dCFE_internal_states, inputs=[basin_dd, epoch_dd], outputs=dcfe_states_out)
        inp.change(fn=plot_dCFE_parameters,      inputs=[basin_dd, epoch_dd], outputs=dcfe_params_out)