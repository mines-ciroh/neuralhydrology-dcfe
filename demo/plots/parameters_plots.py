import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {
    "dynamic":"#636EFA",  # blue
    "operational":"#EF553B",  # red
    "oracle":"#00CC96",  # green
}
SPINUP_OPACITY = 0.4

def plot_shm_parameters(basin: str, epoch: int, shm_data: pd.DataFrame) -> go.Figure:
    basin_id  = basin.split(":")[0]
    epoch_key = f"epoch{epoch}"

    all_output_dynamic     = shm_data["shm_dynamic"][epoch_key]["all_output"]
    all_output_operational = shm_data["shm_operational"][epoch_key]["all_output"]
    all_output_oracle      = shm_data["shm_oracle"][epoch_key]["all_output"]

    dynamic     = all_output_dynamic[basin_id]
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
        fig.add_trace(go.Scatter(
            x=param_dates, y=dynamic["spinup_parameters"][param],
            name="Dynamic Spinup", line=dict(color=COLORS["dynamic"], dash="dash", width=1),
            opacity=SPINUP_OPACITY, legendgroup="dynamic", showlegend=show_legend,
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=param_dates, y=dynamic["prediction_parameters"][param],
            name="Dynamic Prediction", line=dict(color=COLORS["dynamic"], width=1.5),
            legendgroup="dynamic_pred", showlegend=show_legend,
        ), row=i, col=1)

        # Operational
        fig.add_trace(go.Scatter(
            x=param_dates, y=operational["spinup_parameters"][param],
            name="Operational Spinup", line=dict(color=COLORS["operational"], dash="dash", width=1),
            opacity=SPINUP_OPACITY, legendgroup="operational", showlegend=show_legend,
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=param_dates, y=operational["prediction_parameters"][param],
            name="Operational Prediction", line=dict(color=COLORS["operational"], width=1.5),
            legendgroup="operational_pred", showlegend=show_legend,
        ), row=i, col=1)

        # Oracle
        fig.add_trace(go.Scatter(
            x=param_dates, y=oracle["spinup_parameters"][param],
            name="Oracle Spinup", line=dict(color=COLORS["oracle"], dash="dash", width=1),
            opacity=SPINUP_OPACITY, legendgroup="oracle", showlegend=show_legend,
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=param_dates, y=oracle["prediction_parameters"][param],
            name="Oracle Prediction", line=dict(color=COLORS["oracle"], width=1.5),
            legendgroup="oracle_pred", showlegend=show_legend,
        ), row=i, col=1)

        fig.update_yaxes(title_text=f"{param} {unit}", zeroline=False, row=i, col=1)

    fig.update_xaxes(zeroline=False)
    fig.update_layout(
        template="plotly_dark",
        title=f"Basin {basin}",
        height=250 * len(param_keys),
        legend=dict(orientation="h", x=0.5, y=-0.03, xanchor="center", yanchor="top"),
        margin=dict(l=60, r=20, t=50, b=80),
    )

    return fig

def plot_dcfe_parameters(basin: str, epoch: int, dcfe_data: pd.DataFrame) -> go.Figure:
    basin_id  = basin.split(":")[0]
    epoch_key = f"epoch{epoch}"

    all_output_dynamic     = dcfe_data["dcfe_dynamic"][epoch_key]["all_output"]
    all_output_operational = dcfe_data["dcfe_operational"][epoch_key]["all_output"]
    all_output_oracle      = dcfe_data["dcfe_oracle"][epoch_key]["all_output"]

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
            name="Dynamic Spinup", line=dict(color=COLORS["dynamic"], dash="dash", width=1),
            opacity=SPINUP_OPACITY, legendgroup="dynamic", showlegend=show_legend,
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=param_dates, y=dynamic["prediction_parameters"][param],
            name="Dynamic Prediction", line=dict(color=COLORS["dynamic"], width=1.5),
            legendgroup="dynamic_pred", showlegend=show_legend,
        ), row=i, col=1)

        # Oracle
        fig.add_trace(go.Scatter(
            x=param_dates, y=oracle["spinup_parameters"][param],
            name="Oracle Spinup", line=dict(color=COLORS["oracle"], dash="dash", width=1),
            opacity=SPINUP_OPACITY, legendgroup="oracle", showlegend=show_legend,
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=param_dates, y=oracle["prediction_parameters"][param],
            name="Oracle Prediction", line=dict(color=COLORS["oracle"], width=1.5),
            legendgroup="oracle_pred", showlegend=show_legend,
        ), row=i, col=1)

        # Operational
        fig.add_trace(go.Scatter(
            x=param_dates, y=operational["spinup_parameters"][param],
            name="Operational Spinup", line=dict(color=COLORS["operational"], dash="dash", width=1),
            opacity=SPINUP_OPACITY, legendgroup="operational", showlegend=show_legend,
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=param_dates, y=operational["prediction_parameters"][param],
            name="Operational Prediction", line=dict(color=COLORS["operational"], width=1.5),
            legendgroup="operational_pred", showlegend=show_legend,
        ), row=i, col=1)

        fig.update_yaxes(title_text=param, zeroline=False, row=i, col=1)

    fig.update_xaxes(zeroline=False)
    fig.update_layout(
        template="plotly_dark",
        title=f"Basin {basin}",
        height=250 * len(param_keys),
        legend=dict(orientation="h", x=0.5, y=-0.03, xanchor="center", yanchor="top"),
        margin=dict(l=60, r=20, t=50, b=80),
    )

    return fig