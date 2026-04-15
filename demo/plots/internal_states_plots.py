import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {
    "dynamic":"#636EFA",  # blue
    "operational":"#EF553B",  # red
    "oracle":"#00CC96",  # green
}
SPINUP_OPACITY = 0.4

def plot_shm_internal_states(basin: str, epoch: int, shm_data: pd.DataFrame) -> go.Figure:
    basin_id  = basin.split(":")[0]
    epoch_key = f"epoch{epoch}"
    
    all_output_dynamic     = shm_data["shm_dynamic"][epoch_key]["all_output"]
    all_output_operational = shm_data["shm_operational"][epoch_key]["all_output"]
    all_output_oracle      = shm_data["shm_oracle"][epoch_key]["all_output"]

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
            name="Dynamic Spinup", line=dict(color=COLORS["dynamic"], dash="dash", width=1),
            opacity=SPINUP_OPACITY, legendgroup="dynamic", showlegend=show_legend,
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=state_dates, y=dynamic["prediction_internal_states"][state],
            name="Dynamic Prediction", line=dict(color=COLORS["dynamic"], width=1.5),
            legendgroup="dynamic_pred", showlegend=show_legend,
        ), row=i, col=1)

        # Operational
        fig.add_trace(go.Scatter(
            x=state_dates, y=operational["spinup_internal_states"][state],
            name="Operational Spinup", line=dict(color=COLORS["operational"], dash="dash", width=1),
            opacity=SPINUP_OPACITY, legendgroup="operational", showlegend=show_legend,
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=state_dates, y=operational["prediction_internal_states"][state],
            name="Operational Prediction", line=dict(color=COLORS["operational"], width=1.5),
            legendgroup="operational_pred", showlegend=show_legend,
        ), row=i, col=1)

        # Oracle
        fig.add_trace(go.Scatter(
            x=state_dates, y=oracle["spinup_internal_states"][state],
            name="Oracle Spinup", line=dict(color=COLORS["oracle"], dash="dash", width=1),
            opacity=SPINUP_OPACITY, legendgroup="oracle", showlegend=show_legend,
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=state_dates, y=oracle["prediction_internal_states"][state],
            name="Oracle Prediction", line=dict(color=COLORS["oracle"], width=1.5),
            legendgroup="oracle_pred", showlegend=show_legend,
        ), row=i, col=1)

        fig.update_yaxes(title_text=f"{state} [mm]", zeroline=False, row=i, col=1)

    fig.update_xaxes(zeroline=False)
    fig.update_layout(
        template="plotly_dark",
        title=f"Basin {basin}",
        height=250 * len(state_keys),
        legend=dict(orientation="h", x=0.5, y=-0.03, xanchor="center", yanchor="top"),
        margin=dict(l=60, r=20, t=50, b=80),
    )

    return fig

def plot_dcfe_internal_states(basin: str, epoch: int, dcfe_data: pd.DataFrame) -> go.Figure:
    basin_id  = basin.split(":")[0]
    epoch_key = f"epoch{epoch}"

    all_output_dynamic     = dcfe_data["dcfe_dynamic"][epoch_key]["all_output"]
    all_output_operational = dcfe_data["dcfe_operational"][epoch_key]["all_output"]
    all_output_oracle      = dcfe_data["dcfe_oracle"][epoch_key]["all_output"]

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
            name="Dynamic Spinup", line=dict(color=COLORS["dynamic"], dash="dash", width=1),
            opacity=SPINUP_OPACITY, legendgroup="dynamic", showlegend=show_legend,
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=state_dates, y=dynamic["prediction_internal_states"][state],
            name="Dynamic Prediction", line=dict(color=COLORS["dynamic"], width=1.5),
            legendgroup="dynamic_pred", showlegend=show_legend,
        ), row=i, col=1)

        # Oracle
        fig.add_trace(go.Scatter(
            x=state_dates, y=oracle["spinup_internal_states"][state],
            name="Oracle Spinup", line=dict(color=COLORS["oracle"], dash="dash", width=1),
            opacity=SPINUP_OPACITY, legendgroup="oracle", showlegend=show_legend,
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=state_dates, y=oracle["prediction_internal_states"][state],
            name="Oracle Prediction", line=dict(color=COLORS["oracle"], width=1.5),
            legendgroup="oracle_pred", showlegend=show_legend,
        ), row=i, col=1)

        # Operational
        fig.add_trace(go.Scatter(
            x=state_dates, y=operational["spinup_internal_states"][state],
            name="Operational Spinup", line=dict(color=COLORS["operational"], dash="dash", width=1),
            opacity=SPINUP_OPACITY, legendgroup="operational", showlegend=show_legend,
        ), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=state_dates, y=operational["prediction_internal_states"][state],
            name="Operational Prediction", line=dict(color=COLORS["operational"], width=1.5),
            legendgroup="operational_pred", showlegend=show_legend,
        ), row=i, col=1)

        fig.update_yaxes(title_text=f"{state} [mm]", zeroline=False, row=i, col=1)

    fig.update_xaxes(zeroline=False)
    fig.update_layout(
        template="plotly_dark",
        title=f"Basin {basin}",
        height=250 * len(state_keys),
        legend=dict(orientation="h", x=0.5, y=-0.03, xanchor="center", yanchor="top"),
        margin=dict(l=60, r=20, t=50, b=80),
    )

    return fig