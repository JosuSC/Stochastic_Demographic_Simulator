"""
Dashboard for visualizing the demographic simulation results.

Design:
- Scientific view: mean population with confidence interval, births vs deaths,
  net growth, and stability.
- Engineering view: growth rate, demographic balance, and sex structure.
- Raw data can be shown if simulation_runs.csv exists.

Run with: streamlit run dashboard.py
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as _px  # pyright: ignore[reportMissingTypeStubs]
import plotly.graph_objects as _go  # pyright: ignore[reportMissingTypeStubs]
import streamlit as _st  # pyright: ignore[reportMissingTypeStubs]

px: Any = _px
go: Any = _go
st: Any = _st


STATS_FILE = Path("simulation_statistics.csv")
RAW_RUNS_FILE = Path("simulation_runs.csv")


st.set_page_config(
    page_title="Demographic Simulator | Statistical Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: radial-gradient(circle at top left, #f6f9ff 0%, #edf2f7 45%, #e9eef7 100%);
            }
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
                max-width: 1500px;
            }
            .dashboard-title {
                font-size: 2.2rem;
                font-weight: 800;
                letter-spacing: -0.03em;
                margin-bottom: 0.25rem;
            }
            .dashboard-subtitle {
                font-size: 1rem;
                color: #475569;
                margin-bottom: 1rem;
            }
            div[data-testid="metric-container"] {
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: 16px;
                background: rgba(255, 255, 255, 0.85);
                padding: 0.8rem 0.9rem;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_statistics(path: Path = STATS_FILE) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    return df.sort_values("year").reset_index(drop=True)


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["population_growth"] = result["population_mean"].diff()
    result["population_growth_rate_pct"] = result["population_mean"].pct_change() * 100.0
    result["births_minus_deaths"] = result["births_mean"] - result["deaths_mean"]
    result["couples_minus_breakups"] = result["couples_mean"] - result.get("breakups_mean", 0)
    result["population_cv_pct"] = np.where(
        result["population_mean"] != 0,
        (result["population_std"] / result["population_mean"]) * 100.0,
        np.nan,
    )
    result["decade"] = (result["year"] // 10) * 10
    result["decade_label"] = result["decade"].astype(str) + "-" + (result["decade"] + 9).astype(str)

    # Add the extra demographic indicators when the data allows it.
    if "births_mean" in result.columns and "population_mean" in result.columns:
        result["births_per_1000"] = np.where(
            result["population_mean"] > 0,
            (result["births_mean"] / result["population_mean"]) * 1000.0, 0.0,
        )
    if "deaths_mean" in result.columns and "population_mean" in result.columns:
        result["deaths_per_1000"] = np.where(
            result["population_mean"] > 0,
            (result["deaths_mean"] / result["population_mean"]) * 1000.0, 0.0,
        )
    if "men_mean" in result.columns and "women_mean" in result.columns:
        result["sex_ratio"] = np.where(
            result["women_mean"] > 0,
            (result["men_mean"] / result["women_mean"]) * 100.0, 0.0,
        )

    return result


def summarize_final_year(df: pd.DataFrame) -> dict[str, float]:
    final_row = df.iloc[-1]
    return {
        "Years": float(len(df)),
        "Final mean population": float(final_row["population_mean"]),
        "CI95 lower": float(final_row.get("population_ci_95_lower", 0)),
        "CI95 upper": float(final_row.get("population_ci_95_upper", 0)),
        "Mean annual growth": float(df["population_growth"].dropna().mean()),
        "Mean CV (%)": float(df["population_cv_pct"].dropna().mean()),
    }


def make_confidence_band_figure(df: pd.DataFrame) -> Any:
    fig = go.Figure()

    has_ci = "population_ci_95_upper" in df.columns and "population_ci_95_lower" in df.columns
    if has_ci:
        fig.add_trace(go.Scatter(
            x=df["year"], y=df["population_ci_95_upper"],
            line=dict(color="rgba(0,0,0,0)"), name="CI 95% upper", showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=df["year"], y=df["population_ci_95_lower"],
            fill="tonexty", fillcolor="rgba(37, 99, 235, 0.16)",
            line=dict(color="rgba(0,0,0,0)"), name="CI 95%", hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=df["year"], y=df["population_mean"],
        mode="lines", line=dict(color="#1d4ed8", width=3), name="Mean population",
    ))

    fig.update_layout(
        title="Population evolution with confidence interval",
        xaxis_title="Year", yaxis_title="Population",
        template="plotly_white", height=460,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def make_birth_death_figure(df: pd.DataFrame) -> Any:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["year"], y=df["births_mean"], mode="lines",
                             name="Births", line=dict(color="#16a34a", width=3)))
    fig.add_trace(go.Scatter(x=df["year"], y=df["deaths_mean"], mode="lines",
                             name="Deaths", line=dict(color="#dc2626", width=3)))
    fig.update_layout(
        title="Vital dynamics: births vs deaths",
        xaxis_title="Year", yaxis_title="Average events",
        template="plotly_white", height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def make_growth_figure(df: pd.DataFrame) -> Any:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["year"], y=df["population_growth"],
                         name="Net growth", marker_color="#0f766e"))
    if "population_growth_rate_pct" in df.columns:
        fig.add_trace(go.Scatter(x=df["year"], y=df["population_growth_rate_pct"],
                                 mode="lines", name="Growth rate (%)", yaxis="y2",
                                 line=dict(color="#7c3aed", width=2.5)))

    fig.update_layout(
        title="Net annual growth and growth rate",
        xaxis_title="Year",
        yaxis=dict(title="Net growth"),
        yaxis2=dict(title="Rate (%)", overlaying="y", side="right", showgrid=False),
        template="plotly_white", height=430,
        margin=dict(l=20, r=20, t=60, b=20),
        barmode="overlay",
    )
    return fig


def make_structure_figure(df: pd.DataFrame) -> Any:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["year"], y=df["men_mean"], mode="lines",
                             stackgroup="one", name="Men", line=dict(color="#2563eb")))
    fig.add_trace(go.Scatter(x=df["year"], y=df["women_mean"], mode="lines",
                             stackgroup="one", name="Women", line=dict(color="#f97316")))
    fig.update_layout(
        title="Population structure by sex (average)",
        xaxis_title="Year", yaxis_title="Mean population",
        template="plotly_white", height=430,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def make_stability_figure(df: pd.DataFrame) -> Any:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["year"], y=df["population_cv_pct"], mode="lines",
                             name="CV population (%)", line=dict(color="#7c2d12", width=3)))
    fig.add_trace(go.Scatter(x=df["year"], y=df["population_std"], mode="lines",
                             name="Std deviation", line=dict(color="#334155", width=2, dash="dot"),
                             yaxis="y2"))

    fig.update_layout(
        title="System stability indicators",
        xaxis_title="Year",
        yaxis=dict(title="CV (%)"),
        yaxis2=dict(title="Std deviation", overlaying="y", side="right", showgrid=False),
        template="plotly_white", height=430,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def make_decade_boxplot(df: pd.DataFrame) -> Any:
    decade_df = df[["decade_label", "population_mean", "births_mean", "deaths_mean"]].copy()
    melted = decade_df.melt(id_vars="decade_label", var_name="metric", value_name="value")
    fig = px.box(
        melted, x="decade_label", y="value", color="metric",
        title="Interannual distribution by decade",
        labels={"decade_label": "Decade", "value": "Mean value"},
        template="plotly_white",
    )
    fig.update_layout(height=480, margin=dict(l=20, r=20, t=60, b=20), boxmode="group")
    return fig


def make_final_distribution_figure(df: pd.DataFrame, raw_df: pd.DataFrame | None) -> Any:
    if raw_df is not None and not raw_df.empty and "year" in raw_df.columns and "population" in raw_df.columns:
        final_year = int(raw_df["year"].max())
        final_data = raw_df[raw_df["year"] == final_year]
        fig = px.histogram(
            final_data, x="population", nbins=18,
            title=f"Final population distribution (year {final_year})",
            labels={"population": "Final population"},
            template="plotly_white",
        )
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
        return fig

    final_row = df.iloc[-1]
    fig = go.Figure()
    categories = ["Min", "P05", "Mean", "P95", "Max"]
    values = [
        final_row.get("population_min", 0),
        final_row.get("population_p05", 0),
        final_row.get("population_mean", 0),
        final_row.get("population_p95", 0),
        final_row.get("population_max", 0),
    ]
    fig.add_trace(go.Bar(x=categories, y=values, marker_color="#0ea5e9"))
    fig.update_layout(
        title="Approximate final population distribution",
        xaxis_title="Summary", yaxis_title="Population",
        template="plotly_white", height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def try_load_raw_runs(path: Path = RAW_RUNS_FILE) -> pd.DataFrame | None:
    if not path.exists():
        return None
    raw = pd.read_csv(path)
    if raw.empty:
        return None
    return raw


def main() -> None:
    _inject_css()

    st.markdown('<div class="dashboard-title">Demographic Simulator - Statistical Dashboard</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="dashboard-subtitle">Scientific + Engineering analysis: uncertainty, '
        'vital dynamics, and system stability.</div>',
        unsafe_allow_html=True,
    )

    if not STATS_FILE.exists():
        st.error(f"File not found: {STATS_FILE}")
        st.info("Run `python run_analysis.py` first to generate the statistics.")
        st.stop()

    loaded_stats = load_statistics()
    if loaded_stats.empty:
        st.error(f"Could not load or empty: {STATS_FILE}")
        st.stop()

    stats_df = add_derived_metrics(loaded_stats)
    raw_df = try_load_raw_runs()

    with st.sidebar:
        st.header("Filters")
        year_min = int(stats_df["year"].min())
        year_max = int(stats_df["year"].max())
        selected_years = st.slider("Year range", year_min, year_max, (year_min, year_max))
        show_raw = st.toggle("Use raw distribution if available", value=True)

    filtered_df = stats_df.loc[(stats_df["year"] >= selected_years[0]) & (stats_df["year"] <= selected_years[1])]

    if filtered_df.empty:
        st.warning("No data for the selected range.")
        st.stop()

    summary = summarize_final_year(filtered_df)
    cols = st.columns(5)
    cols[0].metric("Years", f"{int(summary['Years'])}")
    cols[1].metric("Final mean pop.", f"{summary['Final mean population']:.1f}")
    cols[2].metric("CI 95% final", f"{summary['CI95 lower']:.1f} - {summary['CI95 upper']:.1f}")
    cols[3].metric("Mean annual growth", f"{summary['Mean annual growth']:.2f}")
    cols[4].metric("Mean CV (%)", f"{summary['Mean CV (%)']:.2f}")

    tab1, tab2, tab3 = st.tabs(["Scientific", "Engineering", "Distributions"])

    with tab1:
        left, right = st.columns((1.35, 1))
        with left:
            st.plotly_chart(make_confidence_band_figure(filtered_df), use_container_width=True)
        with right:
            st.plotly_chart(make_birth_death_figure(filtered_df), use_container_width=True)
        st.plotly_chart(make_growth_figure(filtered_df), use_container_width=True)

    with tab2:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(make_structure_figure(filtered_df), use_container_width=True)
        with right:
            st.plotly_chart(make_stability_figure(filtered_df), use_container_width=True)

        st.subheader("Key indicators")
        k1, k2, k3 = st.columns(3)
        k1.metric("Births - Deaths balance", f"{filtered_df['births_minus_deaths'].mean():.2f}")
        k2.metric("Couples - Breakups balance", f"{filtered_df['couples_minus_breakups'].mean():.2f}")
        k3.metric("Max growth rate (%)", f"{filtered_df['population_growth_rate_pct'].max():.2f}")

    with tab3:
        st.plotly_chart(make_decade_boxplot(filtered_df), use_container_width=True)
        if show_raw:
            st.plotly_chart(make_final_distribution_figure(filtered_df, raw_df), use_container_width=True)
            if raw_df is None:
                st.info("No simulation_runs.csv found. Showing approximate summary instead of histogram.")
        else:
            st.plotly_chart(make_final_distribution_figure(filtered_df, None), use_container_width=True)

    st.caption("Source: simulation_statistics.csv. Dashboard can use simulation_runs.csv for real distributions.")


if __name__ == "__main__":
    main()
