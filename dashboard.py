"""
Dashboard profesional para visualizar estadísticas de simulación demográfica.

Diseño:
- Enfoque científico: población media con intervalo de confianza, nacimientos vs muertes,
  crecimiento neto y estabilidad.
- Enfoque de ingeniería: tasa de crecimiento, balance demográfico y estructura por sexo.
- Soporte opcional para datos crudos si existe simulation_runs.csv.
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
    page_title="Simulador Demográfico | Dashboard Estadístico",
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

    return result


def summarize_final_year(df: pd.DataFrame) -> dict[str, float]:
    final_row = df.iloc[-1]
    return {
        "Años": float(len(df)),
        "Población media final": float(final_row["population_mean"]),
        "IC95 inferior": float(final_row["population_ci_95_lower"]),
        "IC95 superior": float(final_row["population_ci_95_upper"]),
        "Crecimiento medio anual": float(df["population_growth"].dropna().mean()),
        "CV medio (%)": float(df["population_cv_pct"].dropna().mean()),
    }


def make_confidence_band_figure(df: pd.DataFrame) -> Any:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["year"],
            y=df["population_ci_95_upper"],
            line=dict(color="rgba(0,0,0,0)"),
            name="IC 95% superior",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["year"],
            y=df["population_ci_95_lower"],
            fill="tonexty",
            fillcolor="rgba(37, 99, 235, 0.16)",
            line=dict(color="rgba(0,0,0,0)"),
            name="IC 95%",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["year"],
            y=df["population_mean"],
            mode="lines",
            line=dict(color="#1d4ed8", width=3),
            name="Población media",
        )
    )

    fig.update_layout(
        title="Evolución temporal de la población media con intervalo de confianza",
        xaxis_title="Año",
        yaxis_title="Población",
        template="plotly_white",
        height=460,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def make_birth_death_figure(df: pd.DataFrame) -> Any:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["year"], y=df["births_mean"], mode="lines", name="Nacimientos", line=dict(color="#16a34a", width=3)))
    fig.add_trace(go.Scatter(x=df["year"], y=df["deaths_mean"], mode="lines", name="Muertes", line=dict(color="#dc2626", width=3)))
    fig.update_layout(
        title="Dinámica vital: nacimientos vs muertes",
        xaxis_title="Año",
        yaxis_title="Eventos promedio",
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def make_growth_figure(df: pd.DataFrame) -> Any:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["year"], y=df["population_growth"], name="Crecimiento neto", marker_color="#0f766e"))
    fig.add_trace(go.Scatter(x=df["year"], y=df["population_growth_rate_pct"], mode="lines", name="Tasa de crecimiento (%)", yaxis="y2", line=dict(color="#7c3aed", width=2.5)))

    fig.update_layout(
        title="Crecimiento neto anual y tasa de crecimiento",
        xaxis_title="Año",
        yaxis=dict(title="Crecimiento neto"),
        yaxis2=dict(title="Tasa (%)", overlaying="y", side="right", showgrid=False),
        template="plotly_white",
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
        barmode="overlay",
    )
    return fig


def make_structure_figure(df: pd.DataFrame) -> Any:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["year"], y=df["men_mean"], mode="lines", stackgroup="one", name="Hombres", line=dict(color="#2563eb")))
    fig.add_trace(go.Scatter(x=df["year"], y=df["women_mean"], mode="lines", stackgroup="one", name="Mujeres", line=dict(color="#f97316")))
    fig.update_layout(
        title="Estructura poblacional promedio por sexo",
        xaxis_title="Año",
        yaxis_title="Población media",
        template="plotly_white",
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def make_stability_figure(df: pd.DataFrame) -> Any:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["year"], y=df["population_cv_pct"], mode="lines", name="CV población (%)", line=dict(color="#7c2d12", width=3)))
    fig.add_trace(go.Scatter(x=df["year"], y=df["population_std"], mode="lines", name="Desviación estándar", line=dict(color="#334155", width=2, dash="dot"), yaxis="y2"))

    fig.update_layout(
        title="Indicadores de estabilidad del sistema",
        xaxis_title="Año",
        yaxis=dict(title="CV población (%)"),
        yaxis2=dict(title="Desviación estándar", overlaying="y", side="right", showgrid=False),
        template="plotly_white",
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def make_decade_boxplot(df: pd.DataFrame) -> Any:
    decade_df = df[["decade_label", "population_mean", "births_mean", "deaths_mean"]].copy()
    melted = decade_df.melt(id_vars="decade_label", var_name="metric", value_name="value")
    fig = px.box(
        melted,
        x="decade_label",
        y="value",
        color="metric",
        title="Distribución interanual por década",
        labels={"decade_label": "Década", "value": "Valor promedio"},
        template="plotly_white",
    )
    fig.update_layout(height=480, margin=dict(l=20, r=20, t=60, b=20), boxmode="group")
    return fig


def make_final_distribution_figure(df: pd.DataFrame, raw_df: pd.DataFrame | None) -> Any:
    if raw_df is not None and not raw_df.empty and "year" in raw_df.columns and "population" in raw_df.columns:
        final_year = int(raw_df["year"].max())
        final_data = raw_df[raw_df["year"] == final_year]
        fig = px.histogram(
            final_data,
            x="population",
            nbins=18,
            title=f"Distribución final de población (año {final_year})",
            labels={"population": "Población final"},
            template="plotly_white",
        )
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
        return fig

    final_row = df.iloc[-1]
    fig = go.Figure()
    categories = ["Min", "P05", "Media", "P95", "Max"]
    values = [final_row["population_min"], final_row["population_p05"], final_row["population_mean"], final_row["population_p95"], final_row["population_max"]]
    fig.add_trace(go.Bar(x=categories, y=values, marker_color="#0ea5e9"))
    fig.update_layout(
        title="Distribución final aproximada de población (resumen estadístico)",
        xaxis_title="Resumen",
        yaxis_title="Población",
        template="plotly_white",
        height=420,
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

    st.markdown('<div class="dashboard-title">Dashboard Estadístico del Simulador Demográfico</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="dashboard-subtitle">Enfoque A + un poco de B: rigor científico, incertidumbre, dinámica vital y estabilidad del sistema.</div>',
        unsafe_allow_html=True,
    )

    if not STATS_FILE.exists():
        st.error(f"No se encontró {STATS_FILE}")
        st.stop()

    loaded_stats = load_statistics()
    if loaded_stats.empty:
        st.error(f"No se pudo cargar {STATS_FILE} o el archivo está vacío.")
        st.stop()

    stats_df = add_derived_metrics(loaded_stats)
    raw_df = try_load_raw_runs()

    with st.sidebar:
        st.header("Filtros")
        year_min = int(stats_df["year"].min())
        year_max = int(stats_df["year"].max())
        selected_years = st.slider("Rango de años", year_min, year_max, (year_min, year_max))
        show_raw = st.toggle("Usar distribución cruda si existe simulation_runs.csv", value=True)

    filtered_df: pd.DataFrame = stats_df.loc[(stats_df["year"] >= selected_years[0]) & (stats_df["year"] <= selected_years[1])]

    if filtered_df.empty:
        st.warning("No hay datos para el rango seleccionado.")
        st.stop()

    summary = summarize_final_year(filtered_df)
    cols = st.columns(5)
    cols[0].metric("Años", f"{int(summary['Años'])}")
    cols[1].metric("Población final media", f"{summary['Población media final']:.2f}")
    cols[2].metric("IC 95% final", f"{summary['IC95 inferior']:.2f} - {summary['IC95 superior']:.2f}")
    cols[3].metric("Crecimiento medio anual", f"{summary['Crecimiento medio anual']:.2f}")
    cols[4].metric("CV medio (%)", f"{summary['CV medio (%)']:.2f}")

    tab1, tab2, tab3 = st.tabs(["Científico", "Ingeniería", "Distribuciones"])

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

        st.subheader("Indicadores clave")
        k1, k2, k3 = st.columns(3)
        k1.metric("Balance nacimientos - muertes", f"{filtered_df['births_minus_deaths'].mean():.2f}")
        k2.metric("Balance parejas - rupturas", f"{filtered_df['couples_minus_breakups'].mean():.2f}")
        k3.metric("Tasa máxima de crecimiento (%)", f"{filtered_df['population_growth_rate_pct'].max():.2f}")

    with tab3:
        st.plotly_chart(make_decade_boxplot(filtered_df), use_container_width=True)
        if show_raw:
            st.plotly_chart(make_final_distribution_figure(filtered_df, raw_df), use_container_width=True)
            if raw_df is None:
                st.info(
                    "No se encontró simulation_runs.csv. Se mostró un resumen estadístico aproximado en lugar de un histograma real."
                )
        else:
            st.plotly_chart(make_final_distribution_figure(filtered_df, None), use_container_width=True)

    st.caption("Fuente: simulation_statistics.csv. El panel puede usar simulation_runs.csv si está disponible para distribuciones finales reales.")


if __name__ == "__main__":
    main()