#!/usr/bin/env python3
"""Streamlit UI for the gray-zone monitor (read-only artifacts)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st

try:  # pydeck is optional; fall back if missing
    import pydeck as pdk  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pdk = None

import matplotlib.pyplot as plt

from deepseek_enrichment import summarize_theater

ENRICHED_DIR = Path("data/enriched")
EXAMPLES_DIR = Path("examples")
GRID_STEP = 0.5


def _latest_parquet(directory: Path) -> Path | None:
    candidates = list(directory.glob("*.parquet"))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0]


def _grid_to_centroid(grid_id: str) -> tuple[float, float] | None:
    if not isinstance(grid_id, str) or not grid_id.startswith("R"):
        return None
    if "C" not in grid_id:
        return None
    try:
        row_part, col_part = grid_id[1:].split("C", 1)
        row = int(row_part)
        col = int(col_part)
    except ValueError:
        return None
    lat = row * GRID_STEP - 90 + GRID_STEP / 2
    lon = col * GRID_STEP - 180 + GRID_STEP / 2
    return lat, lon


def _prepare_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for grid_id, group in df.groupby("grid_id"):
        centroid = _grid_to_centroid(grid_id)
        if centroid is None:
            continue
        lat, lon = centroid
        mean_risk = pd.to_numeric(group["risk_score"], errors="coerce").mean()
        rows.append({
            "grid_id": grid_id,
            "lat": lat,
            "lon": lon,
            "risk_score": mean_risk,
        })
    return pd.DataFrame(rows)


def _watch_cells(
    df: pd.DataFrame, cutoff: float, only_mnd: bool) -> pd.DataFrame:
    scope = df[df["risk_score"].notna()].copy()
    if only_mnd:
        scope = scope[scope["source"] == "MND"]
    scope["risk_score"] = pd.to_numeric(scope["risk_score"], errors="coerce")
    scope = scope.dropna(subset=["risk_score"])
    if scope.empty:
        return scope
    grouped = scope.groupby("grid_id").agg(
        mean_risk=("risk_score", "mean"),
        count=("grid_id", "size"),
        last_seen=("dt", "max"),
        actors_sample=("actors", lambda x: _sample_actors(x)),
    )
    grouped = grouped[grouped["mean_risk"] >= cutoff]
    grouped = grouped.sort_values("mean_risk", ascending=False).head(10)
    grouped["last_seen"] = grouped["last_seen"].dt.tz_convert("UTC")
    grouped["last_seen"] = grouped["last_seen"].dt.strftime("%Y-%m-%d %H:%MZ")
    grouped["mean_risk"] = grouped["mean_risk"].round(3)
    return grouped.reset_index()


def _sample_actors(values: Iterable[Any]) -> str:
    samples: list[str] = []
    for value in values:
        if isinstance(value, list):
            samples.extend(str(item) for item in value if item)
        elif value:
            samples.append(str(value))
    unique = sorted({item for item in samples if item})
    if not unique:
        return ""
    return ", ".join(unique[:3])


def _filter_window(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    if df.empty or "dt" not in df.columns:
        return df
    df = df.copy()
    df["dt"] = pd.to_datetime(df["dt"], utc=True, errors="coerce")
    df = df.dropna(subset=["dt"])
    latest = df["dt"].max()
    if pd.isna(latest):
        return df
    window_start = latest - timedelta(hours=hours)
    return df[df["dt"] >= window_start]


def load_artifacts(
    parquet_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    parquet_path = _latest_parquet(parquet_dir)
    if parquet_path is None:
        st.error("No enriched parquet files found.")
        return pd.DataFrame(), pd.DataFrame(), None
    df = pd.read_parquet(parquet_path)
    risk_path = parquet_dir / "daily_grid_risk.csv"
    if not risk_path.exists():
        risk_path = Path("data/enriched/daily_grid_risk.csv")
    if not risk_path.exists():
        st.error("daily_grid_risk.csv not found.")
        risk_df = pd.DataFrame()
    else:
        risk_df = pd.read_csv(risk_path)
    brief_path = EXAMPLES_DIR / "airops_brief_24h.md"
    brief_text = None
    if brief_path.exists():
        brief_text = brief_path.read_text(encoding="utf-8")
    return df, risk_df, brief_text


def render_heatmap(view_df: pd.DataFrame) -> None:
    if view_df.empty:
        st.info("No incident data for the selected window.")
        return
    heat_df = _prepare_heatmap(view_df)
    if heat_df.empty:
        st.info("No grid centroids available for map display.")
        return
    if pdk is not None:
        layer = pdk.Layer(
            "HeatmapLayer",
            data=heat_df,
            get_position="[lon, lat]",
            get_weight="risk_score",
        )
        view_state = pdk.ViewState(latitude=23.5, longitude=121.0, zoom=5)
        deck = pdk.Deck(layers=[layer], initial_view_state=view_state)
        st.pydeck_chart(deck)
    else:
        st.map(heat_df.rename(columns={"lat": "latitude", "lon": "longitude"}))


def render_anomaly_chart(df: pd.DataFrame) -> None:
    chart_df = df.copy()
    if "validation_score" not in chart_df.columns or chart_df.empty:
        st.info("No validation scores available for chart.")
        return
    chart_df["dt_hour"] = chart_df["dt"].dt.floor("H")
    grouped = (
        chart_df.groupby("dt_hour")["validation_score"].mean().reset_index()
    )
    if grouped.empty:
        st.info("No validation data in selected window.")
        return
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(grouped["dt_hour"], grouped["validation_score"], marker="o")
    ax.set_ylabel("Mean validation score")
    ax.set_xlabel("UTC hour")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    st.pyplot(fig)


def render_brief(df: pd.DataFrame, brief_text: str | None) -> None:
    if brief_text:
        st.markdown(brief_text)
        return
    try:
        summary = summarize_theater(df, horizon="24h")
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Failed to generate summary: {exc}")
        return
    st.write(summary)


def main() -> None:
    st.set_page_config(page_title="Gray-Zone Monitor", layout="wide")
    st.title("Gray-Zone Air & Sea Monitor")
    st.caption("Display bbox focus: CORE 118,20,123,26")

    df, risk_df, brief_text = load_artifacts(ENRICHED_DIR)
    if df.empty:
        return

    sidebar = st.sidebar
    hours = sidebar.slider(
        "Time window (hours)", min_value=6, max_value=48, value=24
    )
    cutoff = sidebar.slider(
        "Risk cutoff", min_value=0.0, max_value=1.0, value=0.35, step=0.05
    )
    only_mnd = sidebar.checkbox("Only MND cells")

    window_df = _filter_window(df, hours)
    if only_mnd:
        window_df = window_df[window_df["source"] == "MND"]

    st.subheader("Heatmap")
    render_heatmap(window_df)

    st.subheader("Watch cells")
    watch_df = _watch_cells(_filter_window(df, hours), cutoff, only_mnd)
    st.dataframe(watch_df)

    st.subheader("MND incidents")
    mnd_table = _filter_window(df[df["source"] == "MND"], hours)
    columns = [
        "dt",
        "grid_id",
        "category",
        "severity_0_5",
        "risk_score",
        "actors",
        "summary_one_line",
        "validation_score",
        "corroborations",
    ]
    for column in columns:
        if column not in mnd_table.columns:
            mnd_table[column] = ""
    st.dataframe(mnd_table[columns])

    st.subheader("OpenSky anomaly chart")
    render_anomaly_chart(_filter_window(df[df["source"] != "MND"], hours))

    st.subheader("LLM brief (24h)")
    render_brief(mnd_table, brief_text)

    st.sidebar.markdown("---")
    latest_parquet = _latest_parquet(ENRICHED_DIR)
    if latest_parquet is not None:
        st.sidebar.download_button(
            label="Download incidents parquet",
            data=latest_parquet.read_bytes(),
            file_name=latest_parquet.name,
        )
    risk_path = ENRICHED_DIR / "daily_grid_risk.csv"
    if risk_path.exists():
        st.sidebar.download_button(
            label="Download daily grid risk",
            data=risk_path.read_bytes(),
            file_name=risk_path.name,
        )


if __name__ == "__main__":
    main()
