#!/usr/bin/env python3
"""Streamlit UI for the gray-zone monitor (read-only artifacts)."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
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
WATCHLIST_PATH = EXAMPLES_DIR / "watchlist.json"
HISTORY_INDEX_PATH = Path("data/history/history_index.json")
GRID_STEP = 0.5
OS_ANOM_PATTERN = re.compile(r"OS_ANOM:([-+]?\d+(?:\.\d+)?)")


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
    if df.empty or "grid_id" not in df.columns:
        return pd.DataFrame()
    working = df.copy()
    working["risk_score"] = pd.to_numeric(
        working.get("risk_score"), errors="coerce"
    )
    working["severity_0_5"] = pd.to_numeric(
        working.get("severity_0_5"), errors="coerce"
    )
    if "corroborations" in working.columns:
        working["os_anom_value"] = working["corroborations"].apply(
            _extract_os_anom
        )
    else:
        working["os_anom_value"] = 0.0
    working = working.dropna(subset=["risk_score"])
    rows: list[dict[str, Any]] = []
    for grid_id, group in working.groupby("grid_id"):
        centroid = _grid_to_centroid(grid_id)
        if centroid is None:
            continue
        mean_risk = float(group["risk_score"].mean())
        mean_os = float(
            pd.to_numeric(group["os_anom_value"], errors="coerce")
            .fillna(0.0)
            .mean()
        )
        mean_severity = float(
            pd.to_numeric(group.get("severity_0_5"), errors="coerce")
            .fillna(0.0)
            .mean()
        )
        last_seen = pd.to_datetime(group.get("dt"), utc=True, errors="coerce")
        last_seen = last_seen.dropna()
        last_text = (
            last_seen.max().strftime("%Y-%m-%d %H:%MZ")
            if not last_seen.empty
            else ""
        )
        actors_text = _sample_actors(group.get("actors", []))
        rows.append(
            {
                "grid_id": grid_id,
                "lat": centroid[0],
                "lon": centroid[1],
                "mean_risk": round(mean_risk, 3),
                "mean_os_anom": round(mean_os, 2),
                "mean_severity": round(mean_severity, 2),
                "last_seen": last_text,
                "actors": actors_text,
            }
        )
    map_df = pd.DataFrame(rows)
    if map_df.empty:
        return map_df
    map_df["risk_tooltip"] = map_df["mean_risk"].map(lambda val: f"{val:.2f}")
    map_df["os_tooltip"] = map_df["mean_os_anom"].map(lambda val: f"{val:.2f}")
    map_df["point_radius"] = (
        map_df["mean_severity"].clip(lower=0.0)
        + map_df["mean_os_anom"].clip(lower=0.0)
        + 1.0
    ) * 6000.0
    return map_df


def _watch_cells(
    df: pd.DataFrame, cutoff: float, only_mnd: bool, top_k: int = 5
) -> pd.DataFrame:
    if df.empty:
        return df.iloc[0:0]
    scope = df.copy()
    if only_mnd and "source" in scope.columns:
        scope = scope[scope["source"].astype(str) == "MND"]
    scope["risk_score"] = pd.to_numeric(scope.get("risk_score"), errors="coerce")
    scope = scope.dropna(subset=["risk_score"])
    if scope.empty:
        return scope.iloc[0:0]
    if "actors" not in scope.columns:
        scope["actors"] = [[] for _ in range(len(scope))]
    grouped = scope.groupby("grid_id").agg(
        mean_risk=("risk_score", "mean"),
        count=("grid_id", "size"),
        last_seen=("dt", "max"),
        actors_sample=("actors", lambda x: _sample_actors(x)),
    )
    filtered = grouped[grouped["mean_risk"] >= cutoff]
    if filtered.empty:
        filtered = grouped.sort_values("mean_risk", ascending=False).head(top_k)
    else:
        filtered = filtered.sort_values("mean_risk", ascending=False).head(top_k)
    if filtered.empty:
        return grouped.iloc[0:0]
    filtered["last_seen"] = filtered["last_seen"].dt.tz_convert("UTC")
    filtered["last_seen"] = filtered["last_seen"].dt.strftime("%Y-%m-%d %H:%MZ")
    filtered["mean_risk"] = filtered["mean_risk"].round(3)
    return filtered.reset_index()


def _window_header(df: pd.DataFrame, hours: int) -> None:
    total = len(df)
    if "source" in df.columns:
        mnd_count = int(df["source"].astype(str).eq("MND").sum())
    else:
        mnd_count = 0
    st.caption(f"Window: last {hours}h | rows: {total} (MND: {mnd_count})")



def _suggest_cutoff(df: pd.DataFrame) -> float | None:
    if df.empty or "risk_score" not in df.columns:
        return None
    series = pd.to_numeric(df["risk_score"], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.quantile(0.75))



def _friendly_summary(value: Any) -> str:
    text = "" if value is None else str(value)
    if "json parse failure" in text.lower():
        return "AI summary unavailable (fallback)"
    return text


def _sample_actors(values: Iterable[Any]) -> str:
    samples: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            candidates = value
        elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
            try:
                candidates = list(value)
            except TypeError:
                candidates = [value]
        else:
            candidates = [value]
        for item in candidates:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                samples.append(text)
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

def _apply_filters(
    df: pd.DataFrame,
    sources: Iterable[str],
    sev_min: float,
    sev_max: float,
    categories: Iterable[str],
) -> pd.DataFrame:
    """Apply sidebar filters to the incident window."""
    if df.empty:
        return df
    working = df.copy()
    if "risk_score" in working.columns:
        working["risk_score"] = pd.to_numeric(
            working["risk_score"], errors="coerce"
        )
    if "severity_0_5" in working.columns:
        working["severity_0_5"] = pd.to_numeric(
            working["severity_0_5"], errors="coerce"
        )
    allowed_sources = {str(item) for item in sources if item}
    if "source" in working.columns and allowed_sources:
        working = working[
            working["source"].astype(str).isin(allowed_sources)
        ]
    if "severity_0_5" in working.columns:
        severity_series = working["severity_0_5"]
        mask = severity_series.between(sev_min, sev_max, inclusive="both")
        if sev_min <= 0.0 and sev_max >= 5.0:
            mask = mask | severity_series.isna()
        working = working[mask]
    allowed_categories = {str(item) for item in categories if item}
    if allowed_categories and "category" in working.columns:
        category_series = working["category"].astype(str)
        working = working[category_series.isin(allowed_categories)]
    return working

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


def _atomic_write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False, dir=path.parent
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except (OSError, AttributeError):
            pass
        temp_path = Path(handle.name)
    try:
        os.replace(temp_path, path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise



def _load_watchlist(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, list):
        return [str(item) for item in payload if item]
    return []



def _save_watchlist(path: Path, grids: Iterable[str]) -> list[str]:
    unique = sorted({str(item) for item in grids if item})
    _atomic_write_json(path, unique)
    return unique



def _extract_os_anom(value: Any) -> float:
    if isinstance(value, list):
        values = value
    else:
        values = [value]
    for item in values:
        match = OS_ANOM_PATTERN.search(str(item))
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return 0.0



def _compute_streaks(df: pd.DataFrame, cutoff: float) -> dict[str, int]:
    streaks: dict[str, int] = {}
    if df.empty or "grid_id" not in df.columns:
        return streaks
    working = df.copy()
    working["dt"] = pd.to_datetime(working["dt"], utc=True, errors="coerce")
    working["risk_score"] = pd.to_numeric(
        working["risk_score"], errors="coerce"
    )
    working = working.dropna(subset=["dt", "risk_score"])
    working = working[working["risk_score"] >= cutoff]
    for grid_id, group in working.groupby("grid_id"):
        group = group.sort_values("dt", ascending=False)
        streak = 0
        previous: datetime | None = None
        for dt_value in group["dt"]:
            if previous is None:
                streak = 1
            else:
                if (previous - dt_value) <= timedelta(hours=1.01):
                    streak += 1
                else:
                    break
            previous = dt_value
        if streak:
            streaks[str(grid_id)] = streak
    return streaks



def _prepare_arcade_metrics(window_df: pd.DataFrame) -> pd.DataFrame:
    arcade_df = window_df.copy()
    if arcade_df.empty:
        return arcade_df
    arcade_df["dt"] = pd.to_datetime(
        arcade_df["dt"], utc=True, errors="coerce"
    )
    arcade_df = arcade_df.dropna(subset=["dt"])
    if arcade_df.empty:
        return arcade_df
    arcade_df["risk_score"] = pd.to_numeric(
        arcade_df.get("risk_score"), errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0)
    arcade_df["severity_0_5"] = pd.to_numeric(
        arcade_df.get("severity_0_5"), errors="coerce"
    ).fillna(0).clip(0, 5)
    if "corroborations" in arcade_df.columns:
        arcade_df["os_anom"] = arcade_df["corroborations"].apply(
            _extract_os_anom
        )
    else:
        arcade_df["os_anom"] = 0.0
    source_series = arcade_df.get("source")
    if source_series is not None:
        is_mnd = source_series.astype(str).eq("MND")
    else:
        is_mnd = pd.Series(False, index=arcade_df.index)
    os_bonus = arcade_df["os_anom"].clip(lower=0.0, upper=3.0)
    base_points = arcade_df["risk_score"] * 100.0
    base_points = base_points + os_bonus * 20.0
    base_points = base_points + arcade_df["severity_0_5"] * 10.0
    base_points = base_points + is_mnd.astype(float) * 5.0
    arcade_df["row_points"] = base_points.round().astype(int)
    return arcade_df



def _load_history_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"days": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"days": []}
    days = payload.get("days")
    if not isinstance(days, list):
        return {"days": []}
    normalized: list[dict[str, str]] = []
    for entry in days:
        if not isinstance(entry, dict):
            continue
        date = entry.get("date")
        incident_path = entry.get("incident_path")
        risk_path = entry.get("risk_path")
        if not date or not incident_path:
            continue
        normalized.append(
            {
                "date": str(date),
                "incident_path": str(incident_path),
                "risk_path": str(risk_path) if risk_path else None,
            }
        )
    return {"days": normalized}



def render_heatmap(view_df: pd.DataFrame, map_layer: str) -> None:
    if view_df.empty:
        st.info("No incident data for the selected window.")
        return
    map_df = _prepare_heatmap(view_df)
    if map_df.empty:
        st.info("No grid centroids available for map display.")
        return
    watchlist = _load_watchlist(WATCHLIST_PATH)
    starred_coords: list[dict[str, float | str]] = []
    for grid_id in watchlist:
        centroid = _grid_to_centroid(grid_id)
        if centroid is None:
            continue
        starred_coords.append(
            {
                "grid_id": grid_id,
                "lat": centroid[0],
                "lon": centroid[1],
            }
        )
    selected_grid = st.session_state.get("selected_grid")
    selected_coord: dict[str, float | str] | None = None
    if isinstance(selected_grid, str):
        centroid = _grid_to_centroid(selected_grid)
        if centroid is not None:
            selected_coord = {
                "grid_id": selected_grid,
                "lat": centroid[0],
                "lon": centroid[1],
            }
    if pdk is None:
        st.map(
            map_df.rename(columns={"lat": "latitude", "lon": "longitude"})
        )
        if selected_coord is not None:
            st.caption(f"Focused grid: {selected_coord['grid_id']}")
        elif starred_coords:
            st.caption(f"Starred grids: {len(starred_coords)}")
        return
    layers = []
    tooltip = {
        "html": (
            "<b>{grid_id}</b><br/>Risk: {risk_tooltip}<br/>"
            "OS_ANOM: {os_tooltip}<br/>Last: {last_seen}<br/>Actors: {actors}"
        ),
        "style": {"backgroundColor": "#0E1117", "color": "#FAFAFA"},
    }
    if map_layer == "Heatmap":
        layers.append(
            pdk.Layer(
                "HeatmapLayer",
                data=map_df,
                get_position="[lon, lat]",
                get_weight="mean_risk",
            )
        )
    elif map_layer == "Hexagons":
        layers.append(
            pdk.Layer(
                "HexagonLayer",
                data=map_df,
                get_position="[lon, lat]",
                radius=25000,
                elevation_scale=60,
                elevation_range=[0, 300],
                extruded=True,
                coverage=1,
                get_elevation_weight="mean_risk",
            )
        )
    else:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position="[lon, lat]",
                get_radius="point_radius",
                get_fill_color="[255, 140, 0, 180]",
                pickable=True,
            )
        )
    view_state = pdk.ViewState(
        latitude=float(map_df["lat"].mean()),
        longitude=float(map_df["lon"].mean()),
        zoom=5.2,
        pitch=30,
    )
    if starred_coords:
        star_df = pd.DataFrame(starred_coords)
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=star_df,
                get_position="[lon, lat]",
                get_radius=1,
                radius_scale=9000,
                radius_min_pixels=8,
                filled=False,
                get_line_color=[255, 255, 255, 255],
                line_width_min_pixels=2,
                pickable=False,
            )
        )
    if selected_coord is not None:
        focus_df = pd.DataFrame([selected_coord])
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=focus_df,
                get_position="[lon, lat]",
                get_radius=1,
                radius_scale=14000,
                radius_min_pixels=12,
                filled=False,
                get_line_color=[255, 215, 0, 240],
                line_width_min_pixels=3,
                pickable=False,
            )
        )
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/dark-v10",
    )
    st.pydeck_chart(deck)


def render_anomaly_chart(df: pd.DataFrame) -> None:
    if df.empty or "validation_score" not in df.columns:
        st.info("No validation scores available for chart.")
        return
    chart_df = df.copy()
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


def render_analyst_tab(
    source_df: pd.DataFrame,
    window_df: pd.DataFrame,
    hours: int,
    cutoff: float,
    only_mnd: bool,
    map_layer: str,
    brief_text: str | None,
) -> None:
    st.subheader("Heatmap")
    render_heatmap(window_df, map_layer)

    st.subheader("Watch cells")
    if source_df.empty:
        watch_df = pd.DataFrame()
    else:
        watch_df = _watch_cells(source_df, cutoff, only_mnd, top_k=5)
    if watch_df.empty:
        st.info("No data in window.")
    else:
        st.dataframe(watch_df)

    st.subheader("MND incidents")
    if source_df.empty or "source" not in source_df.columns:
        mnd_table = pd.DataFrame()
    else:
        mnd_mask = source_df["source"].astype(str) == "MND"
        mnd_table = source_df[mnd_mask].copy()
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
    display_table = mnd_table.copy()
    for column in columns:
        if column not in display_table.columns:
            display_table[column] = ""
    if "summary_one_line" in display_table.columns:
        display_table["summary_one_line"] = display_table["summary_one_line"].apply(
            _friendly_summary
        )
    st.dataframe(display_table[columns])

    st.subheader("MND anomaly chart")
    render_anomaly_chart(mnd_table)

    st.subheader("LLM brief (24h)")
    render_brief(mnd_table, brief_text)


def render_arcade_tab(
    window_df: pd.DataFrame,
    cutoff: float,
    watchlist_path: Path,
    map_layer: str,
) -> None:
    arcade_df = _prepare_arcade_metrics(window_df)
    if arcade_df.empty:
        st.info("No incidents in the selected window.")
        return

    total_points = int(arcade_df["row_points"].sum())
    level = total_points // 500
    progress = total_points % 500
    st.subheader("Arcade scoreboard")
    score_cols = st.columns([1, 1, 1])
    score_cols[0].metric("Total points", f"{total_points:,}")
    score_cols[1].metric("Level", level)
    score_cols[2].metric("Rows scored", len(arcade_df))
    st.progress(progress / 500 if total_points else 0.0)
    st.caption(f"{progress}/500 to next level")

    streaks = _compute_streaks(window_df, cutoff)
    last_seen = arcade_df.groupby("grid_id")["dt"].max()
    st.subheader("Streaks")
    if streaks:
        streak_rows: list[dict[str, Any]] = []
        for grid_id, streak in streaks.items():
            last_dt = last_seen.get(grid_id)
            label = ""
            if pd.notna(last_dt):
                label = last_dt.tz_convert(timezone.utc).strftime(
                    "%Y-%m-%d %H:%MZ"
                )
            streak_rows.append(
                {"grid_id": grid_id, "streak": streak, "last_seen": label}
            )
        streak_table = pd.DataFrame(streak_rows)
        streak_table = streak_table.sort_values(
            "streak", ascending=False
        )
        st.dataframe(streak_table.reset_index(drop=True))
    else:
        st.info("No grids are above the cutoff consecutively.")

    watchlist = _load_watchlist(watchlist_path)
    available_grids = sorted(
        {str(item) for item in arcade_df["grid_id"].dropna()}
    )
    st.subheader("Watchlist")
    starred = st.multiselect(
        "Star grids", available_grids, default=watchlist
    )
    if set(starred) != set(watchlist):
        starred = _save_watchlist(watchlist_path, starred)
        st.success("Watchlist updated.")

    points_by_grid = arcade_df.groupby("grid_id")["row_points"].sum()
    risk_by_grid = arcade_df.groupby("grid_id")["risk_score"].mean()
    star_rows: list[dict[str, Any]] = []
    for grid_id in starred:
        mean_risk = risk_by_grid.get(grid_id)
        last_dt = last_seen.get(grid_id)
        label = ""
        if pd.notna(last_dt):
            label = last_dt.tz_convert(timezone.utc).strftime(
                "%Y-%m-%d %H:%MZ"
            )
        star_rows.append(
            {
                "grid_id": grid_id,
                "points": int(points_by_grid.get(grid_id, 0)),
                "streak": streaks.get(grid_id, 0),
                "mean_risk": round(mean_risk, 3)
                if pd.notna(mean_risk)
                else float("nan"),
                "last_seen": label,
            }
        )
    if star_rows:
        st.dataframe(pd.DataFrame(star_rows))
    else:
        st.info("Star grids to build a watchlist.")

    starred_above_cutoff = sum(
        1 for grid_id in starred if risk_by_grid.get(grid_id, 0.0) >= cutoff
    )
    badge_status = [
        ("First OS_ANOM>1.50", arcade_df["os_anom"].gt(1.5).any()),
        ("Level Up I", level >= 1),
        ("Watch Captain", starred_above_cutoff >= 3),
    ]
    unlocked = {name for name, status in badge_status if status}
    seen = set(st.session_state.get("arcade_badges", []))
    newly = [name for name in unlocked if name not in seen]
    if newly:
        st.balloons()
    st.session_state["arcade_badges"] = list(unlocked)
    st.subheader("Badges")
    for name, status in badge_status:
        marker = "[x]" if status else "[ ]"
        st.write(f"{marker} {name}")

    st.subheader("Visuals")
    vis_cols = st.columns([2, 1, 1])
    with vis_cols[0]:
        render_heatmap(arcade_df, map_layer)
    with vis_cols[1]:
        risk_bins = pd.cut(
            arcade_df["risk_score"],
            bins=[-0.01, 0.3, 0.6, 1.0],
            labels=["Low", "Medium", "High"],
        )
        counts = risk_bins.value_counts().sort_index()
        if counts.empty:
            st.info("No risk data for pie chart.")
        else:
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.pie(counts, labels=counts.index, autopct="%1.0f%%")
            ax.set_title("Risk mix")
            st.pyplot(fig)
    with vis_cols[2]:
        actor_counts: dict[str, int] = {}
        if "actors" in arcade_df.columns:
            for entry in arcade_df["actors"]:
                if isinstance(entry, list):
                    actors = entry
                elif entry:
                    actors = [entry]
                else:
                    continue
                for actor in actors:
                    name = str(actor).strip()
                    if not name:
                        continue
                    actor_counts[name] = actor_counts.get(name, 0) + 1
        if not actor_counts:
            st.info("No actor data for this window.")
        else:
            series = pd.Series(actor_counts)
            series = series.sort_values(ascending=False).head(8)
            fig, ax = plt.subplots(figsize=(3, 3))
            series.sort_values().plot.barh(ax=ax)
            ax.set_xlabel("Mentions")
            ax.set_ylabel("Actor")
            ax.set_title("Top actors")
            st.pyplot(fig)



def render_history_tab(
    manifest: dict[str, Any],
    hours: int,
    cutoff: float,
    only_mnd: bool,
    map_layer: str,
) -> None:
    entries = manifest.get("days", [])
    if not entries:
        st.info("No history data available.")
        return
    date_options = [entry["date"] for entry in entries]
    max_days = len(entries)
    default_window = min(7, max_days)
    if (
        "history_selected_date" not in st.session_state
        or st.session_state["history_selected_date"] not in date_options
    ):
        st.session_state["history_selected_date"] = date_options[-1]
    st.session_state.setdefault("history_playback_days", default_window)
    lookback = st.slider(
        "Playback range (days)",
        min_value=1,
        max_value=max_days,
        value=st.session_state["history_playback_days"],
        key="history_playback_days",
    )
    selected = st.selectbox(
        "History date (UTC)",
        date_options,
        index=date_options.index(st.session_state["history_selected_date"]),
        key="history_selected_date",
    )
    entry_lookup = {entry["date"]: entry for entry in entries}
    entry = entry_lookup.get(selected)
    if entry is None:
        st.warning("History entry not found.")
        return
    incident_path = Path(entry["incident_path"])
    try:
        history_df = pd.read_parquet(incident_path)
    except Exception as exc:
        st.error(f"Failed to load incidents for {selected}: {exc}")
        return
    filtered_df = _filter_window(history_df, hours)
    display_df = filtered_df
    if only_mnd and "source" in display_df.columns:
        display_df = display_df[
            display_df["source"].astype(str) == "MND"
        ]
    _window_header(display_df, hours)

    arcade_df = _prepare_arcade_metrics(display_df)
    if arcade_df.empty:
        total_points = 0
        os_mean = 0.0
    else:
        total_points = int(arcade_df["row_points"].sum())
        os_mean = float(arcade_df["os_anom"].mean())
    if "source" in history_df.columns:
        mnd_mask = history_df["source"].astype(str) == "MND"
        mnd_count = int(mnd_mask.sum())
    else:
        mnd_count = 0
    card_cols = st.columns(3)
    card_cols[0].metric("Total points", f"{total_points:,}")
    card_cols[1].metric("Mean OS_ANOM", f"{os_mean:.2f}")
    card_cols[2].metric("MND incidents", mnd_count)
    playback_entries = entries[-lookback:]
    trend_rows: list[dict[str, Any]] = []
    for trend_entry in playback_entries:
        try:
            trend_df = pd.read_parquet(trend_entry["incident_path"])
        except Exception:
            continue
        trend_filtered = _filter_window(trend_df, hours)
        if only_mnd and "source" in trend_filtered.columns:
            trend_filtered = trend_filtered[
                trend_filtered["source"].astype(str) == "MND"
            ]
        trend_metrics = _prepare_arcade_metrics(trend_filtered)
        points = (
            int(trend_metrics["row_points"].sum())
            if not trend_metrics.empty
            else 0
        )
        anomalies = (
            int(trend_metrics["os_anom"].gt(0).sum())
            if not trend_metrics.empty
            else 0
        )
        trend_rows.append(
            {
                "date": trend_entry["date"],
                "points": points,
                "anomalies": anomalies,
            }
        )
    if trend_rows:
        trend_df_plot = pd.DataFrame(trend_rows).set_index("date")
        st.subheader(f"Playback summary (last {lookback} days)")
        st.line_chart(trend_df_plot)
    render_analyst_tab(
        filtered_df, display_df, hours, cutoff, only_mnd, map_layer, None
    )
    if len(playback_entries) > 1 and st.button(
        f"Play last {lookback} days",
        key="history_play_button",
    ):
        placeholder = st.empty()
        for playback_entry in playback_entries:
            message_parts = [playback_entry["date"]]
            matching = [
                row
                for row in trend_rows
                if row["date"] == playback_entry["date"]
            ]
            if matching:
                message_parts.append(
                    f"{matching[0]['points']:,} points"
                )
                message_parts.append(
                    f"{matching[0]['anomalies']} anomalies"
                )
            placeholder.info(" | ".join(message_parts))
            st.session_state["history_selected_date"] = playback_entry["date"]
            time.sleep(0.6)
        placeholder.success(
            f"Playback finished. Showing {playback_entries[-1]['date']}."
        )
        st.experimental_rerun()



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
    window_df = _filter_window(df, hours)
    cutoff = sidebar.slider(
        "Risk cutoff", min_value=0.0, max_value=1.0, value=0.35, step=0.05
    )
    suggestion = _suggest_cutoff(window_df)
    if suggestion is not None:
        sidebar.caption(f"Suggested cutoff: {suggestion:.2f}")
    map_layer = sidebar.selectbox(
        "Map layer", ("Heatmap", "Hexagons", "Points"), index=0
    )
    only_mnd = sidebar.checkbox("Only MND cells")

    if "source" in window_df.columns:
        source_options = sorted(
            {
                str(item)
                for item in window_df["source"].dropna().unique()
            }
        )
    else:
        source_options = []
    if not source_options:
        source_options = ["MND", "OpenSky"]
    selected_sources = sidebar.multiselect(
        "Sources",
        source_options,
        default=source_options,
    )
    severity_min, severity_max = sidebar.slider(
        "Severity range",
        min_value=0.0,
        max_value=5.0,
        value=(0.0, 5.0),
        step=0.1,
    )
    category_options: list[str] = []
    if "category" in window_df.columns:
        category_series = window_df["category"].dropna().astype(str)
        category_options = sorted(
            {
                item.strip()
                for item in category_series
                if item.strip()
            }
        )
    selected_categories = sidebar.multiselect(
        "Categories",
        category_options,
        default=category_options,
    )
    filtered_window = _apply_filters(
        window_df,
        selected_sources,
        severity_min,
        severity_max,
        selected_categories,
    )
    display_df = filtered_window
    if only_mnd and "source" in display_df.columns:
        display_df = display_df[
            display_df["source"].astype(str) == "MND"
        ]

    manifest = _load_history_manifest(HISTORY_INDEX_PATH)
    analyst_tab, arcade_tab, history_tab = st.tabs(["Analyst", "Arcade", "History"])
    with analyst_tab:
        _window_header(display_df, hours)
        render_analyst_tab(
            filtered_window, display_df, hours, cutoff, only_mnd, map_layer,
            brief_text
        )
    with arcade_tab:
        _window_header(display_df, hours)
        render_arcade_tab(display_df, cutoff, WATCHLIST_PATH, map_layer)
    with history_tab:
        render_history_tab(manifest, hours, cutoff, only_mnd, map_layer)

    sidebar.markdown("---")
    latest_parquet = _latest_parquet(ENRICHED_DIR)
    if latest_parquet is not None:
        sidebar.download_button(
            label="Download incidents parquet",
            data=latest_parquet.read_bytes(),
            file_name=latest_parquet.name,
        )
    risk_path = ENRICHED_DIR / "daily_grid_risk.csv"
    if risk_path.exists():
        sidebar.download_button(
            label="Download daily grid risk",
            data=risk_path.read_bytes(),
            file_name=risk_path.name,
        )
if __name__ == "__main__":
    main()
