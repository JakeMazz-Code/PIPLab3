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

try:
    import pydeck as pdk  # type: ignore
except Exception:
    pdk = None

import matplotlib.pyplot as plt

from deepseek_enrichment import summarize_theater

ENRICHED_DIR = Path("data/enriched")
EXAMPLES_DIR = Path("examples")
WATCHLIST_PATH = EXAMPLES_DIR / "watchlist.json"
HISTORY_INDEX_PATH = Path("data/history/history_index.json")
GRID_STEP = 0.5
OS_ANOM_PATTERN = re.compile(r"OS_ANOM:([-+]?\d+(?:\.\d+)?)")
MAX_TABLE_ROWS = 1000

MAP_LATITUDE = 23.5
MAP_LONGITUDE = 121.0
MAP_ZOOM = 5.0
MAP_HEIGHT = 520

if pdk is not None:
    _DEFAULT_VIEW = pdk.ViewState(
        latitude=MAP_LATITUDE, longitude=MAP_LONGITUDE, zoom=MAP_ZOOM
    )
else:
    _DEFAULT_VIEW = None


@st.cache_data(ttl=60)
def _load_latest_enriched_cached(path_str: str, mtime: float) -> pd.DataFrame:
    _ = mtime
    try:
        return pd.read_parquet(path_str)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def _load_snapshot_risk_cached(path_str: str, mtime: float) -> pd.DataFrame:
    _ = mtime
    try:
        return pd.read_csv(path_str)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def _list_history_days_cached(
    files: tuple[tuple[str, float], ...]
) -> list[str]:
    days: list[str] = []
    for path_str, _ in files:
        name = Path(path_str).name
        if name.startswith("incidents_") and name.endswith(".parquet"):
            days.append(name.split("_", 1)[1].split(".", 1)[0])
    return sorted({day for day in days if day})


@st.cache_data(ttl=60)
def _cached_history_parquet(
    files: tuple[tuple[str, float], ...]
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path_str, _ in files:
        try:
            frame = pd.read_parquet(path_str)
        except Exception:
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    working = pd.concat(frames, ignore_index=True)
    working["dt"] = pd.to_datetime(
        working.get("dt"), utc=True, errors="coerce"
    )
    working = working.dropna(subset=["dt"])
    return working


@st.cache_data(ttl=60)
def _cached_history_risk(
    files: tuple[tuple[str, float], ...]
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path_str, _ in files:
        try:
            frame = pd.read_csv(path_str)
        except Exception:
            continue
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    combined["grid_id"] = combined.get("grid_id", "").astype(str).str.strip()
    combined["risk_score"] = pd.to_numeric(
        combined.get("risk_score"), errors="coerce"
    )
    combined = combined.dropna(subset=["grid_id", "risk_score"])
    if combined.empty:
        return combined
    aggregated = (
        combined.groupby("grid_id")["risk_score"].mean().reset_index()
    )
    aggregated["lat"] = pd.to_numeric(aggregated.get("lat"), errors="coerce")
    aggregated["lon"] = pd.to_numeric(aggregated.get("lon"), errors="coerce")
    need_coords = aggregated["lat"].isna() | aggregated["lon"].isna()
    if need_coords.any():
        coords = aggregated.loc[need_coords, "grid_id"].apply(
            _grid_to_centroid
        )
        aggregated.loc[need_coords, "lat"] = coords.apply(
            lambda value: value[0] if value else float("nan")
        )
        aggregated.loc[need_coords, "lon"] = coords.apply(
            lambda value: value[1] if value else float("nan")
        )
    aggregated = aggregated.dropna(subset=["lat", "lon"])
    return aggregated


def _load_latest_enriched() -> tuple[pd.DataFrame, Path | None]:
    dir_mtime = _path_mtime(ENRICHED_DIR)
    parquet_path = _latest_parquet(ENRICHED_DIR, dir_mtime)
    if parquet_path is None:
        return pd.DataFrame(), None
    try:
        parquet_mtime = parquet_path.stat().st_mtime
    except OSError:
        parquet_mtime = time.time()
    df = _load_latest_enriched_cached(str(parquet_path), parquet_mtime)
    if df.empty:
        return pd.DataFrame(), parquet_path
    working = df.copy()
    working["dt"] = pd.to_datetime(
        working.get("dt"), utc=True, errors="coerce"
    )
    working = working.dropna(subset=["dt"])
    working["source"] = working.get("source", "").astype(str)
    working["source"] = working["source"].str.upper().str.strip()
    working.loc[working["source"].eq("OPENSKY"), "source"] = "OS"
    working.loc[working["source"].eq("OPEN SKY"), "source"] = "OS"
    for column in ("lat", "lon"):
        if column not in working.columns:
            working[column] = None
        working[column] = pd.to_numeric(working[column], errors="coerce")
    if "grid_id" not in working.columns:
        working["grid_id"] = None
    return working, parquet_path


def _load_snapshot_risk() -> tuple[pd.DataFrame, Path | None]:
    risk_path = ENRICHED_DIR / "daily_grid_risk.csv"
    if not risk_path.exists():
        return pd.DataFrame(), None
    try:
        risk_mtime = risk_path.stat().st_mtime
    except OSError:
        risk_mtime = time.time()
    df = _load_snapshot_risk_cached(str(risk_path), risk_mtime)
    if df.empty:
        return pd.DataFrame(), risk_path
    working = df.copy()
    working["grid_id"] = working.get("grid_id", "").astype(str).str.strip()
    working["risk_score"] = pd.to_numeric(
        working.get("risk_score"), errors="coerce"
    ).clip(lower=0.0, upper=1.0)
    working = working.dropna(subset=["grid_id", "risk_score"])
    for column in ("lat", "lon"):
        if column not in working.columns:
            working[column] = None
        working[column] = pd.to_numeric(working[column], errors="coerce")
    needs_coords = working["lat"].isna() | working["lon"].isna()
    if needs_coords.any():
        coords = working.loc[needs_coords, "grid_id"].apply(_grid_to_centroid)
        working.loc[needs_coords, "lat"] = coords.apply(
            lambda value: value[0] if value else float("nan")
        )
        working.loc[needs_coords, "lon"] = coords.apply(
            lambda value: value[1] if value else float("nan")
        )
    working = working.dropna(subset=["lat", "lon"])
    return working, risk_path


def _list_history_days() -> list[str]:
    incidents_dir = ENRICHED_DIR.parent / "history" / "incidents_enriched"
    if not incidents_dir.exists():
        return []
    files: list[tuple[str, float]] = []
    for path in incidents_dir.glob("incidents_*.parquet"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        files.append((str(path), mtime))
    return _list_history_days_cached(tuple(sorted(files)))


def _history_incident_files(days: int) -> tuple[tuple[str, float], ...]:
    incidents_dir = ENRICHED_DIR.parent / "history" / "incidents_enriched"
    days_available = _list_history_days()
    if not days_available:
        return tuple()
    selected = days_available[-days:]
    files: list[tuple[str, float]] = []
    for day in selected:
        path = incidents_dir / f"incidents_{day}.parquet"
        if not path.exists():
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        files.append((str(path), mtime))
    return tuple(sorted(files))


def _history_risk_files(days: int) -> tuple[tuple[str, float], ...]:
    risk_dir = ENRICHED_DIR.parent / "history" / "daily_grid_risk"
    days_available = _list_history_days()
    if not days_available:
        return tuple()
    selected = days_available[-days:]
    files: list[tuple[str, float]] = []
    for day in selected:
        path = risk_dir / f"risk_{day}.csv"
        if not path.exists():
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        files.append((str(path), mtime))
    return tuple(sorted(files))


def _load_history_enriched(days: int) -> pd.DataFrame:
    files = _history_incident_files(days)
    if not files:
        return pd.DataFrame()
    return _cached_history_parquet(files)


def _load_history_risk(days: int) -> pd.DataFrame:
    files = _history_risk_files(days)
    if not files:
        return pd.DataFrame()
    return _cached_history_risk(files)
def _map_view_state() -> Any:
    if pdk is None:
        return None
    return pdk.ViewState(
        latitude=MAP_LATITUDE, longitude=MAP_LONGITUDE, zoom=MAP_ZOOM
    )


def _split_sources(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    empty = df.iloc[0:0] if not df.empty else pd.DataFrame()
    if df.empty or "source" not in df.columns:
        return empty.copy(), empty.copy()
    working = df.copy()
    working["source"] = working["source"].astype(str).str.upper().str.strip()
    os_mask = working["source"].isin({"OS", "OPENSKY"})
    mnd_mask = working["source"].eq("MND")
    df_os = working[os_mask].copy()
    df_mnd = working[mnd_mask].copy()
    if not df_os.empty:
        df_os["lat"] = pd.to_numeric(df_os.get("lat"), errors="coerce")
        df_os["lon"] = pd.to_numeric(df_os.get("lon"), errors="coerce")
        df_os = df_os.dropna(subset=["lat", "lon"])
        df_os["velocity"] = pd.to_numeric(
            df_os.get("velocity"), errors="coerce"
        ).fillna(0.0)
        velocity = df_os["velocity"].clip(lower=0.0).fillna(0.0)
        df_os["radius"] = (velocity + 1.0) * 2000.0
    if not df_mnd.empty:
        df_mnd["lat"] = pd.to_numeric(df_mnd.get("lat"), errors="coerce")
        df_mnd["lon"] = pd.to_numeric(df_mnd.get("lon"), errors="coerce")
        need_coords = df_mnd["lat"].isna() | df_mnd["lon"].isna()
        if need_coords.any():
            coords = df_mnd.loc[need_coords, "grid_id"].apply(
                _grid_to_centroid
            )
            df_mnd.loc[need_coords, "lat"] = coords.apply(
                lambda value: value[0] if value else float("nan")
            )
            df_mnd.loc[need_coords, "lon"] = coords.apply(
                lambda value: value[1] if value else float("nan")
            )
        df_mnd = df_mnd.dropna(subset=["lat", "lon"])
        df_mnd["radius"] = 16000.0
    return df_os, df_mnd


def _format_dt_labels(series: pd.Series) -> pd.Series:
    dt_series = pd.to_datetime(series, utc=True, errors="coerce")
    formatted = dt_series.dt.strftime("%Y-%m-%d %H:%MZ")
    return formatted.where(dt_series.notna(), "")


def _filter_risk_window(
    risk_df: pd.DataFrame, start: datetime, end: datetime
) -> pd.DataFrame:
    if risk_df.empty:
        return risk_df
    working = risk_df.copy()
    working["day"] = pd.to_datetime(
        working.get("day"), utc=True, errors="coerce"
    )
    working = working.dropna(subset=["day"])
    if working.empty:
        return working
    start_utc = start.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)
    start_floor = start_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    mask = (working["day"] >= start_floor) & (working["day"] <= end_utc)
    filtered = working.loc[mask].copy()
    if filtered.empty:
        latest_day = working["day"].max()
        if pd.isna(latest_day):
            return filtered
        filtered = working[working["day"].eq(latest_day)].copy()
    filtered = filtered.sort_values("risk_score", ascending=False)
    filtered = filtered.drop_duplicates(subset=["grid_id"], keep="first")
    return filtered.reset_index(drop=True)



def _path_mtime(path: Path) -> float:
    """Return a safe modification time for cache keys."""

    path = Path(path)
    if not path.exists():
        return 0.0
    try:
        return path.stat().st_mtime
    except OSError:
        return time.time()


@st.cache_data(ttl=60)
def _latest_parquet(directory: Path, dir_mtime: float) -> Path | None:
    _ = dir_mtime
    path = Path(directory)
    try:
        candidates = list(path.glob("*.parquet"))
    except Exception:
        return None
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0]


@st.cache_data(ttl=60)
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
    if working.empty:
        return pd.DataFrame()
    if not working["risk_score"].ne(0).any():
        working.loc[:, "risk_score"] = 0.05
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
    map_df["lat"] = pd.to_numeric(map_df["lat"], errors="coerce")
    map_df["lon"] = pd.to_numeric(map_df["lon"], errors="coerce")
    map_df = map_df.dropna(subset=["lat", "lon"])
    if map_df.empty:
        return map_df
    map_df = map_df[
        map_df["lat"].between(-90.0, 90.0)
        & map_df["lon"].between(-180.0, 180.0)
    ].copy()
    if map_df.empty:
        return map_df
    map_df.loc[:, "lat"] = map_df["lat"].clip(-90.0, 90.0)
    map_df.loc[:, "lon"] = map_df["lon"].clip(-180.0, 180.0)
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
    if "source" in scope.columns:
        scope = scope[scope["source"].astype(str).str.upper() == "MND"]
    if scope.empty:
        return scope.iloc[0:0]
    scope["risk_score"] = pd.to_numeric(
        scope.get("risk_score"),
        errors="coerce",
    )
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
        filtered = grouped.sort_values(
            "mean_risk", ascending=False
        ).head(top_k)
    else:
        filtered = filtered.sort_values(
            "mean_risk", ascending=False
        ).head(top_k)
    if filtered.empty:
        return grouped.iloc[0:0]
    filtered["last_seen"] = filtered["last_seen"].dt.tz_convert("UTC")
    filtered["last_seen"] = filtered["last_seen"].dt.strftime(
        "%Y-%m-%d %H:%MZ"
    )
    filtered["mean_risk"] = filtered["mean_risk"].round(3)
    return filtered.reset_index()


def _render_window_header(
    df: pd.DataFrame, hours: int, risk_available: bool | None = None
) -> None:
    """Render the UTC window bounds and counts for the current view."""

    total = len(df)
    os_count = 0
    mnd_count = 0
    if "source" in df.columns:
        source_series = df["source"].astype(str).str.upper().str.strip()
        mnd_count = int(source_series.eq("MND").sum())
        os_count = int(source_series.isin({"OS", "OPENSKY"}).sum())

    start_dt: datetime | None = None
    end_dt: datetime | None = None
    if not df.empty and "dt" in df.columns:
        dt_series = pd.to_datetime(df["dt"], utc=True, errors="coerce")
        dt_series = dt_series.dropna()
        if not dt_series.empty:
            start_dt = dt_series.min().to_pydatetime()
            end_dt = dt_series.max().to_pydatetime()
    if start_dt is None or end_dt is None:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(hours=hours)
    start_dt = start_dt.astimezone(timezone.utc)
    end_dt = end_dt.astimezone(timezone.utc)
    start_floor = start_dt.replace(minute=0, second=0, microsecond=0)
    end_floor = end_dt.replace(minute=0, second=0, microsecond=0)
    span_parts = [f"last {hours} h"]
    if hours >= 24:
        days_value = hours / 24
        if hours % 24 == 0:
            days_label = f"last {int(days_value)} d"
        else:
            days_label = f"last {days_value:.1f} d"
        span_parts.append(days_label)
    span_text = " / ".join(span_parts)
    start_text = start_floor.strftime("%Y-%m-%d %H:%MZ")
    end_text = end_floor.strftime("%Y-%m-%d %H:%MZ")
    caption = (
        f"Window: {start_text} -> {end_text}  ({span_text}) | rows: {total} "
        f"(OS: {os_count} / MND: {mnd_count})"
    )
    if risk_available is not None:
        caption += f" | Risk weights: {'yes' if risk_available else 'no'}"
    st.caption(caption)


    total = len(df)
    os_count = 0
    mnd_count = 0
    if "source" in df.columns:
        source_series = df["source"].astype(str).str.upper().str.strip()
        mnd_count = int(source_series.eq("MND").sum())
        os_count = int(source_series.isin({"OS", "OPENSKY"}).sum())

    start_dt: datetime | None = None
    end_dt: datetime | None = None
    if not df.empty and "dt" in df.columns:
        dt_series = pd.to_datetime(df["dt"], utc=True, errors="coerce")
        dt_series = dt_series.dropna()
        if not dt_series.empty:
            start_dt = dt_series.min().to_pydatetime()
            end_dt = dt_series.max().to_pydatetime()
    if start_dt is None or end_dt is None:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(hours=hours)
    start_dt = start_dt.astimezone(timezone.utc)
    end_dt = end_dt.astimezone(timezone.utc)
    start_floor = start_dt.replace(minute=0, second=0, microsecond=0)
    end_floor = end_dt.replace(minute=0, second=0, microsecond=0)
    span_parts = [f"last {hours} h"]
    if hours >= 24:
        days_value = hours / 24
        if hours % 24 == 0:
            days_label = f"last {int(days_value)} d"
        else:
            days_label = f"last {days_value:.1f} d"
        span_parts.append(days_label)
    span_text = " / ".join(span_parts)
    start_text = start_floor.strftime("%Y-%m-%d %H:%MZ")
    end_text = end_floor.strftime("%Y-%m-%d %H:%MZ")
    st.caption(
        f"Window: {start_text} -> {end_text}  ({span_text}) | rows: {total} "
        f"(MND: {mnd_count})"
    )


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
        elif (
            hasattr(value, '__iter__')
            and not isinstance(value, (str, bytes))
        ):
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

def _row_actors_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return _sample_actors(value)
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        try:
            return _sample_actors(value)
        except TypeError:
            pass
    return str(value).strip()



def _filter_window(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    if df.empty or "dt" not in df.columns:
        return df
    working = df.copy()
    working["dt"] = pd.to_datetime(working["dt"], utc=True, errors="coerce")
    working = working.dropna(subset=["dt"])
    if working.empty:
        return working
    window_end = datetime.now(timezone.utc)
    window_start = window_end - timedelta(hours=hours)
    mask = (working["dt"] >= window_start) & (working["dt"] <= window_end)
    filtered = working.loc[mask]
    if filtered.empty:
        latest = working["dt"].max()
        if pd.notna(latest):
            fallback_start = latest - timedelta(hours=hours)
            filtered = working[working["dt"] >= fallback_start]
    return filtered


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


def compute_kpis(
    df_window: pd.DataFrame,
    df_prev: pd.DataFrame | None = None,
) -> dict[str, str | float]:
    """Compute KPI metrics for the analyst dashboard."""
    result: dict[str, str | float] = {
        "mnd_count": 0.0,
        "mean_risk": float("nan"),
        "mean_os_anom": float("nan"),
        "top_actor": "",
        "d_mnd": float("nan"),
        "d_mean_risk": float("nan"),
        "d_mean_os_anom": float("nan"),
    }
    if df_window.empty:
        return result
    def _aggregate(frame: pd.DataFrame) -> tuple[float, float, float]:
        if frame.empty:
            return 0.0, float("nan"), float("nan")
        working = frame.copy()
        if "risk_score" in working.columns:
            working["risk_score"] = pd.to_numeric(
                working["risk_score"], errors="coerce"
            )
        risk_mean = float("nan")
        if "risk_score" in working.columns:
            risk_series = working["risk_score"].dropna()
            if not risk_series.empty:
                risk_mean = float(risk_series.mean())
        if "source" in working.columns:
            mnd_mask = working["source"].astype(str) == "MND"
            mnd_count = float(mnd_mask.sum())
            mnd_rows = working[mnd_mask].copy()
        else:
            mnd_count = 0.0
            mnd_rows = pd.DataFrame()
        mean_os = float("nan")
        if not mnd_rows.empty and "validation_score" in mnd_rows.columns:
            mnd_rows["validation_score"] = pd.to_numeric(
                mnd_rows["validation_score"], errors="coerce"
            )
            val_series = mnd_rows["validation_score"].dropna()
            if not val_series.empty:
                mean_os = float(val_series.mean())
        return mnd_count, risk_mean, mean_os
    current_mnd, current_risk, current_os = _aggregate(df_window)
    actor_name = ""
    if "actors" in df_window.columns:
        counts: dict[str, int] = {}
        for raw in df_window["actors"]:
            if raw is None:
                continue
            if isinstance(raw, (list, tuple, set)):
                items = raw
            elif isinstance(raw, str):
                splits = re.split(r"[;,]", raw)
                items = [
                    item
                    for item in (part.strip() for part in splits)
                    if item
                ]
            elif hasattr(raw, "__iter__"):
                try:
                    items = list(raw)
                except TypeError:
                    items = [raw]
            else:
                items = [raw]
            for actor in items:
                name = str(actor).strip()
                if not name:
                    continue
                counts[name] = counts.get(name, 0) + 1
        if counts:
            actor_name = max(
                counts.items(),
                key=lambda item: (item[1], item[0]),
            )[0]
    result["mnd_count"] = current_mnd
    result["mean_risk"] = current_risk
    result["mean_os_anom"] = current_os
    result["top_actor"] = actor_name
    if df_prev is not None and not df_prev.empty:
        prev_mnd, prev_risk, prev_os = _aggregate(df_prev)
        if not pd.isna(prev_mnd):
            result["d_mnd"] = current_mnd - prev_mnd
        if not pd.isna(prev_risk):
            result["d_mean_risk"] = current_risk - prev_risk
        if not pd.isna(prev_os):
            result["d_mean_os_anom"] = current_os - prev_os
    return result



def load_artifacts(
    parquet_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    df = pd.DataFrame()
    risk_df = pd.DataFrame()
    if parquet_dir == ENRICHED_DIR:
        df, _ = _load_latest_enriched()
        risk_df, _ = _load_snapshot_risk()
    if df.empty:
        dir_mtime = _path_mtime(parquet_dir)
        parquet_path = _latest_parquet(parquet_dir, dir_mtime)
        if parquet_path is None:
            st.error("No enriched parquet files found.")
            return pd.DataFrame(), pd.DataFrame(), None
        try:
            parquet_mtime = parquet_path.stat().st_mtime
        except OSError:
            parquet_mtime = time.time()
        df = _load_parquet_df(str(parquet_path), parquet_mtime)
    if df.empty:
        st.error("Failed to load incidents parquet.")
        return pd.DataFrame(), pd.DataFrame(), None
    if risk_df.empty():
        risk_path = parquet_dir / "daily_grid_risk.csv"
        if risk_path.exists():
            try:
                risk_mtime = risk_path.stat().st_mtime
            except OSError:
                risk_mtime = time.time()
            risk_df = _load_snapshot_risk_cached(str(risk_path), risk_mtime)
        if risk_df.empty():
            st.info("daily_grid_risk.csv not found.")
    if not risk_df.empty and "risk_score" in risk_df.columns:
        risk_series = pd.to_numeric(
            risk_df["risk_score"], errors="coerce"
        ).dropna()
        if not risk_series.empty and risk_series.nunique() == 1:
            st.info("daily_grid_risk.csv has a flat risk score.")
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


@st.cache_data(ttl=60)
def _load_watchlist(path_str: str, mtime: float) -> list[str]:
    _ = mtime
    path = Path(path_str)
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
    try:
        _load_watchlist.clear()
    except AttributeError:
        pass
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


@st.cache_data(ttl=60)
def _load_parquet_df(path_str: str, mtime: float) -> pd.DataFrame:
    _ = mtime
    try:
        return pd.read_parquet(path_str)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def _load_csv_df(path_str: str, mtime: float) -> pd.DataFrame:
    _ = mtime
    try:
        return pd.read_csv(path_str)
    except Exception:
        return pd.DataFrame()


def _row_points(row: pd.Series) -> int:
    """Compute arcade points for a single incident row."""
    risk = pd.to_numeric(row.get("risk_score"), errors="coerce")
    severity = pd.to_numeric(row.get("severity_0_5"), errors="coerce")
    risk_value = float(risk) if pd.notna(risk) else 0.0
    severity_value = float(severity) if pd.notna(severity) else 0.0
    os_anom = _extract_os_anom(row.get("corroborations"))
    os_bonus = min(max(os_anom, 0.0), 3.0)
    points = risk_value * 100.0
    points += os_bonus * 20.0
    points += severity_value * 10.0
    source_label = str(row.get("source", "")).strip()
    if source_label == "MND":
        points += 5.0
    return int(round(points))


def compute_points_per_day(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate arcade points and OS_ANOM mean per day."""
    base = pd.DataFrame(columns=["date", "points", "os_anom_mean"])
    if df.empty or "dt" not in df.columns:
        return base
    working = df.copy()
    working["dt"] = pd.to_datetime(working["dt"], utc=True, errors="coerce")
    working = working.dropna(subset=["dt"])
    if working.empty:
        return base
    working["date"] = working["dt"].dt.strftime("%Y-%m-%d")
    working["row_points"] = working.apply(_row_points, axis=1)
    points = working.groupby("date")["row_points"].sum()
    os_mean = pd.Series(dtype=float)
    if ("source" in working.columns and "validation_score" in working.columns):
        mnd_mask = working["source"].astype(str) == "MND"
        scores = pd.to_numeric(
            working.loc[mnd_mask, "validation_score"], errors="coerce"
        )
        os_mean = (
            pd.DataFrame(
                {
                    "date": working.loc[mnd_mask, "date"],
                    "score": scores,
                }
            )
            .dropna(subset=["score"])
            .groupby("date")["score"]
            .mean()
        )
    result = points.rename("points").reset_index()
    if not os_mean.empty:
        result = result.merge(
            os_mean.rename("os_anom_mean").reset_index(),
            on="date",
            how="left",
        )
    else:
        result["os_anom_mean"] = float("nan")
    result["points"] = result["points"].astype(int)
    result["os_anom_mean"] = result["os_anom_mean"].astype(float)
    return result


def _render_history_summary(summary: pd.DataFrame) -> None:
    """Render points/day and OS_ANOM mean/day chart for history."""
    if summary.empty or "date" not in summary.columns:
        st.info("Not enough history to summarize.")
        return
    summary = summary.dropna(subset=["date"])
    if summary.empty:
        st.info("Not enough history to summarize.")
        return
    summary = summary.groupby("date").agg(
        points=("points", "sum"),
        os_anom_mean=("os_anom_mean", "mean"),
    ).reset_index()
    summary = summary.sort_values("date")
    dates = pd.to_datetime(summary["date"], errors="coerce")
    valid = pd.notna(dates)
    if not valid.any():
        st.info("Not enough history to summarize.")
        return
    summary = summary.loc[valid].reset_index(drop=True)
    dates = dates[valid]
    fig, ax_points = plt.subplots(figsize=(6, 2.4))
    ax_points.plot(
        dates,
        summary["points"],
        marker="o",
        color="tab:orange",
        label="Points/day",
    )
    ax_points.set_ylabel("Points/day")
    ax_points.set_xlabel("Day")
    ax_points.grid(True, axis="y", alpha=0.3)
    ax_os = ax_points.twinx()
    ax_os.plot(
        dates,
        summary["os_anom_mean"],
        marker="s",
        color="tab:blue",
        label="OS_ANOM mean",
    )
    ax_os.set_ylabel("OS_ANOM mean")
    fig.autofmt_xdate(rotation=45)
    lines = ax_points.get_lines() + ax_os.get_lines()
    labels = [line.get_label() for line in lines]
    ax_points.legend(lines, labels, loc="upper left")
    st.pyplot(fig)
    plt.close(fig)


@st.cache_data(ttl=60)
def _load_history_manifest(path_str: str, mtime: float) -> dict[str, Any]:
    _ = mtime
    path = Path(path_str)
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
    watch_mtime = _path_mtime(WATCHLIST_PATH)
    watchlist = _load_watchlist(str(WATCHLIST_PATH), watch_mtime)
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
    def _render_fallback() -> None:
        fb = map_df.rename(
            columns={"lat": "latitude", "lon": "longitude"}
        )
        if {"latitude", "longitude"}.issubset(fb.columns):
            st.map(fb[["latitude", "longitude"]])
        else:
            st.warning("No valid lat/lon to display.")
        if selected_coord is not None:
            st.caption(f"Focused grid: {selected_coord['grid_id']}")
        elif starred_coords:
            st.caption(f"Starred grids: {len(starred_coords)}")

    if pdk is None:
        _render_fallback()
        return
    layers = []
    tooltip = {
        "html": (
            "<b>{grid_id}</b><br/>Risk: {risk_tooltip}<br/>"
            "OS_ANOM: {os_tooltip}<br/>Last: {last_seen}<br/>Actors: {actors}"
        ),
        "style": {"backgroundColor": "#0E1117", "color": "#FAFAFA"},
    }
    weight_series = (
        pd.to_numeric(map_df.get("mean_risk"), errors="coerce")
        if "mean_risk" in map_df.columns
        else pd.Series(dtype=float)
    )
    all_nan_weights = weight_series.empty or weight_series.isna().all()
    small_sample = len(map_df) < 5
    use_scatter = (
        map_layer not in {"Heatmap", "Hexagons"}
        or small_sample
        or all_nan_weights
    )
    if (
        map_layer in {"Heatmap", "Hexagons"}
        and (small_sample or all_nan_weights)
    ):
        st.caption("Map weights unavailable; showing points.")
    if not use_scatter and map_layer == "Heatmap":
        layers.append(
            pdk.Layer(
                "HeatmapLayer",
                data=map_df,
                get_position="[lon, lat]",
                get_weight="mean_risk",
            )
        )
    elif not use_scatter and map_layer == "Hexagons":
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
        latitude=23.5,
        longitude=121.0,
        zoom=5.0,
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
        map_provider="carto",
        map_style="dark",
    )
    try:
        st.pydeck_chart(deck, use_container_width=True, height=520)
    except Exception:
        _render_fallback()



def _wkey(ns: str, name: str) -> str:
    """Stable Streamlit widget key using a small namespace."""

    return f"{ns}__{name}"



def _build_fallback_points(
    df_os: pd.DataFrame,
    df_mnd: pd.DataFrame,
    risk_df: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for label, frame in (("OS", df_os), ("MND", df_mnd), ("RISK", risk_df)):
        if frame is None or frame.empty:
            continue
        subset = frame[["lat", "lon"]].copy()
        subset = subset.dropna(subset=["lat", "lon"])
        if subset.empty:
            continue
        subset["layer"] = label
        frames.append(subset)
    if not frames:
        return pd.DataFrame(columns=["lat", "lon", "layer"])
    return pd.concat(frames, ignore_index=True)


def _build_map_layers(
    df_os: pd.DataFrame,
    df_mnd: pd.DataFrame,
    risk_df: pd.DataFrame,
    show_hex: bool,
    show_os: bool,
    show_mnd: bool,
    starred: Iterable[str] | None,
    selected_grid: str | None,
) -> tuple[list[Any], pd.DataFrame, bool]:
    layers: list[Any] = []
    risk_layer = pd.DataFrame()
    weights_present = False
    if not risk_df.empty:
        risk_layer = risk_df.copy()
        risk_layer["weight"] = pd.to_numeric(
            risk_layer.get("risk_score"), errors="coerce"
        )
        risk_layer = risk_layer.dropna(subset=["lat", "lon", "weight"])
        if not risk_layer.empty and risk_layer["weight"].nunique() > 1:
            weights_present = True
    if not weights_present and not df_mnd.empty:
        fallback = df_mnd.copy()
        fallback["lat"] = pd.to_numeric(fallback.get("lat"), errors="coerce")
        fallback["lon"] = pd.to_numeric(fallback.get("lon"), errors="coerce")
        needs_coords = fallback["lat"].isna() | fallback["lon"].isna()
        if needs_coords.any():
            coords = fallback.loc[needs_coords, "grid_id"].apply(
                _grid_to_centroid
            )
            fallback.loc[needs_coords, "lat"] = coords.apply(
                lambda value: value[0] if value else float("nan")
            )
            fallback.loc[needs_coords, "lon"] = coords.apply(
                lambda value: value[1] if value else float("nan")
            )
        fallback = fallback.dropna(subset=["lat", "lon"])
        if not fallback.empty:
            severity = pd.to_numeric(
                fallback.get("severity_0_5"), errors="coerce"
            ).fillna(0.0)
            os_series = fallback.get("corroborations")
            if os_series is not None:
                os_bonus = os_series.apply(_extract_os_anom).apply(
                    lambda value: 0.2 if value > 0 else 0.0
                )
            else:
                os_bonus = pd.Series(0.0, index=fallback.index)
            fallback["weight"] = severity + os_bonus
            grouped = fallback.groupby("grid_id").agg(
                weight=("weight", "mean"),
                lat=("lat", "mean"),
                lon=("lon", "mean"),
            )
            grouped = grouped.reset_index().dropna(subset=["lat", "lon"])
            if not grouped.empty and grouped["weight"].nunique() > 1:
                risk_layer = grouped
                weights_present = True
    fallback_points = _build_fallback_points(df_os, df_mnd, risk_layer)
    if pdk is None:
        return [], fallback_points, weights_present
    if weights_present:
        data_source = risk_layer.copy()
        if show_hex:
            layers.append(
                pdk.Layer(
                    "HexagonLayer",
                    data=data_source,
                    get_position="[lon, lat]",
                    get_weight="weight",
                    radius=20000,
                    elevation_scale=400,
                    extruded=True,
                    elevation_range=[0, 3000],
                )
            )
        else:
            layers.append(
                pdk.Layer(
                    "HeatmapLayer",
                    data=data_source,
                    get_position="[lon, lat]",
                    get_weight="weight",
                    radius_pixels=60,
                )
            )
    if show_os and not df_os.empty:
        os_data = df_os.copy()
        os_data["dt_label"] = _format_dt_labels(os_data.get("dt"))
        tooltip = "OS<br/>dt: " + os_data["dt_label"].fillna("")
        callsign = os_data.get("callsign")
        if callsign is not None:
            callsign = callsign.fillna("").astype(str).str.strip()
            mask = callsign.ne("")
            tooltip = tooltip.where(~mask, tooltip + "<br/>" + callsign)
        raw_text = os_data.get("raw_text")
        if raw_text is not None:
            raw_series = raw_text.fillna("").astype(str).str.strip()
            mask = raw_series.ne("")
            tooltip = tooltip.where(~mask, tooltip + "<br/>" + raw_series)
        os_data["tooltip"] = tooltip
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=os_data,
                get_position="[lon, lat]",
                get_radius="radius",
                get_fill_color="[30, 144, 255, 140]",
                pickable=True,
                auto_highlight=True,
            )
        )
    if show_mnd and not df_mnd.empty:
        mnd_data = df_mnd.copy()
        mnd_data["dt_label"] = _format_dt_labels(mnd_data.get("dt"))
        summary = mnd_data.get("summary_one_line")
        actors = mnd_data.get("actors")
        raw_text = mnd_data.get("raw_text")
        tooltip = "MND<br/>dt: " + mnd_data["dt_label"].fillna("")
        for extra in (summary, raw_text):
            if extra is None:
                continue
            series = extra.fillna("").astype(str).str.strip()
            mask = series.ne("")
            tooltip = tooltip.where(~mask, tooltip + "<br/>" + series)
        if actors is not None:
            actor_series = actors.apply(_row_actors_text).fillna("")
            actor_series = actor_series.astype(str).str.strip()
            mask = actor_series.ne("")
            tooltip = tooltip.where(~mask, tooltip + "<br/>" + actor_series)
        mnd_data["tooltip"] = tooltip
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=mnd_data,
                get_position="[lon, lat]",
                get_radius="radius",
                get_fill_color="[255, 140, 0, 220]",
                pickable=True,
                auto_highlight=True,
            )
        )
    star_ids = list(starred or [])
    if star_ids:
        star_coords: list[dict[str, float | str]] = []
        for grid_id in star_ids:
            centroid = _grid_to_centroid(grid_id)
            if centroid is None:
                continue
            star_coords.append(
                {"grid_id": grid_id, "lat": centroid[0], "lon": centroid[1]}
            )
        if star_coords:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=pd.DataFrame(star_coords),
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
    if selected_grid:
        centroid = _grid_to_centroid(selected_grid)
        if centroid is not None:
            focus_df = pd.DataFrame(
                [
                    {
                        "grid_id": selected_grid,
                        "lat": centroid[0],
                        "lon": centroid[1],
                    }
                ]
            )
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
    return layers, fallback_points, weights_present


def _render_incident_map(
    df_os: pd.DataFrame,
    df_mnd: pd.DataFrame,
    risk_df: pd.DataFrame,
    show_hex: bool,
    show_os: bool,
    show_mnd: bool,
    starred: Iterable[str] | None,
    selected_grid: str | None,
    widget_ns: str,
) -> bool:
    layers, fallback_points, weights_present = _build_map_layers(
        df_os,
        df_mnd,
        risk_df,
        show_hex,
        show_os,
        show_mnd,
        starred,
        selected_grid,
    )
    if pdk is None or not layers:
        fallback = fallback_points.dropna(subset=["lat", "lon"])
        if fallback.empty:
            fallback = _build_fallback_points(df_os, df_mnd, risk_df)
            fallback = fallback.dropna(subset=["lat", "lon"])
        if fallback.empty:
            st.info("No location data available for this window.")
        else:
            st.map(fallback[["lat", "lon"]])
        return weights_present
    try:
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=_DEFAULT_VIEW,
            map_provider="carto",
            map_style="dark",
        )
        st.pydeck_chart(deck, use_container_width=True, height=MAP_HEIGHT)
    except Exception:
        fallback = fallback_points.dropna(subset=["lat", "lon"])
        if fallback.empty:
            st.warning("Map rendering failed with no fallback points.")
        else:
            st.map(fallback[["lat", "lon"]])
    return weights_present

def render_anomaly_chart(df: pd.DataFrame) -> None:
    if df.empty or "validation_score" not in df.columns:
        st.info("No validation scores available for chart.")
        return
    chart_df = df.copy()
    chart_df["dt_hour"] = chart_df["dt"].dt.floor("h")
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


def render_monitor_tab(
    snapshot_data: tuple[pd.DataFrame, Path | None],
    snapshot_risk: tuple[pd.DataFrame, Path | None],
    history_days: list[str],
    brief_text: str | None,
) -> None:
    snapshot_df, snapshot_path = snapshot_data
    snapshot_risk_df, _ = snapshot_risk
    st.subheader("Monitor Controls")
    window_mode = st.radio(
        "Window",
        ("Last 24h", "Last 48h", "Last N days"),
        key=_wkey("mon", "window_mode"),
    )
    history_max = min(14, len(history_days)) if history_days else 0
    selected_days = 1
    window_hours = 24
    risk_window = pd.DataFrame()
    source_label = "snapshot"
    if window_mode == "Last N days":
        if history_max == 0:
            st.info("History data unavailable. Run backfill.")
            return
        selected_days = st.slider(
            "Days",
            min_value=1,
            max_value=history_max,
            value=min(7, history_max),
            key=_wkey("mon", "days"),
        )
        window_df = _load_history_enriched(selected_days)
        risk_window = _load_history_risk(selected_days)
        source_label = f"history:{selected_days}d"
        window_hours = selected_days * 24
    else:
        hours = 24 if window_mode == "Last 24h" else 48
        window_hours = hours
        if snapshot_df.empty:
            if history_max == 0:
                st.info(
                    "Snapshot unavailable and no history data present."
                )
                return
            selected_days = min(max(1, hours // 24), history_max)
            window_df = _load_history_enriched(selected_days)
            risk_window = _load_history_risk(selected_days)
            source_label = f"history:{selected_days}d (fallback)"
            window_hours = selected_days * 24
        else:
            window_df = _filter_window(snapshot_df, hours)
            risk_window = snapshot_risk_df
    if window_df.empty:
        st.info("No incidents available for the selected window.")
        return
    df_os, df_mnd = _split_sources(window_df)
    toggle_cols = st.columns(3)
    show_hex = toggle_cols[0].checkbox(
        "Use hexagons",
        value=False,
        key=_wkey("mon", "show_hex"),
    )
    show_os = toggle_cols[1].checkbox(
        "Show OpenSky points",
        value=True,
        key=_wkey("mon", "show_os"),
    )
    show_mnd = toggle_cols[2].checkbox(
        "Show MND markers",
        value=True,
        key=_wkey("mon", "show_mnd"),
    )
    watch_mtime = _path_mtime(WATCHLIST_PATH)
    saved_watchlist = _load_watchlist(str(WATCHLIST_PATH), watch_mtime)
    grid_series = window_df.get("grid_id")
    grid_options = (
        sorted({str(item).strip() for item in grid_series.dropna() if item})
        if grid_series is not None
        else []
    )
    defaults = [grid for grid in saved_watchlist if grid in grid_options]
    current_focus = st.session_state.get("selected_grid")
    if current_focus not in grid_options:
        current_focus = None
        st.session_state["selected_grid"] = None
    starred = st.multiselect(
        "Star grids",
        options=grid_options,
        default=defaults,
        key=_wkey("mon", "starred"),
    )
    if set(starred) != set(saved_watchlist):
        saved_watchlist = _save_watchlist(WATCHLIST_PATH, starred)
    missing_saved = sorted(set(saved_watchlist) - set(grid_options))
    if missing_saved:
        st.caption(
            "Watchlist defaults pruned to in-window grids: "
            + ", ".join(missing_saved)
        )
    if grid_options:
        focus_index = (
            grid_options.index(current_focus)
            if current_focus in grid_options
            else 0
        )
        focus_choice = st.selectbox(
            "Focus grid",
            grid_options,
            index=focus_index,
            key=_wkey("mon", "focus"),
        )
        if focus_choice != st.session_state.get("selected_grid"):
            st.session_state["selected_grid"] = focus_choice
    else:
        focus_choice = None
    weights_present = _render_incident_map(
        df_os,
        df_mnd,
        risk_window,
        show_hex,
        show_os,
        show_mnd,
        starred,
        st.session_state.get("selected_grid"),
        "monitor_map",
    )
    _render_window_header(
        window_df,
        window_hours,
        risk_available=weights_present,
    )
    risk_rows = len(risk_window) if isinstance(risk_window, pd.DataFrame) else 0
    source_caption = (
        f"Monitor data: source={source_label} rows={len(window_df)} "
        f"| OS: {len(df_os)} | MND: {len(df_mnd)} "
        f"| risk_rows={risk_rows} "
        f"| Risk weights: {'yes' if weights_present else 'no'}"
    )
    st.caption(source_caption)
    if snapshot_path is not None and source_label.startswith("snapshot"):
        st.caption(f"Snapshot file: {snapshot_path.name}")
    st.subheader("Watch cells")
    watch_df = _watch_cells(window_df, 0.35, True, top_k=5)
    if watch_df.empty:
        st.info("No MND grids in the selected window.")
    else:
        st.dataframe(watch_df.reset_index(drop=True))
    st.subheader("MND incidents")
    mnd_table = df_mnd.copy()
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
    if "summary_one_line" in mnd_table.columns:
        mnd_table["summary_one_line"] = mnd_table["summary_one_line"].apply(
            _friendly_summary
        )
    table_to_show = mnd_table
    if len(table_to_show) > MAX_TABLE_ROWS:
        table_to_show = table_to_show.iloc[:MAX_TABLE_ROWS]
        st.caption(
            "Table truncated for display (showing first 1000 rows)."
        )
    if not table_to_show.empty:
        st.dataframe(table_to_show[columns])
    else:
        st.info("No MND incidents in the selected window.")
    st.subheader("MND anomaly chart")
    render_anomaly_chart(mnd_table)
    st.subheader("LLM brief (24h)")
    render_brief(mnd_table, brief_text)

def render_history_tab(history_days: list[str]) -> None:
    st.subheader("History Controls")
    if not history_days:
        st.info("No history available. Run backfill.")
        return
    history_max = min(14, len(history_days))
    days = st.slider(
        "Days",
        min_value=1,
        max_value=history_max,
        value=min(7, history_max),
        key=_wkey("hist", "days"),
    )
    history_df = _load_history_enriched(days)
    if history_df.empty:
        st.info("No incidents available for the selected history window.")
        return
    risk_df = _load_history_risk(days)
    toggle_cols = st.columns(3)
    show_hex = toggle_cols[0].checkbox(
        "Use hexagons",
        value=False,
        key=_wkey("hist", "show_hex"),
    )
    show_os = toggle_cols[1].checkbox(
        "Show OpenSky points",
        value=True,
        key=_wkey("hist", "show_os"),
    )
    show_mnd = toggle_cols[2].checkbox(
        "Show MND markers",
        value=True,
        key=_wkey("hist", "show_mnd"),
    )
    df_os, df_mnd = _split_sources(history_df)
    weights_present = _render_incident_map(
        df_os,
        df_mnd,
        risk_df,
        show_hex,
        show_os,
        show_mnd,
        [],
        None,
        "history_map",
    )
    window_hours = days * 24
    _render_window_header(
        history_df,
        window_hours,
        risk_available=weights_present,
    )
    risk_rows = len(risk_df) if isinstance(risk_df, pd.DataFrame) else 0
    history_caption = (
        f"History window: days={days} rows={len(history_df)} "
        f"| OS: {len(df_os)} | MND: {len(df_mnd)} "
        f"| risk_rows={risk_rows} "
        f"| Risk weights: {'yes' if weights_present else 'no'}"
    )
    st.caption(history_caption)
    st.subheader("Watch cells")
    watch_df = _watch_cells(history_df, 0.35, True, top_k=5)
    if watch_df.empty:
        st.info("No MND grids in the selected history window.")
    else:
        st.dataframe(watch_df.reset_index(drop=True))
    st.subheader("MND incidents")
    if df_mnd.empty:
        st.info("No MND incidents in this history range.")
    else:
        history_table = df_mnd.copy()
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
            if column not in history_table.columns:
                history_table[column] = ""
        history_table["summary_one_line"] = history_table[
            "summary_one_line"
        ].apply(_friendly_summary)
        if len(history_table) > MAX_TABLE_ROWS:
            history_table = history_table.iloc[:MAX_TABLE_ROWS]
            st.caption(
                "Table truncated for display (showing first 1000 rows)."
            )
        st.dataframe(history_table[columns])
    st.subheader("MND anomaly chart")
    render_anomaly_chart(df_mnd)

def render_history_tab(
    manifest: dict[str, Any],
    hours: int,
    cutoff: float,
    only_mnd: bool,
    map_layer: str,
    risk_df: pd.DataFrame,
) -> None:
    entries = manifest.get("days", [])
    available_days = len(entries)
    if available_days == 0:
        st.info("No history available. Run backfill.")
        return
    date_options = [entry["date"] for entry in entries]
    selected_key = _wkey("history", "date")
    if (
        selected_key not in st.session_state
        or st.session_state[selected_key] not in date_options
    ):
        st.session_state[selected_key] = date_options[-1]

    slider_key = _wkey("history", "days")
    max_value = max(1, available_days)
    default_value = min(7, max_value)
    slider_value = st.session_state.get(slider_key, default_value)
    slider_value = max(1, min(slider_value, max_value))
    st.session_state[slider_key] = slider_value
    lookback = st.slider(
        "Playback range (days)",
        min_value=1,
        max_value=max_value,
        value=slider_value,
        key=slider_key,
    )

    snapshot_key = _wkey("history", "snapshot")
    st.session_state.setdefault(snapshot_key, True)
    show_snapshot = st.checkbox(
        "Show 24h snapshot",
        value=st.session_state[snapshot_key],
        key=snapshot_key,
    )

    selected_date = st.selectbox(
        "History date (UTC)",
        date_options,
        index=date_options.index(st.session_state[selected_key]),
        key=selected_key,
    )

    entry_lookup = {entry["date"]: entry for entry in entries}
    playback_entries = entries[-lookback:]
    day_cache: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    trend_rows: list[dict[str, Any]] = []

    for playback_entry in playback_entries:
        day_data, error = _prepare_history_day_entry(
            playback_entry, hours, only_mnd
        )
        if day_data is None:
            continue
        day_cache[day_data["date"]] = day_data
        day_summary = compute_points_per_day(day_data["history_df"])
        if not day_summary.empty:
            for _, day_row in day_summary.iterrows():
                summary_rows.append(
                    {
                        "date": str(
                            day_row.get("date", day_data["date"])
                        ),
                        "points": int(day_row.get("points", 0)),
                        "os_anom_mean": float(
                            day_row.get("os_anom_mean", float("nan"))
                        ),
                    }
                )
        history_df = day_data.get("history_df", pd.DataFrame())
        os_count = 0
        if not history_df.empty and "source" in history_df.columns:
            os_mask = history_df["source"].astype(str).str.upper().isin(\
            {"OS", "OPENSKY"}\
        )
            os_count = int(os_mask.sum())
        trend_rows.append(
            {
                "date": day_data["date"],
                "os_count": os_count,
                "os_anom_mean": float(day_data.get("os_mean", float("nan"))),
            }
        )

    selected_entry = entry_lookup.get(selected_date)
    selected_day = day_cache.get(selected_date)
    if selected_day is None and selected_entry is not None:
        selected_day, error = _prepare_history_day_entry(
            selected_entry, hours, only_mnd
        )
        if selected_day is None:
            message = error or (
                f"Failed to load incidents for {selected_date}."
            )
            st.error(message)
            return
        day_cache[selected_day["date"]] = selected_day

    if selected_day is None:
        st.warning("History entry not found.")
        return

    map_frames: list[pd.DataFrame] = []
    if show_snapshot:
        if "history_df" in selected_day:
            map_frames.append(selected_day["history_df"])
        map_hours = 24
    else:
        for playback_entry in playback_entries[-lookback:]:
            cached = day_cache.get(playback_entry["date"])
            if cached is None:
                continue
            map_frames.append(cached.get("history_df", pd.DataFrame()))
        map_hours = max(24, lookback * 24) if map_frames else 24
    if map_frames:
        map_df = pd.concat(map_frames, ignore_index=True)
    else:
        map_df = pd.DataFrame()
    selected_for_view = selected_day.copy()
    selected_for_view["map_df"] = map_df
    selected_for_view["map_hours"] = map_hours

    header_placeholder = st.empty()
    summary_df = (
        pd.DataFrame(summary_rows)
        if summary_rows
        else pd.DataFrame(
            columns=["date", "points", "os_anom_mean"]
        )
    )
    _render_history_day_header(header_placeholder, selected_for_view)
    st.subheader("History summary")
    _render_history_summary(summary_df)
    if trend_rows:
        trend_df_plot = pd.DataFrame(trend_rows).set_index("date")
        available_cols = [
            col
            for col in ("os_count", "os_anom_mean")
            if col in trend_df_plot.columns
        ]
        if len(trend_df_plot.index.unique()) >= 2 and available_cols:
            st.subheader(f"OS activity vs OS_ANOM (last {lookback} days)")
            st.line_chart(trend_df_plot[available_cols])

    map_placeholder = st.empty()
    _render_history_day_map(
        map_placeholder,
        selected_for_view,
        map_layer,
        risk_df,
        widget_ns="history_map",
    )
    status_placeholder = st.empty()
    snapshot_placeholder = st.empty()
    trend_lookup = {
        row.get("date"): (
            row.get("os_count", 0),
            row.get("os_anom_mean", float("nan")),
        )
        for row in trend_rows
    }
    playback_dates = [
        entry["date"]
        for entry in playback_entries
        if entry["date"] in day_cache
    ]
    if len(playback_dates) > 1:
        if st.button("Play timelapse", key=_wkey("history", "play")):
            snapshot_placeholder.empty()
            for date in playback_dates:
                day_data = day_cache.get(date)
                if day_data is None:
                    continue
                _render_history_day_view(
                    header_placeholder,
                    map_placeholder,
                    day_data,
                    map_layer,
                    risk_df,
                    widget_ns=_wkey("history", f"tl_{date}"),
                )
                os_count, os_mean = trend_lookup.get(
                    date, (0, float("nan"))
                )
                if pd.notna(os_mean):
                    status_placeholder.info(
                        f"{date}: OS={os_count:,} OS_ANOM mean={os_mean:.2f}"
                    )
                else:
                    status_placeholder.info(
                        f"{date}: OS={os_count:,}"
                    )
                st.session_state[selected_key] = date
            final_date = playback_dates[-1]
            st.session_state[selected_key] = final_date
            status_placeholder.success(
                f"Playback finished. Showing {final_date}."
            )
            if show_snapshot:
                snapshot_data = _prepare_snapshot_day(only_mnd)
                if snapshot_data is not None:
                    with snapshot_placeholder.container():
                        st.markdown("---")
                        st.subheader("Latest 24h snapshot")
                        snap_header = st.empty()
                        snap_map = st.empty()
                        snapshot_view = snapshot_data.copy()
                        snapshot_view["map_df"] = snapshot_data.get(
                            "history_df", pd.DataFrame()
                        )
                        snapshot_view["map_hours"] = 24
                        _render_history_day_header(
                            snap_header, snapshot_view
                        )
                        _render_history_day_map(
                            snap_map,
                            snapshot_view,
                            map_layer,
                            risk_df,
                            widget_ns="history_snapshot",
                        )
                else:
                    snapshot_placeholder.info(
                        "24h snapshot unavailable."
                    )
    else:
        status_placeholder.info(
            "Not enough history to play a timelapse."
        )
        snapshot_placeholder.empty()


def main() -> None:
    st.set_page_config(page_title="Gray-Zone Monitor", layout="wide")
    st.title("Gray-Zone Air & Sea Monitor")
    st.caption("Display bbox focus: CORE 118,20,123,26")
    snapshot_data = _load_latest_enriched()
    snapshot_risk = _load_snapshot_risk()
    history_days = _list_history_days()
    brief_text: str | None = None
    brief_path = EXAMPLES_DIR / "airops_brief_24h.md"
    if brief_path.exists():
        brief_text = brief_path.read_text(encoding="utf-8")
    monitor_tab, history_tab = st.tabs(("Monitor", "History"))
    with monitor_tab:
        render_monitor_tab(
            snapshot_data,
            snapshot_risk,
            history_days,
            brief_text,
        )
    with history_tab:
        render_history_tab(history_days)


if __name__ == "__main__":
    main()
