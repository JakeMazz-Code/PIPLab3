#!/usr/bin/env python3
"""Streamlit UI for the gray-zone monitor (read-only artifacts)."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
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

MAP_LAT = 23.5
MAP_LON = 121.0
MAP_ZOOM = 5.0
MAP_HEIGHT = 520

if pdk is not None:
    _DEFAULT_VIEW = pdk.ViewState(
        latitude=MAP_LAT, longitude=MAP_LON, zoom=MAP_ZOOM
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
        latitude=MAP_LAT, longitude=MAP_LON, zoom=MAP_ZOOM
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


def summarize_counts(df: pd.DataFrame) -> tuple[int, int, int]:
    if df.empty or "source" not in df.columns:
        return 0, 0, int(df.shape[0])
    series = df["source"].astype(str).str.upper().str.strip()
    os_mask = series.isin({"OS", "OPENSKY"})
    mnd_mask = series.eq("MND")
    return int(os_mask.sum()), int(mnd_mask.sum()), int(df.shape[0])



def _format_utc(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%MZ")



def _window_stats(
    df: pd.DataFrame, start: datetime, end: datetime
) -> tuple[datetime, datetime, float, float]:
    requested = max((end - start).total_seconds() / 3600, 0.0)
    if df.empty or "dt" not in df.columns:
        return start, end, requested, 0.0
    series = pd.to_datetime(df["dt"], utc=True, errors="coerce").dropna()
    if series.empty:
        return start, end, requested, 0.0
    actual_start = series.min().to_pydatetime().astimezone(timezone.utc)
    actual_end = series.max().to_pydatetime().astimezone(timezone.utc)
    if actual_start < start:
        actual_start = start
    if actual_end > end:
        actual_end = end
    if actual_end < actual_start:
        actual_end = actual_start
    actual = max((actual_end - actual_start).total_seconds() / 3600, 0.0)
    return actual_start, actual_end, requested, actual



def _filter_time_range(
    df: pd.DataFrame, start: datetime, end: datetime
) -> pd.DataFrame:
    if df.empty or "dt" not in df.columns:
        return df.iloc[0:0].copy()
    working = df.copy()
    working["dt"] = pd.to_datetime(working["dt"], utc=True, errors="coerce")
    working = working.dropna(subset=["dt"])
    mask = (working["dt"] >= start) & (working["dt"] < end)
    return working.loc[mask].copy()



def _filter_risk_range(
    df: pd.DataFrame, start: datetime, end: datetime
) -> pd.DataFrame:
    if df.empty:
        return df.iloc[0:0].copy()
    working = df.copy()
    dt_column = None
    for candidate in ("dt", "day", "date", "timestamp"):
        if candidate in working.columns:
            dt_series = pd.to_datetime(
                working[candidate], utc=True, errors="coerce"
            )
            if dt_series.notna().any():
                working["_window_dt"] = dt_series
                dt_column = candidate
                break
    if dt_column is None:
        return working
    mask = (working["_window_dt"] >= start) & (working["_window_dt"] < end)
    filtered = working.loc[mask].copy()
    return filtered.drop(columns=["_window_dt"])



def _dataset_bounds(
    df: pd.DataFrame,
) -> tuple[datetime | None, datetime | None]:
    if df.empty or "dt" not in df.columns:
        return None, None
    series = pd.to_datetime(df["dt"], utc=True, errors="coerce").dropna()
    if series.empty:
        return None, None
    return (
        series.min().to_pydatetime().astimezone(timezone.utc),
        series.max().to_pydatetime().astimezone(timezone.utc),
    )



def _window_bounds(
    now_utc: datetime,
    mode: str,
    days: int,
    default_hours: int,
    data_min: datetime | None = None,
) -> tuple[datetime, datetime, str]:
    mode_key = mode.lower()
    if mode_key == "24h":
        hours = 24
        label = "last 24 h"
    elif mode_key == "48h":
        hours = 48
        label = "last 48 h"
    elif mode_key == "ndays":
        days = max(1, days)
        hours = days * 24
        label = f"last {days} d"
    else:
        hours = max(1, default_hours)
        if hours % 24 == 0:
            label = f"last {hours // 24} d"
        else:
            label = f"last {hours} h"
    end = now_utc.astimezone(timezone.utc)
    start = end - timedelta(hours=hours)
    if data_min is not None:
        data_min_utc = data_min.astimezone(timezone.utc)
        if data_min_utc > start:
            start = data_min_utc
    if end < start:
        end = start
    return start, end, label



def _filter_by_grid(df: pd.DataFrame, grid_id: str | None) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    if grid_id is None:
        return df.copy()
    if df.empty or "grid_id" not in df.columns:
        return df.copy()
    series = df["grid_id"].astype(str).str.strip()
    return df.loc[series.eq(grid_id)].copy()


def _decode_log_bytes(payload: bytes) -> str | None:
    if payload is None:
        return None
    if not payload:
        return ""
    bom_map = {
        b"\xff\xfe": "utf-16",
        b"\xfe\xff": "utf-16",
        b"\xef\xbb\xbf": "utf-8-sig",
    }
    for marker, encoding in bom_map.items():
        if payload.startswith(marker):
            try:
                return payload.decode(encoding)
            except UnicodeDecodeError:
                return None
    encodings = ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "cp1252"]
    for encoding in encodings:
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None
def _latest_metrics_line() -> str | None:
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return None
    try:
        candidates = sorted(
            (path for path in logs_dir.iterdir() if path.is_file()),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return None
    for log_path in candidates:
        try:
            payload = log_path.read_bytes()
        except OSError:
            continue
        text = _decode_log_bytes(payload)
        if not text:
            continue
        for line in reversed(text.splitlines()):
            if line.startswith("METRICS |"):
                return line.strip()
    return None


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



def _prepare_point_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.iloc[0:0].copy()
    working = df.copy()
    working["lat"] = pd.to_numeric(working.get("lat"), errors="coerce")
    working["lon"] = pd.to_numeric(working.get("lon"), errors="coerce")
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
    return working



def _prepare_os_points(df_os: pd.DataFrame) -> pd.DataFrame:
    points = _prepare_point_dataframe(df_os)
    if points.empty:
        return points
    points["dt_label"] = _format_dt_labels(points.get("dt"))
    tooltip = "OS<br/>dt: " + points["dt_label"].fillna("")
    callsign = points.get("callsign")
    if callsign is not None:
        call_series = callsign.fillna("").astype(str).str.strip()
        mask = call_series.ne("")
        tooltip = tooltip.where(~mask, tooltip + "<br/>" + call_series)
    raw_text = points.get("raw_text")
    if raw_text is not None:
        raw_series = raw_text.fillna("").astype(str).str.strip()
        mask = raw_series.ne("")
        tooltip = tooltip.where(~mask, tooltip + "<br/>" + raw_series)
    points["tooltip"] = tooltip
    return points



def _prepare_mnd_points(df_mnd: pd.DataFrame) -> pd.DataFrame:
    points = _prepare_point_dataframe(df_mnd)
    if points.empty:
        return points
    points["dt_label"] = _format_dt_labels(points.get("dt"))
    tooltip = "MND<br/>dt: " + points["dt_label"].fillna("")
    summary = points.get("summary_one_line")
    raw_text = points.get("raw_text")
    actors = points.get("actors")
    for extra in (summary, raw_text):
        if extra is None:
            continue
        series = extra.fillna("").astype(str).str.strip()
        mask = series.ne("")
        tooltip = tooltip.where(~mask, tooltip + "<br/>" + series)
    if actors is not None:
        actor_series = actors.apply(_row_actors_text).fillna("").astype(str)
        mask = actor_series.ne("")
        tooltip = tooltip.where(~mask, tooltip + "<br/>" + actor_series)
    points["tooltip"] = tooltip
    return points



def _prepare_risk_layer(risk_df: pd.DataFrame) -> pd.DataFrame:
    if risk_df.empty:
        return risk_df.iloc[0:0].copy()
    working = risk_df.copy()
    working["risk_score"] = pd.to_numeric(
        working.get("risk_score"), errors="coerce"
    )
    working = working.dropna(subset=["risk_score"])
    for column in ("lat", "lon"):
        working[column] = pd.to_numeric(working.get(column), errors="coerce")
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
    working["weight"] = working["risk_score"].clip(lower=0.0, upper=1.0)
    return working



def _hex_source(
    os_points: pd.DataFrame, mnd_points: pd.DataFrame
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for frame in (os_points, mnd_points):
        if frame is None or frame.empty:
            continue
        frames.append(frame[["lat", "lon"]])
    if not frames:
        return pd.DataFrame(columns=["lat", "lon"])
    return pd.concat(frames, ignore_index=True)



@dataclass
class MapRenderResult:
    weights_present: bool
    effective_mode: str
    risk_cells: int
    fallback_reason: str | None



def _build_map_layers(
    layer_mode: str,
    df_os: pd.DataFrame,
    df_mnd: pd.DataFrame,
    risk_df: pd.DataFrame,
    show_os: bool,
    show_mnd: bool,
    use_hexagons: bool,
    starred: Iterable[str] | None,
    selected_grid: str | None,
) -> tuple[list[Any], str, bool, int, str | None]:
    layers: list[Any] = []
    os_points = _prepare_os_points(df_os)
    mnd_points = _prepare_mnd_points(df_mnd)
    risk_layer = _prepare_risk_layer(risk_df)
    risk_cells = int(risk_layer.shape[0]) if not risk_layer.empty else 0
    weights_present = not risk_layer.empty
    effective_mode = layer_mode
    fallback_reason: str | None = None

    if layer_mode == "Heatmap":
        if weights_present:
            layers.append(
                pdk.Layer(
                    "HeatmapLayer",
                    data=risk_layer,
                    get_position="[lon, lat]",
                    get_weight="weight",
                    radius_pixels=48,
                )
            )
        else:
            effective_mode = "Points"
            fallback_reason = (
                "Risk weights unavailable; falling back to points."
            )
    elif layer_mode == "Hexagons":
        if use_hexagons:
            hex_points = _hex_source(os_points, mnd_points)
            if not hex_points.empty:
                layers.append(
                    pdk.Layer(
                        "HexagonLayer",
                        data=hex_points,
                        get_position="[lon, lat]",
                        radius=20000,
                        elevation_scale=40,
                        elevation_range=[0, 2000],
                        extruded=True,
                        coverage=1.0,
                        pickable=True,
                    )
                )
            else:
                effective_mode = "Points"
                fallback_reason = (
                    "No coordinates for hexagon layer; showing points."
                )
        else:
            effective_mode = "Points"
            fallback_reason = "Hexagon layer disabled; showing points."

    if effective_mode == "Points":
        if show_os and not os_points.empty:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=os_points,
                    get_position="[lon, lat]",
                    get_radius=90,
                    radius_min_pixels=3,
                    radius_max_pixels=8,
                    get_fill_color=[255, 120, 60, 180],
                    get_line_color=[120, 60, 20, 200],
                    line_width_min_pixels=1,
                    pickable=True,
                    auto_highlight=True,
                )
            )
        if show_mnd and not mnd_points.empty:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=mnd_points,
                    get_position="[lon, lat]",
                    get_radius=120,
                    radius_min_pixels=4,
                    radius_max_pixels=10,
                    get_fill_color=[220, 200, 80, 220],
                    get_line_color=[150, 120, 40, 200],
                    line_width_min_pixels=1,
                    pickable=True,
                    auto_highlight=True,
                )
            )
    else:
        if show_os and not os_points.empty:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=os_points,
                    get_position="[lon, lat]",
                    get_radius=80,
                    radius_min_pixels=2,
                    radius_max_pixels=6,
                    get_fill_color=[255, 120, 60, 150],
                    pickable=True,
                )
            )
        if show_mnd and not mnd_points.empty:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=mnd_points,
                    get_position="[lon, lat]",
                    get_radius=110,
                    radius_min_pixels=3,
                    radius_max_pixels=8,
                    get_fill_color=[220, 200, 80, 190],
                    pickable=True,
                )
            )

    star_ids = list(starred or [])
    if star_ids:
        star_rows: list[dict[str, float | str]] = []
        for grid_id in star_ids:
            centroid = _grid_to_centroid(grid_id)
            if centroid is None:
                continue
            star_rows.append(
                {"grid_id": grid_id, "lat": centroid[0], "lon": centroid[1]}
            )
        if star_rows:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=pd.DataFrame(star_rows),
                    get_position="[lon, lat]",
                    get_radius=1,
                    radius_scale=6000,
                    radius_min_pixels=7,
                    filled=False,
                    get_line_color=[255, 255, 255, 230],
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
                    radius_scale=9000,
                    radius_min_pixels=9,
                    filled=False,
                    get_line_color=[255, 215, 0, 240],
                    line_width_min_pixels=3,
                    pickable=False,
                )
            )
    return layers, effective_mode, weights_present, risk_cells, fallback_reason



def _render_incident_map(
    df_os: pd.DataFrame,
    df_mnd: pd.DataFrame,
    risk_df: pd.DataFrame,
    layer_mode: str,
    show_os: bool,
    show_mnd: bool,
    starred: Iterable[str] | None,
    selected_grid: str | None,
    widget_ns: str,
    use_hexagons: bool,
) -> MapRenderResult:
    if pdk is None:
        st.info("pydeck unavailable; map rendering skipped.")
        return MapRenderResult(False, "Unavailable", 0, "pydeck unavailable")
    (
        layers,
        effective_mode,
        weights_present,
        risk_cells,
        message,
    ) = _build_map_layers(
        layer_mode,
        df_os,
        df_mnd,
        risk_df,
        show_os,
        show_mnd,
        use_hexagons,
        starred,
        selected_grid,
    )
    if not layers:
        st.info("No geolocated data for this window.")
        return MapRenderResult(weights_present, effective_mode, risk_cells, message)
    tooltip = {
        "html": "{tooltip}",
        "style": {"backgroundColor": "#0E1117", "color": "#FAFAFA"},
    }
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=_DEFAULT_VIEW,
        map_provider="carto",
        map_style="dark",
        tooltip=tooltip,
    )
    st.pydeck_chart(
        deck,
        use_container_width=True,
        height=MAP_HEIGHT,
        key=f"{widget_ns}_deck",
    )
    return MapRenderResult(weights_present, effective_mode, risk_cells, message)



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
    mode_key = "24h"
    default_hours = 24
    source_label = "snapshot"
    working_df = snapshot_df
    risk_working: pd.DataFrame | None = snapshot_risk_df
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
        mode_key = "ndays"
        default_hours = selected_days * 24
        working_df = _load_history_enriched(selected_days)
        risk_working = _load_history_risk(selected_days)
        source_label = f"history:{selected_days}d"
    else:
        if window_mode == "Last 48h":
            mode_key = "48h"
            default_hours = 48
        else:
            mode_key = "24h"
            default_hours = 24
        if working_df.empty:
            if history_max == 0:
                st.info(
                    "Snapshot unavailable and no history data present."
                )
                return
            selected_days = min(max(1, default_hours // 24), history_max)
            working_df = _load_history_enriched(selected_days)
            risk_working = _load_history_risk(selected_days)
            source_label = f"history:{selected_days}d (fallback)"
    if working_df.empty:
        st.info("No incidents available for the selected window.")
        return
    data_min, data_max = _dataset_bounds(working_df)
    now_utc = data_max or datetime.now(timezone.utc)
    start, end, label = _window_bounds(
        now_utc,
        mode_key,
        selected_days,
        default_hours,
        data_min,
    )
    window_df = _filter_time_range(working_df, start, end)
    if window_df.empty:
        st.info("No incidents available for the selected window.")
        return
    if isinstance(risk_working, pd.DataFrame):
        risk_source_df = risk_working
    else:
        risk_source_df = pd.DataFrame()
    risk_window = _filter_risk_range(risk_source_df, start, end)
    layer_mode = st.radio(
        "Layer mode",
        ("Points", "Heatmap", "Hexagons"),
        index=0,
        key=_wkey("mon", "layer_mode"),
    )
    toggle_cols = st.columns(2)
    show_os = toggle_cols[0].checkbox(
        "Show OpenSky points",
        value=True,
        key=_wkey("mon", "show_os"),
    )
    show_mnd = toggle_cols[1].checkbox(
        "Show MND markers",
        value=True,
        key=_wkey("mon", "show_mnd"),
    )
    use_hexagons = False
    if layer_mode == "Hexagons":
        use_hexagons = st.checkbox(
            "Use hexagons",
            value=True,
            key=_wkey("mon", "use_hexagons"),
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
    focus_options = ["All grids"] + grid_options
    focus_key = _wkey("mon", "focus")
    st.session_state.setdefault(focus_key, focus_options[0])
    if st.session_state[focus_key] not in focus_options:
        st.session_state[focus_key] = focus_options[0]
    focus_choice = st.selectbox(
        "Focus grid",
        focus_options,
        index=focus_options.index(st.session_state[focus_key]),
        key=focus_key,
    )
    selected_grid = None if focus_choice == "All grids" else focus_choice
    st.session_state["selected_grid"] = selected_grid
    filter_focus = st.checkbox(
        "Filter to focus grid",
        value=False,
        key=_wkey("mon", "filter_focus"),
    )
    filter_grid = selected_grid if filter_focus else None
    view_df = _filter_by_grid(window_df, filter_grid)
    risk_view = _filter_by_grid(risk_window, filter_grid)
    df_os, df_mnd = _split_sources(view_df)
    map_result = _render_incident_map(
        df_os,
        df_mnd,
        risk_view,
        layer_mode,
        show_os,
        show_mnd,
        starred,
        selected_grid,
        "monitor_map",
        use_hexagons,
    )
    if map_result.fallback_reason:
        st.caption(map_result.fallback_reason)
    if layer_mode == "Points" and map_result.risk_cells > 0:
        st.caption(
            f"Try 'Heatmap' to see weighted risk (cells={map_result.risk_cells})."
        )
    actual_start, actual_end, requested_hours, actual_hours = _window_stats(
        view_df, start, end
    )
    os_count, mnd_count, total = summarize_counts(view_df)
    header_text = (
        f"Window: {_format_utc(actual_start)} -> {_format_utc(actual_end)} "
        f"({label}; rows: {total} (OS: {os_count} / MND: {mnd_count}))"
    )
    st.caption(header_text)
    if (
        window_mode in {"Last 24h", "Last 48h"}
        and actual_hours + 0.1 < requested_hours
    ):
        st.caption(
            f"Only {actual_hours:.1f} h available in snapshot; clamped."
        )
    risk_rows = int(risk_view.shape[0]) if not risk_view.empty else 0
    source_caption = (
        f"Monitor data: source={source_label} | rows={total} "
        f"| OS: {os_count} | MND: {mnd_count} | risk_rows={risk_rows} "
        f"| Risk weights: {'yes' if map_result.weights_present else 'no'}"
    )
    st.caption(source_caption)
    if source_label.startswith("snapshot") and snapshot_path is not None:
        source_note = snapshot_path.name
    else:
        source_note = source_label
    st.caption(
        "OpenSky points: raw aircraft state vectors in the window. "
        "MND markers: Taiwan MND daily bulletin centroids. "
        "Heatmap/Hexagons use risk weights derived from MND text enrichment. "
        f"Source file: {source_note}"
    )
    st.subheader("Watch cells")
    watch_df = _watch_cells(view_df, 0.35, True, top_k=5)
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
        st.caption("Table truncated for display (showing first 1000 rows).")
    if not table_to_show.empty:
        st.dataframe(table_to_show[columns])
    else:
        st.info("No MND incidents in the selected window.")
    st.subheader("MND anomaly chart")
    render_anomaly_chart(mnd_table)
    st.subheader("LLM brief (24h)")
    render_brief(mnd_table, brief_text)
    with st.expander("How this works / Data & Metrics"):
        st.markdown(
            "**How data is collected**\n\n"
            "- **OpenSky**: recent state vectors within the extraction bbox; "
            "shown as OS points when available.\n"
            "- **MND bulletins**: daily Taiwan MND summaries parsed offline; "
            "DeepSeek enrichment augments detail when configured.\n"
            "- **Risk grid**: daily risk CSV powers heatmap/hex layers; "
            "if weights are missing the view falls back to points.\n\n"
            "**METRICS fields**\n"
            "- `opensky_points`, `mnd_rows`, `merged_rows`, `enriched_rows` "
            "track ETL volume.\n"
            "- `llm_success`, `llm_invalid_json`, `llm_retries` show AI parse "
            "health.\n"
            "- `needs_review_count`, `validation_sparse_fallbacks`, "
            "`os_anom_rows`, `wall_ms` cover QA and timing."
        )
        metrics_line = _latest_metrics_line()
        if metrics_line:
            st.code(metrics_line)
        else:
            st.caption("Metrics unavailable.")




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
    risk_table = _load_history_risk(days)
    data_min, data_max = _dataset_bounds(history_df)
    now_utc = data_max or datetime.now(timezone.utc)
    start, end, label = _window_bounds(
        now_utc,
        "ndays",
        days,
        days * 24,
        data_min,
    )
    window_df = _filter_time_range(history_df, start, end)
    risk_window = _filter_risk_range(risk_table, start, end)
    layer_mode = st.radio(
        "Layer mode",
        ("Points", "Heatmap", "Hexagons"),
        index=0,
        key=_wkey("hist", "layer_mode"),
    )
    toggle_cols = st.columns(2)
    show_os = toggle_cols[0].checkbox(
        "Show OpenSky points",
        value=True,
        key=_wkey("hist", "show_os"),
    )
    show_mnd = toggle_cols[1].checkbox(
        "Show MND markers",
        value=True,
        key=_wkey("hist", "show_mnd"),
    )
    use_hexagons = False
    if layer_mode == "Hexagons":
        use_hexagons = st.checkbox(
            "Use hexagons",
            value=True,
            key=_wkey("hist", "use_hexagons"),
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
    starred = st.multiselect(
        "Star grids",
        options=grid_options,
        default=defaults,
        key=_wkey("hist", "starred"),
    )
    if set(starred) != set(saved_watchlist):
        _save_watchlist(WATCHLIST_PATH, starred)
    focus_options = ["All grids"] + grid_options
    focus_key = _wkey("hist", "focus")
    st.session_state.setdefault(focus_key, focus_options[0])
    if st.session_state[focus_key] not in focus_options:
        st.session_state[focus_key] = focus_options[0]
    focus_choice = st.selectbox(
        "Focus grid",
        focus_options,
        index=focus_options.index(st.session_state[focus_key]),
        key=focus_key,
    )
    selected_grid = None if focus_choice == "All grids" else focus_choice
    st.session_state["selected_grid"] = selected_grid
    filter_focus = st.checkbox(
        "Filter to focus grid",
        value=False,
        key=_wkey("hist", "filter_focus"),
    )
    filter_grid = selected_grid if filter_focus else None
    view_df = _filter_by_grid(window_df, filter_grid)
    risk_view = _filter_by_grid(risk_window, filter_grid)
    df_os, df_mnd = _split_sources(view_df)
    map_result = _render_incident_map(
        df_os,
        df_mnd,
        risk_view,
        layer_mode,
        show_os,
        show_mnd,
        starred,
        selected_grid,
        "history_map",
        use_hexagons,
    )
    if map_result.fallback_reason:
        st.caption(map_result.fallback_reason)
    if layer_mode == "Points" and map_result.risk_cells > 0:
        st.caption(
            f"Try 'Heatmap' to see weighted risk (cells={map_result.risk_cells})."
        )
    actual_start, actual_end, _, _ = _window_stats(
        view_df, start, end
    )
    os_count, mnd_count, total = summarize_counts(view_df)
    header_text = (
        f"Window: {_format_utc(actual_start)} -> {_format_utc(actual_end)} "
        f"({label}; rows: {total} (OS: {os_count} / MND: {mnd_count}))"
    )
    st.caption(header_text)
    risk_rows = int(risk_view.shape[0]) if not risk_view.empty else 0
    history_caption = (
        f"History window: days={days} | rows={total} "
        f"| OS: {os_count} | MND: {mnd_count} | risk_rows={risk_rows} "
        f"| Risk weights: {'yes' if map_result.weights_present else 'no'}"
    )
    st.caption(history_caption)
    latest_note = history_days[-1] if history_days else "n/a"
    st.caption(
        "OpenSky points: raw aircraft state vectors in the window. "
        "MND markers: Taiwan MND daily bulletin centroids. "
        "Heatmap/Hexagons use risk weights derived from MND text enrichment. "
        f"History snapshot: {latest_note}"
    )
    st.subheader("Watch cells")
    watch_df = _watch_cells(view_df, 0.35, True, top_k=5)
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
        history_table["summary_one_line"] = history_table["summary_one_line"].apply(
            _friendly_summary
        )
        if len(history_table) > MAX_TABLE_ROWS:
            history_table = history_table.iloc[:MAX_TABLE_ROWS]
            st.caption("Table truncated for display (showing first 1000 rows).")
        st.dataframe(history_table[columns])
    st.subheader("MND anomaly chart")
    render_anomaly_chart(df_mnd)



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
