"""Main ETL pipeline for Taiwan gray-zone monitoring."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from deepseek_enrichment import (
    enrich_incidents,
    get_llm_metrics,
    reset_llm_metrics,
    summarize_theater,
)


logging.basicConfig(
    level=os.getenv("GRAYZONE_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
ENRICHED_DIR = BASE_DIR / "data" / "enriched"
EXAMPLES_DIR = BASE_DIR / "examples"

BBOX_CORE = [118.0, 20.0, 123.0, 26.0]
BBOX_WIDE = [115.0, 18.0, 126.0, 28.0]
BBOX_MAX = [112.0, 16.0, 128.0, 30.0]
DEFAULT_BBOX = BBOX_CORE
GRID_STEP = 0.5
AUTOFALLBACK_THRESHOLD = 2000
OPENSKY_URL = "https://opensky-network.org/api/states/all"
MND_LIST_URL = "https://www.mnd.gov.tw/PublishTable.aspx"
MND_PARAMS = {
    "Types": "\u5373\u6642\u8ecd\u4e8b\u52d5\u614b",
    "title": "\u570b\u9632\u6d88\u606f",
}
REQUEST_TIMEOUT = 20

TAIWAN_LAT_MIN = 20.0
TAIWAN_LAT_MAX = 26.0
TAIWAN_LON_MIN = 118.0
TAIWAN_LON_MAX = 123.0


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically write *data* to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='wb', delete=False, dir=path.parent, suffix='.tmp'
    ) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    try:
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise



def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a parquet file atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        delete=False, dir=path.parent, suffix='.tmp'
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.to_parquet(tmp_path, index=False)
        with tmp_path.open('rb') as handle:
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise



def _atomic_write_df_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV file atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', newline='', delete=False, dir=path.parent,
        suffix='.tmp'
    ) as tmp:
        df.to_csv(tmp, index=False)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    try:
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise



def _atomic_write_text(path: Path, text: str, encoding: str = 'utf-8') -> None:
    """Atomically write text to *path*."""
    data = text.encode(encoding)
    _atomic_write_bytes(path, data)



def _atomic_write_json(path: Path, data: Any) -> None:
    """Atomically write JSON data to *path*."""
    payload = json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8')
    _atomic_write_bytes(path, payload)



def _ensure_directories() -> None:
    """Create required directories if they are missing."""
    for path in (RAW_DIR, ENRICHED_DIR, EXAMPLES_DIR):
        path.mkdir(parents=True, exist_ok=True)


_ensure_directories()



def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)



def _timestamp() -> str:
    """Return compact timestamp string for filenames."""
    return _now_utc().strftime("%Y%m%d%H")



def _write_text(path: Path, content: str, encoding: str = 'utf-8') -> None:
    """Write *content* to *path* with safe encoding."""
    _atomic_write_text(path, content, encoding=encoding)



def _write_json(path: Path, data: Any) -> None:
    """Write JSON data with UTF-8 encoding."""
    _atomic_write_json(path, data)


def _parse_float(value: Any) -> float | None:
    """Parse a float from arbitrary value."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def latlon_to_grid(
    lat: float | None,
    lon: float | None,
    step: float = GRID_STEP,
) -> str | None:
    """Map latitude/longitude to a 0.5° grid identifier."""
    if lat is None or lon is None:
        return None
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return None
    row = math.floor((lat + 90.0) / step)
    col = math.floor((lon + 180.0) / step)
    return f"R{row}C{col}"


def _compute_grid_id(lat: float | None, lon: float | None) -> str:
    """Backwards-compatible wrapper for grid ID computation."""
    grid = latlon_to_grid(lat, lon)
    return grid or "RNaNCNaN"


def _lat_to_row(lat: float, step: float = GRID_STEP) -> int:
    """Convert latitude to grid row index."""
    return math.floor((lat + 90.0) / step)


def _lon_to_col(lon: float, step: float = GRID_STEP) -> int:
    """Convert longitude to grid column index."""
    return math.floor((lon + 180.0) / step)


GRID_EPSILON = 1e-9
CORE_ROW_MIN = _lat_to_row(TAIWAN_LAT_MIN)
CORE_ROW_MAX = _lat_to_row(TAIWAN_LAT_MAX - GRID_EPSILON)
CORE_COL_MIN = _lon_to_col(TAIWAN_LON_MIN)
CORE_COL_MAX = _lon_to_col(TAIWAN_LON_MAX - GRID_EPSILON)
NEIGHBOR_RADIUS_CELLS = int(round(1.0 / GRID_STEP))
CORE_ROW_MARGIN_MIN = CORE_ROW_MIN - NEIGHBOR_RADIUS_CELLS
CORE_ROW_MARGIN_MAX = CORE_ROW_MAX + NEIGHBOR_RADIUS_CELLS
CORE_COL_MARGIN_MIN = CORE_COL_MIN - NEIGHBOR_RADIUS_CELLS
CORE_COL_MARGIN_MAX = CORE_COL_MAX + NEIGHBOR_RADIUS_CELLS


def _grid_to_indices(grid_id: str | None) -> tuple[int | None, int | None]:
    """Parse a grid identifier into row/column indices."""
    if not grid_id or not isinstance(grid_id, str):
        return None, None
    if not grid_id.startswith("R") or "C" not in grid_id:
        return None, None
    try:
        row_part, col_part = grid_id[1:].split("C", 1)
        return int(row_part), int(col_part)
    except (ValueError, TypeError):
        return None, None


def mnd_where_to_grid(where_guess: str | None, step: float = 0.5) -> str:
    """Approximate an MND grid cell from a textual location hint."""
    if not where_guess:
        return "RNaNCNaN"
    text = where_guess.lower()
    center_lat = (TAIWAN_LAT_MIN + TAIWAN_LAT_MAX) / 2
    center_lon = (TAIWAN_LON_MIN + TAIWAN_LON_MAX) / 2
    lat = center_lat
    lon = center_lon
    has_direction = False
    median_tokens = (
        "median line",
        "median-line",
        "medianline",
        "中線",
        "中线",
    )
    north_tokens = ("north", "northern", "北")
    south_tokens = ("south", "southern", "南")
    east_tokens = ("east", "eastern", "東", "东")
    west_tokens = ("west", "western", "西")
    if any(token in text for token in north_tokens):
        lat = min(TAIWAN_LAT_MAX - step / 2, TAIWAN_LAT_MAX - 0.25)
        has_direction = True
    if any(token in text for token in south_tokens):
        lat = max(TAIWAN_LAT_MIN + step / 2, TAIWAN_LAT_MIN + 0.25)
        has_direction = True
    if any(token in text for token in east_tokens):
        lon = min(TAIWAN_LON_MAX - step / 2, TAIWAN_LON_MAX - 0.25)
        has_direction = True
    if any(token in text for token in west_tokens):
        lon = max(TAIWAN_LON_MIN + step / 2, TAIWAN_LON_MIN + 0.25)
        has_direction = True
    if has_direction:
        return _compute_grid_id(lat, lon)
    if any(token in text for token in median_tokens):
        lat = center_lat
        lon = 121.0
        return _compute_grid_id(lat, lon)
    return "RNaNCNaN"


def default_mnd_where_guess(timestamp: Any, position: int) -> str:
    """Return a deterministic textual guess for MND incidents."""
    directions = ("north", "south", "east", "west")
    if isinstance(timestamp, pd.Timestamp) and not pd.isna(timestamp):
        index = int(timestamp.day) % len(directions)
    else:
        index = position % len(directions)
    return f"{directions[index]} median line"


def _roc_to_datetime(text: str) -> datetime | None:
    """Convert ROC calendar date (e.g., 114/09/17) to UTC datetime."""
    try:
        year_part, month_part, day_part = text.strip().split("/")
        year = int(year_part) + 1911
        month = int(month_part)
        day = int(day_part)
        return datetime(year, month, day, tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return None


def _to_builtin(value: Any) -> Any:
    """Convert numpy or pandas scalar types to plain Python types."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover
            return str(value)
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return str(value)


def _chunked(iterable: Iterable[Any], size: int) -> list[list[Any]]:
    """Chunk *iterable* into lists of length *size*."""
    bucket: list[Any] = []
    chunks: list[list[Any]] = []
    for item in iterable:
        bucket.append(item)
        if len(bucket) >= size:
            chunks.append(bucket)
            bucket = []
    if bucket:
        chunks.append(bucket)
    return chunks


def _opensky_auth() -> tuple[str, str] | None:
    """Return optional OpenSky basic auth from environment."""
    user = os.getenv("OPENSKY_USER")
    password = os.getenv("OPENSKY_PASS")
    if user and password:
        return user, password
    return None


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _fetch_opensky(params: dict[str, Any]) -> dict[str, Any]:
    """Fetch OpenSky states with retries."""
    auth = _opensky_auth()
    response = requests.get(
        OPENSKY_URL,
        params=params,
        auth=auth,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def extract_opensky(
    hours: int = 6,
    bbox: list | None = None,
) -> pd.DataFrame:
    """Fetch OpenSky states and return normalized DataFrame."""
    attempts: list[list[float]]
    if bbox is None:
        attempts = [BBOX_WIDE, BBOX_MAX]
    else:
        attempts = [bbox]
    data: dict[str, Any] | None = None
    states: list[list[Any]] = []
    for attempt_index, current_bbox in enumerate(attempts):
        params = {
            "lamin": current_bbox[1],
            "lomax": current_bbox[3],
            "lamax": current_bbox[2],
            "lomin": current_bbox[0],
        }
        try:
            data = _fetch_opensky(params)
        except requests.RequestException as exc:
            logger.error("OpenSky fetch failed: %s", exc)
            if attempt_index == len(attempts) - 1:
                return pd.DataFrame(columns=[
                    "dt",
                    "lat",
                    "lon",
                    "source",
                    "raw_text",
                    "country",
                    "grid_id",
                ])
            continue
        states = data.get("states") or []
        if (
            bbox is None
            and attempt_index == 0
            and len(states) < AUTOFALLBACK_THRESHOLD
            and len(attempts) > 1
        ):
            logger.info(
                "OpenSky sparse (%s points) with WIDE bbox; retrying with MAX.",
                len(states),
            )
            continue
        break
    if data is None:
        return pd.DataFrame(columns=[
            "dt",
            "lat",
            "lon",
            "source",
            "raw_text",
            "country",
            "grid_id",
        ])
    raw_path = RAW_DIR / f"opensky_{_timestamp()}.json"
    _write_json(raw_path, data)
    window_start = _now_utc() - timedelta(hours=hours)
    records: list[dict[str, Any]] = []
    for entry in states:
        if not isinstance(entry, list) or len(entry) < 17:
            continue
        lon = _parse_float(entry[5])
        lat = _parse_float(entry[6])
        last_contact = entry[4]
        if last_contact is None:
            continue
        dt = datetime.fromtimestamp(last_contact, tz=timezone.utc)
        if dt < window_start:
            continue
        grid_id = latlon_to_grid(lat, lon)
        record = {
            "dt": dt,
            "lat": lat,
            "lon": lon,
            "source": "OpenSky",
            "raw_text": f"icao24={entry[0]} callsign={entry[1]}",
            "country": entry[2],
            "grid_id": grid_id or "RNaNCNaN",
            "icao24": entry[0],
            "callsign": (entry[1] or "").strip(),
            "velocity": _parse_float(entry[9]),
            "heading": _parse_float(entry[10]),
        }
        records.append(record)
    if not records:
        return pd.DataFrame(columns=[
            "dt",
            "lat",
            "lon",
            "source",
            "raw_text",
            "country",
            "grid_id",
        ])
    df = pd.DataFrame(records)
    df["dt"] = pd.to_datetime(df["dt"], utc=True)
    return df


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _fetch_mnd_html() -> str:
    """Retrieve the MND bulletin list HTML."""
    response = requests.get(
        MND_LIST_URL,
        params=MND_PARAMS,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.text


def _parse_mnd_table(html: str) -> list[dict[str, Any]]:
    """Parse the MND bulletin list into structured rows."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return []
    rows: list[dict[str, Any]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 2:
            continue
        date_text = cells[0].get_text(strip=True)
        title = cells[1].get_text(" ", strip=True)
        dt_parsed = _roc_to_datetime(date_text)
        raw_html = str(tr)
        rows.append({
            "dt": dt_parsed,
            "title": title,
            "raw_html": raw_html,
        })
    return rows


def scrape_mnd() -> list[dict]:
    """Scrape Taiwan MND daily PLA activities bulletins."""
    try:
        html = _fetch_mnd_html()
    except requests.RequestException as exc:
        logger.error("MND fetch failed: %s", exc)
        return []
    raw_path = RAW_DIR / f"mnd_{_timestamp()}.html"
    _write_text(raw_path, html)
    rows = _parse_mnd_table(html)
    results: list[dict[str, Any]] = []
    for row in rows:
        dt_value = row.get("dt")
        if dt_value is None:
            continue
        title = row["title"]
        results.append({
            "dt": dt_value,
            "lat": None,
            "lon": None,
            "source": "MND",
            "raw_text": title,
            "country": "CN",
            "grid_id": "RNaNCNaN",
            "where_guess": None,
            "extra": {
                "raw_html": row["raw_html"],
            },
        })
    return results


def clean_merge(
    os_df: pd.DataFrame,
    mnd_rows: list[dict],
) -> pd.DataFrame:
    """Combine OpenSky and MND data into a canonical frame."""
    os_cols = [
        "dt",
        "lat",
        "lon",
        "source",
        "raw_text",
        "country",
        "grid_id",
        "icao24",
        "callsign",
        "velocity",
        "heading",
        "where_guess",
    ]
    os_clean = os_df.copy()
    for column in os_cols:
        if column not in os_clean.columns:
            os_clean[column] = None
    mnd_df = pd.DataFrame(mnd_rows)
    if not mnd_df.empty:
        mnd_df["dt"] = pd.to_datetime(mnd_df["dt"], utc=True)
        for column in ["icao24", "callsign", "velocity", "heading"]:
            mnd_df[column] = None
        if "where_guess" not in mnd_df.columns:
            mnd_df["where_guess"] = None
        missing_guess = mnd_df["where_guess"].isna() | (
            mnd_df["where_guess"].astype(str).str.strip() == ""
        )
        if missing_guess.any():
            fills = [
                default_mnd_where_guess(dt_value, position)
                for position, dt_value in enumerate(
                    mnd_df.loc[missing_guess, "dt"]
                )
            ]
            mnd_df.loc[missing_guess, "where_guess"] = fills
        needs_guess = (
            mnd_df["grid_id"].isna() | (mnd_df["grid_id"] == "RNaNCNaN")
        )
        if needs_guess.any():
            mnd_df.loc[needs_guess, "grid_id"] = mnd_df.loc[
                needs_guess, "where_guess"
            ].apply(mnd_where_to_grid)
    combined = pd.concat(
        [os_clean[os_cols], mnd_df[os_cols]],
        ignore_index=True,
    )
    combined["dt"] = pd.to_datetime(combined["dt"], utc=True)
    return combined


def _prepare_validation(
    os_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-grid and per-hour OpenSky densities."""
    if os_df.empty:
        empty_grid = pd.DataFrame(columns=["grid_id", "hour", "count"])
        empty_hour = pd.DataFrame(columns=["hour", "count"])
        return empty_grid, empty_hour
    frame = os_df.copy()
    frame["hour"] = frame["dt"].dt.floor("h")
    grid_density = (
        frame.groupby(["grid_id", "hour"])
        .size()
        .reset_index(name="count")
    )
    hour_density = (
        frame.groupby("hour")
        .size()
        .reset_index(name="count")
    )
    return grid_density, hour_density


def _assign_mnd_grids_from_guess(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with grid IDs derived from textual hints for MND rows."""
    if df.empty:
        return df
    updated = df.copy()
    if "where_guess" not in updated.columns:
        updated["where_guess"] = None
    mnd_mask = updated["source"] == "MND"
    if not mnd_mask.any():
        return updated
    missing_guess = mnd_mask & (
        updated["where_guess"].isna()
        | (updated["where_guess"].astype(str).str.strip() == "")
    )
    if missing_guess.any():
        dt_series = updated.loc[missing_guess, "dt"]
        fills = [
            default_mnd_where_guess(dt_value, position)
            for position, dt_value in enumerate(dt_series)
        ]
        updated.loc[missing_guess, "where_guess"] = fills
    needs_grid = mnd_mask & (
        updated["grid_id"].isna() | (updated["grid_id"] == "RNaNCNaN")
    )
    if needs_grid.any():
        updated.loc[needs_grid, "grid_id"] = updated.loc[
            needs_grid, "where_guess"
        ].apply(mnd_where_to_grid)
    return updated


def _apply_validation(
    mnd_df: pd.DataFrame,
    grid_density: pd.DataFrame,
    hour_density: pd.DataFrame,
    metrics: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Attach validation scores to MND incidents."""
    if metrics is None:
        metrics = {}
    fallback_key = "validation_sparse_fallbacks"
    metrics.setdefault(fallback_key, 0)
    if mnd_df.empty:
        mnd_df["validation_score"] = 1.0
        mnd_df["corroborations"] = [[] for _ in range(len(mnd_df))]
        return mnd_df
    scores: list[float] = []
    corroborations: list[list[str]] = []
    fallback_count = metrics[fallback_key]
    grid_counts: dict[tuple[str, pd.Timestamp], float] = {}
    if not grid_density.empty:
        local = grid_density.copy()
        local["hour"] = pd.to_datetime(local["hour"], utc=True)
        grid_counts = {
            (row.grid_id, row.hour): float(row.count)
            for row in local.itertuples()
        }
    core_counts = pd.Series(dtype=float)
    core_mean = None
    core_std = None
    if not hour_density.empty:
        hour_density = hour_density.copy()
        hour_density["hour"] = pd.to_datetime(hour_density["hour"], utc=True)
        core_counts = hour_density.set_index("hour")["count"].astype(float)
        core_mean = core_counts.mean()
        core_std = core_counts.std(ddof=0)
    for position, row in enumerate(mnd_df.itertuples()):
        dt = getattr(row, "dt", None)
        grid = getattr(row, "grid_id", None)
        if not isinstance(dt, pd.Timestamp):
            fallback_count += 1
            scores.append(1.0)
            corroborations.append([])
            continue
        hour = dt.floor("h")
        row_idx, col_idx = _grid_to_indices(grid)
        if (
            row_idx is None
            or col_idx is None
            or row_idx < CORE_ROW_MARGIN_MIN
            or row_idx > CORE_ROW_MARGIN_MAX
            or col_idx < CORE_COL_MARGIN_MIN
            or col_idx > CORE_COL_MARGIN_MAX
        ):
            fallback_count += 1
            scores.append(1.0)
            corroborations.append([])
            continue
        neighbors = [
            f"R{r}C{c}"
            for r in range(
                row_idx - NEIGHBOR_RADIUS_CELLS,
                row_idx + NEIGHBOR_RADIUS_CELLS + 1,
            )
            for c in range(
                col_idx - NEIGHBOR_RADIUS_CELLS,
                col_idx + NEIGHBOR_RADIUS_CELLS + 1,
            )
        ]
        neighbor_values = [
            grid_counts.get((neighbor, hour), 0.0)
            for neighbor in neighbors
        ]
        neighbor_total = float(sum(neighbor_values))
        if neighbor_total >= 15:
            counts_array = np.array(neighbor_values, dtype=float)
            mean = counts_array.mean()
            std = counts_array.std(ddof=0)
            target_count = grid_counts.get((grid, hour), 0.0)
            if std == 0 or np.isnan(std):
                z_score = 0.0
            else:
                z_score = (target_count - mean) / std
            validation = max(0.1, min(2.0, 1.0 + z_score))
            scores.append(validation)
            tag = f"OS_ANOM:{z_score:.2f}"
            corroborations.append([tag])
            continue
        if (
            not core_counts.empty
            and hour in core_counts.index
            and core_mean is not None
        ):
            total_count = core_counts.loc[hour]
            if core_std is None or np.isnan(core_std) or core_std == 0:
                z_total = 0.0
            else:
                z_total = (total_count - core_mean) / core_std
            validation = max(0.1, min(2.0, 1.0 + z_total))
            scores.append(validation)
            tag = f"OS_ANOM:{z_total:.2f}"
            corroborations.append([tag])
            continue
        fallback_count += 1
        scores.append(1.0)
        corroborations.append([])
    metrics[fallback_key] = fallback_count
    mnd_df["validation_score"] = scores
    mnd_df["corroborations"] = corroborations
    return mnd_df


def _write_enriched(df: pd.DataFrame, prefix: str | None) -> Path:
    """Persist enriched DataFrame to parquet (CSV fallback)."""
    stamp = _timestamp()
    label = prefix or "incidents"
    parquet_path = ENRICHED_DIR / f"{label}_enriched_{stamp}.parquet"
    try:
        _atomic_write_parquet(df, parquet_path)
        return parquet_path
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        logger.warning("Parquet write failed (%s); falling back to CSV", exc)
        csv_path = parquet_path.with_suffix(".csv")
        _atomic_write_df_csv(df, csv_path)
        return csv_path


def _write_daily_grid_risk(df: pd.DataFrame) -> Path:
    """Write daily grid risk summary CSV."""
    path = ENRICHED_DIR / "daily_grid_risk.csv"
    if df.empty or "risk_score" not in df.columns:
        empty = pd.DataFrame(columns=["day", "grid_id", "risk_score"])
        _atomic_write_df_csv(empty, path)
        return path
    enriched = df.copy()
    enriched["dt"] = pd.to_datetime(enriched["dt"], utc=True)
    enriched = enriched.dropna(subset=["dt"])
    if enriched.empty:
        empty = pd.DataFrame(columns=["day", "grid_id", "risk_score"])
        _atomic_write_df_csv(empty, path)
        return path
    enriched["day"] = enriched["dt"].dt.floor("D")
    grouped = (
        enriched.groupby(["day", "grid_id"])["risk_score"]
        .mean()
        .reset_index()
    )
    _atomic_write_df_csv(grouped, path)
    return path


def _latest_enriched() -> Path | None:
    """Return the most recent enriched file path."""
    candidates = sorted(ENRICHED_DIR.glob("*_enriched_*.parquet"))
    if candidates:
        return candidates[-1]
    candidates = sorted(ENRICHED_DIR.glob("*_enriched_*.csv"))
    if candidates:
        return candidates[-1]
    return None


def _load_enriched(path: Path) -> pd.DataFrame:
    """Load enriched dataset from parquet or CSV."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["dt"], infer_datetime_format=True)


def _build_examples(df: pd.DataFrame, summary: str) -> None:
    """Create example files showing enrichment output."""
    if df.empty:
        return
    mnd_sample = df[df["source"] == "MND"].head(1)
    if not mnd_sample.empty:
        row = mnd_sample.iloc[0]
        before_after = EXAMPLES_DIR / "mnd_before_after_001.md"
        content = (
            "# MND Bulletin Enrichment\n\n"
            f"**Raw snippet**: {row.get('raw_text','')}\n\n"
            "**Enriched JSON**:\n\n"
            "```json\n"
            f"{json.dumps({
                'category': _to_builtin(row.get('category')),
                'severity_0_5': _to_builtin(row.get('severity_0_5')),
                'risk_score': _to_builtin(row.get('risk_score')),
                'actors': _to_builtin(row.get('actors')),
                'geo_quality': _to_builtin(row.get('geo_quality')),
                'summary_one_line': _to_builtin(row.get('summary_one_line')),
            }, ensure_ascii=False, indent=2)}\n"
            "```\n"
        )
        _write_text(before_after, content)
    summary_path = EXAMPLES_DIR / "airops_brief_24h.md"
    _write_text(summary_path, summary)


def _run_pipeline(hours: int, bbox: list | None, prefix: str | None) -> None:
    """Execute full ETL + enrichment pipeline."""
    start_time = time.perf_counter()
    logger.info("Starting extraction phase")
    os_df = extract_opensky(hours=hours, bbox=bbox)
    mnd_rows = scrape_mnd()
    metrics: dict[str, int] = {
        "opensky_points": len(os_df),
        "mnd_rows": len(mnd_rows),
        "validation_sparse_fallbacks": 0,
    }
    merged = clean_merge(os_df, mnd_rows)
    merged = _assign_mnd_grids_from_guess(merged)
    metrics["merged_rows"] = len(merged)
    grid_density, hour_density = _prepare_validation(os_df)
    mnd_df = merged[merged["source"] == "MND"].copy()
    mnd_df = _apply_validation(mnd_df, grid_density, hour_density, metrics)
    reset_llm_metrics()
    logger.info("Enriching %s incidents via DeepSeek", len(mnd_df))
    enriched_mnd = enrich_incidents(mnd_df)
    enriched_mnd = _assign_mnd_grids_from_guess(enriched_mnd)
    os_only = merged[merged["source"] != "MND"].copy()
    combined = pd.concat([os_only, enriched_mnd], ignore_index=True)
    combined = _assign_mnd_grids_from_guess(combined)
    combined["dt"] = pd.to_datetime(combined["dt"], utc=True)
    enriched_path = _write_enriched(combined, prefix)
    logger.info("Enriched dataset written to %s", enriched_path)
    grid_risk_path = _write_daily_grid_risk(combined)
    logger.info("Daily grid risk written to %s", grid_risk_path)
    summary = summarize_theater(enriched_mnd, horizon="24h")
    _build_examples(enriched_mnd, summary)
    metrics["enriched_rows"] = len(enriched_mnd)
    needs_review_count = 0
    if "needs_review" in enriched_mnd.columns:
        review_series = enriched_mnd["needs_review"].fillna(False).astype(bool)
        needs_review_count = int(review_series.sum())
    metrics["needs_review_count"] = needs_review_count
    llm_metrics = get_llm_metrics()
    metrics["llm_total_calls"] = llm_metrics.total_calls
    metrics["llm_success"] = llm_metrics.success
    metrics["llm_invalid_json"] = llm_metrics.invalid_json
    metrics["llm_retries"] = llm_metrics.retries
    metrics["llm_fallbacks"] = llm_metrics.fallbacks
    os_anom_rows = 0
    if "corroborations" in combined.columns:
        mnd_mask = combined["source"].eq("MND")
        if mnd_mask.any():
            corrs = combined.loc[mnd_mask, "corroborations"]

            def _contains_os_anom(value: Any) -> bool:
                if isinstance(value, str):
                    return "OS_ANOM" in value
                if isinstance(value, (list, tuple, set)):
                    return any("OS_ANOM" in str(item) for item in value)
                return "OS_ANOM" in str(value)

            os_anom_rows = int(corrs.apply(_contains_os_anom).sum())
    metrics["os_anom_rows"] = os_anom_rows
    metrics["wall_ms"] = int((time.perf_counter() - start_time) * 1000)
    metrics_line = (
        "METRICS | opensky_points={opensky_points} "
        "mnd_rows={mnd_rows} merged_rows={merged_rows} "
        "enriched_rows={enriched_rows} llm_success={llm_success} "
        "llm_invalid_json={llm_invalid_json} llm_retries={llm_retries} "
        "needs_review_count={needs_review_count} "
        "validation_sparse_fallbacks={validation_sparse_fallbacks} "
        "os_anom_rows={os_anom_rows} wall_ms={wall_ms}"
    ).format(**metrics)
    print(metrics_line)



def simulate_air_ops(
    df_enriched: pd.DataFrame,
    n_runs: int = 100,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate air operations response times based on risk profile."""
    rng = np.random.default_rng(seed)
    if df_enriched.empty or "risk_score" not in df_enriched.columns:
        baseline = 0.4
    else:
        risk_series = pd.to_numeric(
            df_enriched["risk_score"],
            errors="coerce",
        ).fillna(0.4)
        baseline = float(risk_series.mean()) if not risk_series.empty else 0.4
    runs: list[dict[str, Any]] = []
    for run_id in range(1, n_runs + 1):
        eta_base = 4.0 + 6.0 * baseline
        eta_noise = rng.normal(loc=0.0, scale=0.5)
        eta_hours = max(0.5, eta_base + eta_noise)
        disruption_mean = min(0.95, 0.3 + 0.7 * baseline)
        disruption = float(np.clip(rng.normal(disruption_mean, 0.1), 0.0, 1.0))
        runs.append({
            "run_id": run_id,
            "eta_hours": round(eta_hours, 2),
            "disruption": round(disruption, 3),
        })
    return pd.DataFrame(runs)


def _run_simulation(runs: int, seed: int | None) -> None:
    """Execute simulation workflow and persist results."""
    latest = _latest_enriched()
    if latest is None:
        raise FileNotFoundError("No enriched dataset available for simulation")
    df = _load_enriched(latest)
    sim_df = simulate_air_ops(df, n_runs=runs, seed=seed)
    path = ENRICHED_DIR / "simulation_runs.csv"
    _atomic_write_df_csv(sim_df, path)
    logger.info("Simulation results written to %s", path)


def _rebuild_artifacts() -> None:
    """Recreate example artifacts from latest enriched dataset."""
    latest = _latest_enriched()
    if latest is None:
        logger.warning("No enriched dataset available for artifact rebuild")
        return
    df = _load_enriched(latest)
    mnd_df = df[df["source"] == "MND"].copy()
    summary = summarize_theater(mnd_df, horizon="24h")
    _build_examples(mnd_df, summary)
    logger.info("Artifacts regenerated from %s", latest)


def parse_bbox(value: str | None) -> list[float] | None:
    """Parse bounding box string into list of floats."""
    if not value:
        return None
    try:
        parts = [float(part) for part in value.split(",")]
        if len(parts) != 4:
            raise ValueError("BBox must have 4 comma-separated values")
        return parts
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Taiwan gray-zone ETL and simulation toolkit",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="OpenSky lookback window in hours",
    )
    parser.add_argument(
        "--bbox",
        type=parse_bbox,
        default=None,
        help="Bounding box lamin,lomin,lamax,lomax",
    )
    parser.add_argument(
        "--out-prefix",
        dest="out_prefix",
        default=None,
        help="Prefix for enriched output filename",
    )
    parser.add_argument(
        "--mode",
        choices=["pipeline", "artifacts", "simulate"],
        default="pipeline",
        help="Execution mode",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=500,
        help="Simulation runs (simulate mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for simulations",
    )
    return parser


def main() -> None:
    """Entry point for CLI execution."""
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "pipeline":
        _run_pipeline(args.hours, args.bbox, args.out_prefix)
    elif args.mode == "artifacts":
        _rebuild_artifacts()
    elif args.mode == "simulate":
        _run_simulation(args.runs, args.seed)
    else:
        parser.error("Unsupported mode")


if __name__ == "__main__":
    main()







