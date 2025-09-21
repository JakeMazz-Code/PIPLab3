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

try:
    from opensky_api import OpenSkyApi  # type: ignore
except Exception:
    OpenSkyApi = None

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
THRESH_OS_MIN = 50  # retry with MAX when points are extremely sparse
_AUTO_RETRY_MAX_RAW = os.getenv("AUTO_RETRY_MAX", "false").lower()
AUTO_RETRY_MAX_ENABLED = _AUTO_RETRY_MAX_RAW in {"1", "true", "t", "yes", "y"}
_MAX_MND_ENRICH_RAW = os.getenv("MAX_MND_ENRICH", "0").strip()
try:
    MAX_MND_ENRICH = int(_MAX_MND_ENRICH_RAW or "0")
except ValueError:
    MAX_MND_ENRICH = 0
if MAX_MND_ENRICH < 0:
    MAX_MND_ENRICH = 0
OPENSKY_URL = "https://opensky-network.org/api/states/all"
MND_LIST_URL = "https://www.mnd.gov.tw/PublishTable.aspx"
MND_PARAMS = {
    "Types": "\u5373\u6642\u8ecd\u4e8b\u52d5\u614b",
    "title": "\u570b\u9632\u6d88\u606f",
}
REQUEST_TIMEOUT = 20

OPENSKY_STATE_COLUMNS = [
    "time",
    "icao24",
    "callsign",
    "origin_country",
    "time_position",
    "last_contact",
    "longitude",
    "latitude",
    "geo_altitude",
    "on_ground",
    "velocity",
    "true_track",
    "vertical_rate",
    "baro_altitude",
    "squawk",
    "spi",
    "position_source",
    "category",
]

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
        delete=False, dir=path.parent, suffix='.parquet'
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.to_parquet(tmp_path, index=False)
        try:
            with tmp_path.open('rb') as handle:
                os.fsync(handle.fileno())
        except (OSError, AttributeError):
            pass
        os.replace(tmp_path, path)
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


def _env_str(name: str, default: str = "") -> str:
    """Return an environment variable as a string."""

    value = os.getenv(name)
    if value is None:
        return default
    return str(value)


def _get_opensky_basic() -> tuple[str | None, str | None]:
    """Return OpenSky basic credentials from the environment."""

    user = _env_str("OPENSKY_USER").strip()
    password = _env_str("OPENSKY_PASS").strip()
    if user and password:
        return user, password
    return None, None


def _os_bearer_token() -> str | None:
    """Return an OAuth2 bearer token for OpenSky if available."""

    import json
    import urllib.parse
    import urllib.request

    url = _env_str(
        "OPENSKY_TOKEN_URL",
        "https://auth.opensky-network.org/oauth/token",
    )
    client_id = _env_str("OPENSKY_CLIENT_ID").strip()
    client_secret = _env_str("OPENSKY_CLIENT_SECRET").strip()
    static = _env_str("OPENSKY_BEARER_TOKEN").strip()
    if static:
        return static
    if not (client_id and client_secret):
        return None
    payload = urllib.parse.urlencode(
        {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            text = response.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    token = data.get("access_token")
    if isinstance(token, str) and token:
        return token
    return None


def _bbox_client_order(
    lamin: float,
    lomin: float,
    lamax: float,
    lomax: float,
) -> tuple[float, float, float, float]:
    """Return bbox in client order (min_lat, max_lat, min_lon, max_lon)."""

    return lamin, lamax, lomin, lomax


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _opensky_rest_call(url: str, headers: dict[str, str]) -> str:
    """Retrieve OpenSky REST payload as text."""

    import urllib.request

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _coerce_state_frame(
    df: pd.DataFrame,
    bbox_cli: tuple[float, float, float, float],
) -> pd.DataFrame:
    """Return state DataFrame with numeric casts and bbox clipping."""

    if df.empty:
        return df
    coerced = df.copy()
    numeric_cols = [
        "time",
        "time_position",
        "last_contact",
        "longitude",
        "latitude",
        "geo_altitude",
        "velocity",
        "true_track",
        "vertical_rate",
        "baro_altitude",
        "position_source",
        "category",
    ]
    for column in numeric_cols:
        if column in coerced.columns:
            coerced[column] = pd.to_numeric(
                coerced[column], errors="coerce"
            )
    for column in ["on_ground", "spi"]:
        if column in coerced.columns:
            coerced[column] = coerced[column].astype("bool", copy=False)
    for column in ["icao24", "callsign", "origin_country", "squawk"]:
        if column in coerced.columns:
            coerced[column] = (
                coerced[column].fillna("")
                .astype(str)
                .str.strip()
            )
    coerced = coerced.dropna(subset=["latitude", "longitude"])
    if coerced.empty:
        return coerced
    coerced = coerced[
        coerced["latitude"].between(bbox_cli[0], bbox_cli[1])
        & coerced["longitude"].between(bbox_cli[2], bbox_cli[3])
    ]
    if coerced.empty:
        return coerced
    finite_mask = np.isfinite(coerced["latitude"]) & np.isfinite(
        coerced["longitude"]
    )
    coerced = coerced[finite_mask]
    return coerced.reset_index(drop=True)


def _fetch_opensky_states(
    t0: datetime,
    t1: datetime,
    bbox_cli: tuple[float, float, float, float],
    bbox_rest: tuple[float, float, float, float],
    prefer_client: bool,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    """Fetch OpenSky states via client (Basic) or REST (OAuth2/anon)."""

    import urllib.parse

    timestamp = int(t1.timestamp())
    window_start = int(t0.timestamp())
    empty_df = pd.DataFrame(columns=OPENSKY_STATE_COLUMNS)
    raw_json = json.dumps(
        {
            "time": timestamp,
            "states": [],
            "window_start": window_start,
            "path": "rest",
        },
        separators=(",", ":"),
    )
    basic_user, basic_pass = _get_opensky_basic()
    if (
        prefer_client
        and OpenSkyApi is not None
        and basic_user
        and basic_pass
    ):
        try:
            api = OpenSkyApi(basic_user, basic_pass)
            states = api.get_states(time_secs=timestamp, bbox=bbox_cli)
        except Exception as exc:  # pragma: no cover - network path
            logger.warning("OpenSky client fetch failed: %s", exc)
        else:
            if states is not None and states.states is not None:
                raw_states: list[list[Any]] = []
                records: list[dict[str, Any]] = []
                for state in states.states:
                    lat = _parse_float(getattr(state, "latitude", None))
                    lon = _parse_float(getattr(state, "longitude", None))
                    if lat is None or lon is None:
                        continue
                    if not (
                        math.isfinite(lat)
                        and math.isfinite(lon)
                    ):
                        continue
                    raw_row = [
                        getattr(state, "icao24", None),
                        getattr(state, "callsign", None),
                        getattr(state, "origin_country", None),
                        getattr(state, "time_position", None),
                        getattr(state, "last_contact", None),
                        lon,
                        lat,
                        getattr(state, "baro_altitude", None),
                        getattr(state, "on_ground", None),
                        getattr(state, "velocity", None),
                        getattr(state, "true_track", None),
                        getattr(state, "vertical_rate", None),
                        getattr(state, "sensors", None),
                        getattr(state, "geo_altitude", None),
                        getattr(state, "squawk", None),
                        getattr(state, "spi", None),
                        getattr(state, "position_source", None),
                        getattr(state, "category", None),
                    ]
                    raw_states.append(raw_row)
                    records.append(
                        {
                            "time": timestamp,
                            "icao24": raw_row[0],
                            "callsign": raw_row[1],
                            "origin_country": raw_row[2],
                            "time_position": raw_row[3],
                            "last_contact": raw_row[4],
                            "longitude": raw_row[5],
                            "latitude": raw_row[6],
                            "geo_altitude": raw_row[13],
                            "on_ground": raw_row[8],
                            "velocity": raw_row[9],
                            "true_track": raw_row[10],
                            "vertical_rate": raw_row[11],
                            "baro_altitude": raw_row[7],
                            "squawk": raw_row[14],
                            "spi": raw_row[15],
                            "position_source": raw_row[16],
                            "category": raw_row[17],
                        }
                    )
                if records:
                    frame = pd.DataFrame.from_records(
                        records, columns=OPENSKY_STATE_COLUMNS
                    )
                else:
                    frame = empty_df.copy()
                frame = _coerce_state_frame(frame, bbox_cli)
                meta = {
                    "path": "client",
                    "count": int(frame.shape[0]),
                }
                raw_json = json.dumps(
                    {
                        "time": timestamp,
                        "states": raw_states,
                        "window_start": window_start,
                        "path": "client",
                    },
                    separators=(",", ":"),
                )
                return frame, meta, raw_json
            logger.info("OpenSky client returned no states.")

    headers = {"Accept": "application/json"}
    token = _os_bearer_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    query = urllib.parse.urlencode(
        {
            "lamin": bbox_rest[0],
            "lomin": bbox_rest[1],
            "lamax": bbox_rest[2],
            "lomax": bbox_rest[3],
        }
    )
    url = f"{OPENSKY_URL}?{query}"
    try:
        raw_json = _opensky_rest_call(url, headers)
    except Exception as exc:  # pragma: no cover - network path
        logger.warning("OpenSky REST fetch failed: %s", exc)
        raw_json = json.dumps(
            {
                "time": timestamp,
                "states": None,
                "window_start": window_start,
                "path": "rest",
                "error": str(exc),
            },
            separators=(",", ":"),
        )
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError:
        payload = {"time": timestamp, "states": None}
    states_list = payload.get("states") or []
    time_value = payload.get("time", timestamp)
    records: list[dict[str, Any]] = []
    for entry in states_list:
        if not isinstance(entry, (list, tuple)):
            continue
        lon = _parse_float(entry[5] if len(entry) > 5 else None)
        lat = _parse_float(entry[6] if len(entry) > 6 else None)
        if lat is None or lon is None:
            continue
        if not (
            math.isfinite(lat)
            and math.isfinite(lon)
        ):
            continue
        records.append(
            {
                "time": time_value,
                "icao24": entry[0] if len(entry) > 0 else None,
                "callsign": entry[1] if len(entry) > 1 else None,
                "origin_country": entry[2] if len(entry) > 2 else None,
                "time_position": entry[3] if len(entry) > 3 else None,
                "last_contact": entry[4] if len(entry) > 4 else None,
                "longitude": lon,
                "latitude": lat,
                "geo_altitude": entry[13] if len(entry) > 13 else None,
                "on_ground": entry[8] if len(entry) > 8 else None,
                "velocity": entry[9] if len(entry) > 9 else None,
                "true_track": entry[10] if len(entry) > 10 else None,
                "vertical_rate": entry[11] if len(entry) > 11 else None,
                "baro_altitude": entry[7] if len(entry) > 7 else None,
                "squawk": entry[14] if len(entry) > 14 else None,
                "spi": entry[15] if len(entry) > 15 else None,
                "position_source": entry[16] if len(entry) > 16 else None,
                "category": entry[17] if len(entry) > 17 else None,
            }
        )
    if records:
        frame = pd.DataFrame.from_records(
            records, columns=OPENSKY_STATE_COLUMNS
        )
    else:
        frame = empty_df.copy()
    frame = _coerce_state_frame(frame, bbox_cli)
    meta = {"path": "rest", "count": int(frame.shape[0])}
    return frame, meta, raw_json


def extract_opensky(
    hours: int = 6,
    bbox: list | None = None,
) -> pd.DataFrame:
    """Fetch OpenSky states and return normalized DataFrame."""
    empty_columns = [
        "dt",
        "lat",
        "lon",
        "source",
        "raw_text",
        "country",
        "grid_id",
    ]
    target_bbox = bbox or BBOX_WIDE
    lamin = target_bbox[1]
    lomin = target_bbox[0]
    lamax = target_bbox[3]
    lomax = target_bbox[2]
    bbox_rest = (lamin, lomin, lamax, lomax)
    bbox_cli = _bbox_client_order(lamin, lomin, lamax, lomax)
    now = _now_utc()
    window_start = now - timedelta(hours=hours)
    basic_creds = _get_opensky_basic()
    prefer_client = bool(OpenSkyApi and basic_creds[0] and basic_creds[1])
    raw_path = RAW_DIR / f"opensky_{_timestamp()}.json"
    try:
        state_df, meta, raw_json = _fetch_opensky_states(
            window_start,
            now,
            bbox_cli,
            bbox_rest,
            prefer_client,
        )
    except Exception as exc:
        logger.error("OpenSky fetch failed: %s", exc)
        payload = {
            "time": int(now.timestamp()),
            "states": [],
            "path": "error",
            "error": str(exc),
        }
        _write_json(raw_path, payload)
        return pd.DataFrame(columns=empty_columns)
    bbox_label = "CUSTOM"
    if bbox is None:
        bbox_label = "WIDE"
    raw_content = raw_json
    logger.info(
        "OpenSky (%s) states=%s for %s bbox",
        meta.get("path", "rest"),
        meta.get("count", 0),
        bbox_label,
    )
    auto_retry = AUTO_RETRY_MAX_ENABLED and bbox is None
    if auto_retry and (
        meta.get("count", 0) < THRESH_OS_MIN or state_df.empty
    ):
        logger.info(
            "OpenSky sparse (%s); retrying once with MAX bbox.",
            meta.get("count", 0),
        )
        max_bbox_rest = (
            BBOX_MAX[1],
            BBOX_MAX[0],
            BBOX_MAX[3],
            BBOX_MAX[2],
        )
        max_bbox_cli = _bbox_client_order(*max_bbox_rest)
        try:
            state_df, meta, raw_json = _fetch_opensky_states(
                window_start,
                now,
                max_bbox_cli,
                max_bbox_rest,
                prefer_client,
            )
        except Exception as exc:
            logger.error("OpenSky MAX retry failed: %s", exc)
        else:
            raw_content = raw_json
            logger.info(
                "OpenSky (%s) states=%s for MAX bbox",
                meta.get("path", "rest"),
                meta.get("count", 0),
            )
    _atomic_write_bytes(raw_path, raw_content.encode("utf-8"))
    if state_df.empty:
        return pd.DataFrame(columns=empty_columns)
    records: list[dict[str, Any]] = []
    for row in state_df.itertuples(index=False):
        last_contact = getattr(row, "last_contact", None)
        if pd.isna(last_contact):
            continue
        try:
            ts_value = float(last_contact)
        except (TypeError, ValueError):
            continue
        dt = datetime.fromtimestamp(ts_value, tz=timezone.utc)
        if dt < window_start:
            continue
        lat = _parse_float(getattr(row, "latitude", None))
        lon = _parse_float(getattr(row, "longitude", None))
        if lat is None or lon is None:
            continue
        grid_id = latlon_to_grid(lat, lon)
        icao24_raw = getattr(row, "icao24", "")
        if pd.isna(icao24_raw):
            icao24_raw = ""
        icao24 = str(icao24_raw).strip().lower()
        callsign_raw = getattr(row, "callsign", "")
        if pd.isna(callsign_raw):
            callsign_raw = ""
        callsign = str(callsign_raw).strip()
        origin = getattr(row, "origin_country", "")
        if pd.isna(origin):
            origin = ""
        velocity = _parse_float(getattr(row, "velocity", None))
        heading = _parse_float(getattr(row, "true_track", None))
        record = {
            "dt": dt,
            "lat": lat,
            "lon": lon,
            "source": "OpenSky",
            "raw_text": f"icao24={icao24} callsign={callsign}",
            "country": origin,
            "grid_id": grid_id or "RNaNCNaN",
            "icao24": icao24,
            "callsign": callsign,
            "velocity": velocity,
            "heading": heading,
        }
        records.append(record)
    if not records:
        return pd.DataFrame(columns=empty_columns)
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
    frames: list[pd.DataFrame] = []
    if not os_clean.empty:
        frames.append(os_clean[os_cols])
    if not mnd_df.empty:
        frames.append(mnd_df[os_cols])
    if not frames:
        combined = pd.DataFrame(columns=os_cols)
    else:
        combined = pd.concat(frames, ignore_index=True)
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
    hourly_grid_counts: dict[pd.Timestamp, list[float]] = {}
    if not grid_density.empty:
        local = grid_density.copy()
        local["hour"] = pd.to_datetime(local["hour"], utc=True)
        grid_counts = {
            (row.grid_id, row.hour): float(row.count)
            for row in local.itertuples()
        }
        for (grid_id, hour), count in grid_counts.items():
            hourly_grid_counts.setdefault(hour, []).append(count)
    core_counts = pd.Series(dtype=float)
    core_mean = None
    core_std = None
    if not hour_density.empty:
        hour_density = hour_density.copy()
        hour_density["hour"] = pd.to_datetime(hour_density["hour"], utc=True)
        core_counts = hour_density.set_index("hour")["count"].astype(float)
        core_mean = core_counts.mean()
        core_std = core_counts.std(ddof=0)
    total_opensky = float(sum(grid_counts.values())) if grid_counts else 0.0
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
        target_count = grid_counts.get((grid, hour), 0.0)
        counts_for_hour = hourly_grid_counts.get(hour, [])
        z_score = 0.0
        if counts_for_hour:
            counts_array = np.array(counts_for_hour, dtype=float)
            mean = float(np.mean(counts_array))
            std = float(np.std(counts_array, ddof=0))
            if std and not np.isnan(std):
                z_score = (target_count - mean) / std
        elif not core_counts.empty and hour in core_counts.index:
            if core_std is not None:
                std = float(core_std)
                if not np.isnan(std) and std != 0.0:
                    total_count = float(core_counts.loc[hour])
                    baseline = float(core_mean or 0.0)
                    z_score = (total_count - baseline) / std
        if total_opensky == 0.0:
            z_str = '0.00'
            scores.append(1.0)
        else:
            clamped = max(-3.0, min(3.0, z_score))
            z_str = f"{clamped:.2f}"
            scores.append(float(z_str))
        corroborations.append([f"OS_ANOM:{z_str}"])
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
    risk_series = pd.to_numeric(
        enriched.get("risk_score"),
        errors="coerce",
    ).fillna(0.4)
    if "validation_score" in enriched.columns:
        val_series = pd.to_numeric(
            enriched["validation_score"],
            errors="coerce",
        ).fillna(0.0)
        adjustment = val_series.clip(-3.0, 3.0) / 6.0
        risk_series = (risk_series + adjustment).clip(0.05, 1.0)
    if risk_series.nunique() <= 1 and "grid_id" in enriched.columns:
        def _grid_jitter(value: str) -> float:
            code = sum(ord(ch) for ch in value)
            return (code % 11) / 20.0
        jitter = enriched["grid_id"].astype(str).map(_grid_jitter)
        risk_series = (risk_series + jitter).clip(0.05, 1.0)
    enriched["risk_score"] = risk_series
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
    if MAX_MND_ENRICH > 0 and len(mnd_df) > MAX_MND_ENRICH:
        logger.info(
            "Limiting MND incidents to %s via MAX_MND_ENRICH",
            MAX_MND_ENRICH,
        )
        mnd_df = mnd_df.head(MAX_MND_ENRICH)
    mnd_df = _apply_validation(mnd_df, grid_density, hour_density, metrics)
    reset_llm_metrics()
    logger.info("Enriching %s incidents via DeepSeek", len(mnd_df))
    enriched_mnd = enrich_incidents(mnd_df)
    enriched_mnd = _assign_mnd_grids_from_guess(enriched_mnd)
    os_only = merged[merged["source"] != "MND"].copy()
    merge_frames = [df for df in (os_only, enriched_mnd) if not df.empty]
    if merge_frames:
        combined = pd.concat(merge_frames, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=merged.columns)
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
        review_series = enriched_mnd["needs_review"]
        review_series = review_series.infer_objects(copy=False)
        review_series = review_series.fillna(False)
        review_series = review_series.astype("bool", copy=False)
        enriched_mnd["needs_review"] = review_series
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







