"""Main ETL pipeline for Taiwan gray-zone monitoring."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from deepseek_enrichment import enrich_incidents, summarize_theater


logging.basicConfig(
    level=os.getenv("GRAYZONE_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
ENRICHED_DIR = BASE_DIR / "data" / "enriched"
EXAMPLES_DIR = BASE_DIR / "examples"

DEFAULT_BBOX = [118.0, 20.0, 123.0, 26.0]
GRID_STEP = 0.5
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


def _write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write *content* to *path* with safe encoding."""
    path.write_text(content, encoding=encoding)


def _write_json(path: Path, data: Any) -> None:
    """Write JSON data with UTF-8 encoding."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def _parse_float(value: Any) -> float | None:
    """Parse a float from arbitrary value."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_grid_id(lat: float | None, lon: float | None) -> str:
    """Return stable 0.5 deg grid identifier for latitude and longitude."""
    if lat is None or lon is None:
        return "RNaNCNaN"
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return "RNaNCNaN"
    row = math.floor((lat + 90.0) / GRID_STEP)
    col = math.floor((lon + 180.0) / GRID_STEP)
    return f"R{row}C{col}"


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
        "\u4e2d\u7dda",
        "\u4e2d\u7ebf",
    )
    north_tokens = ("north", "northern", "\u5317")
    south_tokens = ("south", "southern", "\u5357")
    east_tokens = ("east", "eastern", "\u6771", "\u4e1c")
    west_tokens = ("west", "western", "\u897f")
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
    bbox = bbox or DEFAULT_BBOX
    params = {
        "lamin": bbox[1],
        "lomax": bbox[3],
        "lamax": bbox[2],
        "lomin": bbox[0],
    }
    try:
        data = _fetch_opensky(params)
    except requests.RequestException as exc:
        logger.error("OpenSky fetch failed: %s", exc)
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
    states = data.get("states") or []
    records: list[dict[str, Any]] = []
    window_start = _now_utc() - timedelta(hours=hours)
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
        record = {
            "dt": dt,
            "lat": lat,
            "lon": lon,
            "source": "OpenSky",
            "raw_text": f"icao24={entry[0]} callsign={entry[1]}",
            "country": entry[2],
            "grid_id": _compute_grid_id(lat, lon),
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
    combined = pd.concat([os_clean[os_cols], mnd_df[os_cols]], ignore_index=True)
    combined["dt"] = pd.to_datetime(combined["dt"], utc=True)
    return combined


def _prepare_validation(os_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-grid hourly density for validation."""
    if os_df.empty:
        return pd.DataFrame(columns=["grid_id", "hour", "count"])
    frame = os_df.copy()
    frame["hour"] = frame["dt"].dt.floor("H")
    grouped = frame.groupby(["grid_id", "hour"]).size().reset_index()
    grouped = grouped.rename(columns={0: "count"})
    return grouped


def _assign_mnd_grids_from_guess(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with grid IDs derived from textual hints for MND rows."""
    if df.empty or "where_guess" not in df.columns:
        return df
    updated = df.copy()
    mask = (updated["source"] == "MND") & (
        updated["grid_id"].isna() | (updated["grid_id"] == "RNaNCNaN")
    )
    if not mask.any():
        return updated
    updated.loc[mask, "grid_id"] = updated.loc[
        mask, "where_guess"
    ].apply(mnd_where_to_grid)
    return updated


def _apply_validation(
    mnd_df: pd.DataFrame,
    density: pd.DataFrame,
) -> pd.DataFrame:
    """Attach validation scores to MND incidents."""
    if mnd_df.empty:
        mnd_df["validation_score"] = 1.0
        mnd_df["corroborations"] = [[] for _ in range(len(mnd_df))]
        return mnd_df
    density_map = density.set_index(["grid_id", "hour"]) if not density.empty else None
    scores: list[float] = []
    corroborations: list[list[str]] = []
    for _, row in mnd_df.iterrows():
        dt = row.get("dt")
        grid = row.get("grid_id")
        if not isinstance(dt, pd.Timestamp):
            scores.append(1.0)
            corroborations.append([])
            continue
        hour = dt.floor("H")
        if density_map is None:
            scores.append(1.0)
            corroborations.append([])
            continue
        total_hour = density_map.xs(hour, level="hour", drop_level=False)
        if total_hour.empty:
            scores.append(1.0)
            corroborations.append([])
            continue
        mean = total_hour["count"].mean()
        target = density_map.loc[(grid, hour)]["count"] if (grid, hour) in density_map.index else None
        if target is None or np.isnan(mean) or mean == 0:
            scores.append(1.0)
            corroborations.append([])
            continue
        deviation = (target - mean) / max(mean, 1.0)
        validation = max(0.1, min(2.0, 1.0 + deviation))
        scores.append(validation)
        corroborations.append([f"OS_ANOM:{deviation:.2f}"])
    mnd_df["validation_score"] = scores
    mnd_df["corroborations"] = corroborations
    return mnd_df


def _write_enriched(df: pd.DataFrame, prefix: str | None) -> Path:
    """Persist enriched DataFrame to parquet (CSV fallback)."""
    stamp = _timestamp()
    label = prefix or "incidents"
    parquet_path = ENRICHED_DIR / f"{label}_enriched_{stamp}.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except (ImportError, ValueError) as exc:
        logger.warning("Parquet write failed (%s); falling back to CSV", exc)
        csv_path = parquet_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def _write_daily_grid_risk(df: pd.DataFrame) -> Path:
    """Write daily grid risk summary CSV."""
    if df.empty or "risk_score" not in df.columns:
        path = ENRICHED_DIR / "daily_grid_risk.csv"
        pd.DataFrame()
        return path
    enriched = df.copy()
    enriched["dt"] = pd.to_datetime(enriched["dt"], utc=True)
    enriched = enriched.dropna(subset=["dt"])
    if enriched.empty:
        path = ENRICHED_DIR / "daily_grid_risk.csv"
        pd.DataFrame().to_csv(path, index=False)
        return path
    enriched["day"] = enriched["dt"].dt.floor("D")
    grouped = (
        enriched.groupby(["day", "grid_id"])["risk_score"]
        .mean()
        .reset_index()
    )
    path = ENRICHED_DIR / "daily_grid_risk.csv"
    grouped.to_csv(path, index=False)
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
    logger.info("Starting extraction phase")
    os_df = extract_opensky(hours=hours, bbox=bbox)
    mnd_rows = scrape_mnd()
    merged = clean_merge(os_df, mnd_rows)
    merged = _assign_mnd_grids_from_guess(merged)
    mnd_df = merged[merged["source"] == "MND"].copy()
    density = _prepare_validation(os_df)
    mnd_df = _apply_validation(mnd_df, density)
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
    sim_df.to_csv(path, index=False)
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


