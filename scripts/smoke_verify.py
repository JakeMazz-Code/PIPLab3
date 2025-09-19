#!/usr/bin/env python3
"""Quick smoke checks for gray-zone artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_METRIC_KEYS = {
    "opensky_points",
    "mnd_rows",
    "merged_rows",
    "enriched_rows",
    "llm_success",
    "llm_invalid_json",
    "llm_retries",
    "needs_review_count",
    "validation_sparse_fallbacks",
    "os_anom_rows",
    "wall_ms",
}


def _latest_parquet(directory: Path) -> Path | None:
    files = list(directory.glob("*.parquet"))
    if not files:
        return None
    files.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return files[0]


def _contains_os_anom(value: Any) -> bool:
    if isinstance(value, str):
        return "OS_ANOM:" in value
    if isinstance(value, (list, tuple, set)):
        return any("OS_ANOM:" in str(item) for item in value)
    return "OS_ANOM:" in str(value)


def _read_metrics_line(path: Path) -> dict[str, str] | None:
    last_line = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("METRICS |"):
                last_line = line.strip()
    if not last_line:
        return None
    _, payload = last_line.split("|", maxsplit=1)
    pairs = payload.strip().split()
    metrics: dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            continue
        key, value = item.split("=", maxsplit=1)
        metrics[key] = value
    return metrics


def check_artifacts(parquet_dir: Path, metrics_log: Path | None) -> str:
    parquet_path = _latest_parquet(parquet_dir)
    if parquet_path is None:
        return f"No parquet files found in {parquet_dir}"
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:  # pragma: no cover - defensive
        return f"Failed to load {parquet_path}: {exc}"

    if "source" not in df.columns:
        return "Parquet missing 'source' column"
    if "corroborations" not in df.columns:
        return "Parquet missing 'corroborations' column"

    mnd = df[df["source"] == "MND"]
    if mnd.empty:
        return "No MND rows found in parquet"
    os_anom = mnd["corroborations"].apply(_contains_os_anom)
    if not bool(os_anom.any()):
        return "No MND corroborations contain OS_ANOM tags"

    risk_csv = parquet_dir / "daily_grid_risk.csv"
    if not risk_csv.exists():
        risk_csv = Path("data/enriched/daily_grid_risk.csv")
    if not risk_csv.exists():
        return "daily_grid_risk.csv not found"
    try:
        grid_df = pd.read_csv(risk_csv)
    except Exception as exc:  # pragma: no cover - defensive
        return f"Failed to load {risk_csv}: {exc}"
    if grid_df.empty or "risk_score" not in grid_df.columns:
        return "Risk CSV missing risk_score data"
    distinct = pd.to_numeric(grid_df["risk_score"], errors="coerce").round(3)
    if distinct.dropna().nunique() <= 1:
        return "Risk scores appear flat (<=1 distinct value)"

    if metrics_log is not None:
        if not metrics_log.exists():
            return f"Metrics log {metrics_log} not found"
        metrics = _read_metrics_line(metrics_log)
        if metrics is None:
            return "Metrics log missing METRICS line"
        missing = REQUIRED_METRIC_KEYS - metrics.keys()
        if missing:
            missing_str = ', '.join(sorted(missing))
            return f"Metrics log missing keys: {missing_str}"

    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate enriched artifacts and metrics",
    )
    parser.add_argument(
        "--parquet-dir",
        default="data/enriched",
        help="Directory containing parquet outputs",
    )
    parser.add_argument(
        "--metrics-log",
        default=None,
        help="Optional path to a log file containing the METRICS line",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parquet_dir = Path(args.parquet_dir)
    metrics_log = Path(args.metrics_log) if args.metrics_log else None
    reason = check_artifacts(parquet_dir, metrics_log)
    if reason:
        print(f"smoke_verify failed: {reason}")
        return 1
    print("smoke_verify passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
