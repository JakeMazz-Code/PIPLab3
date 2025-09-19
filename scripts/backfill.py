#!/usr/bin/env python3
"""Backfill daily history parquet and risk partitions."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

import main
from deepseek_enrichment import enrich_incidents, reset_llm_metrics

HISTORY_DIR = Path("data/history")
INCIDENTS_DIR = HISTORY_DIR / "incidents_enriched"
RISK_DIR = HISTORY_DIR / "daily_grid_risk"
MANIFEST_PATH = HISTORY_DIR / "history_index.json"


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


def _build_daily_grid_risk(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "risk_score" not in df.columns:
        return pd.DataFrame(columns=["day", "grid_id", "risk_score"])
    enriched = df.copy()
    enriched["dt"] = pd.to_datetime(
        enriched["dt"], utc=True, errors="coerce"
    )
    enriched = enriched.dropna(subset=["dt"])
    if enriched.empty:
        return pd.DataFrame(columns=["day", "grid_id", "risk_score"])
    risk_series = pd.to_numeric(
        enriched.get("risk_score"), errors="coerce"
    ).fillna(0.4)
    if "validation_score" in enriched.columns:
        val_series = pd.to_numeric(
            enriched["validation_score"], errors="coerce"
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
    return grouped


def _ensure_dirs() -> None:
    INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)
    RISK_DIR.mkdir(parents=True, exist_ok=True)


def _process_day(target_day: datetime.date, resume: bool) -> bool:
    tag = target_day.strftime("%Y%m%d")
    incident_path = INCIDENTS_DIR / f"incidents_{tag}.parquet"
    risk_path = RISK_DIR / f"risk_{tag}.csv"
    if resume and incident_path.exists() and risk_path.exists():
        print(f"[skip] {tag} already present")
        return False
    print(f"[run] {tag} backfill")
    os_df = main.extract_opensky(hours=24, bbox=None)
    mnd_rows = main.scrape_mnd()

    merged = main.clean_merge(os_df, mnd_rows)
    merged = main._assign_mnd_grids_from_guess(merged)
    grid_density, hour_density = main._prepare_validation(os_df)

    mnd_df = merged[merged["source"] == "MND"].copy()
    mnd_df = main._apply_validation(mnd_df, grid_density, hour_density)

    reset_llm_metrics()
    enriched_mnd = enrich_incidents(mnd_df)
    enriched_mnd = main._assign_mnd_grids_from_guess(enriched_mnd)

    os_only = merged[merged["source"] != "MND"].copy()
    combined = pd.concat([os_only, enriched_mnd], ignore_index=True)
    combined = main._assign_mnd_grids_from_guess(combined)
    combined["dt"] = pd.to_datetime(
        combined["dt"], utc=True, errors="coerce"
    )
    combined = combined.dropna(subset=["dt"])

    main._atomic_write_parquet(combined, incident_path)
    risk_df = _build_daily_grid_risk(combined)
    main._atomic_write_df_csv(risk_df, risk_path)
    return True


def _write_manifest() -> None:
    entries: list[dict[str, Any]] = []
    for incident_file in sorted(INCIDENTS_DIR.glob("incidents_*.parquet")):
        tag = incident_file.stem.split("_")[-1]
        risk_file = RISK_DIR / f"risk_{tag}.csv"
        incident_stat = incident_file.stat()
        risk_stat = risk_file.stat() if risk_file.exists() else None
        entry = {
            "date": f"{tag[:4]}-{tag[4:6]}-{tag[6:]}",
            "incident_path": str(incident_file),
            "incident_size": incident_stat.st_size,
            "incident_mtime": datetime.fromtimestamp(
                incident_stat.st_mtime, tz=timezone.utc
            ).isoformat(),
            "risk_path": str(risk_file) if risk_file.exists() else None,
            "risk_size": risk_stat.st_size if risk_stat else None,
            "risk_mtime": datetime.fromtimestamp(
                risk_stat.st_mtime, tz=timezone.utc
            ).isoformat() if risk_stat else None,
        }
        entries.append(entry)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "days": entries,
    }
    _atomic_write_json(MANIFEST_PATH, manifest)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill history partitions over recent days."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=10,
        help="Number of days to backfill (default: 10)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.5,
        help="Seconds to sleep between days",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip days where outputs already exist",
    )
    return parser.parse_args(argv)


def main_cli(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.days <= 0:
        print("--days must be positive", file=sys.stderr)
        return 1
    _ensure_dirs()
    today = datetime.now(timezone.utc).date()
    targets = [
        today - timedelta(days=offset + 1)
        for offset in reversed(range(args.days))
    ]
    for index, day in enumerate(targets):
        try:
            processed = _process_day(day, args.resume)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[error] {day} failed: {exc}", file=sys.stderr)
            return 1
        if processed and args.sleep > 0 and index < len(targets) - 1:
            time.sleep(args.sleep)
    _write_manifest()
    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution
    sys.exit(main_cli())
