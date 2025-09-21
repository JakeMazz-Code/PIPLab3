"""Lightweight smoke checks for Streamlit monitor helpers."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    app_mod = importlib.import_module("app")
    print("app import OK")
    snapshot_df, _ = app_mod._load_latest_enriched()
    snapshot_risk, _ = app_mod._load_snapshot_risk()
    history_days = app_mod._list_history_days()
    _ = app_mod._load_history_enriched(1)
    _ = app_mod._load_history_risk(1)
    synthetic = pd.DataFrame(
        {
            "dt": pd.to_datetime(["2025-01-01T00:00:00Z"]),
            "lat": [23.5],
            "lon": [121.0],
            "source": ["MND"],
            "grid_id": ["R235C602"],
            "severity_0_5": [2.0],
            "risk_score": [0.4],
            "actors": [["synthetic"]],
            "corroborations": ["OS_ANOM:1.2"],
        }
    )
    df_os, df_mnd = app_mod._split_sources(synthetic)
    risk_sample = pd.DataFrame(
        {
            "grid_id": ["R235C602"],
            "risk_score": [0.6],
            "lat": [23.5],
            "lon": [121.0],
        }
    )
    if app_mod.pdk is not None:
        layers, fallback, _ = app_mod._build_map_layers(
            df_os,
            df_mnd,
            risk_sample,
            show_hex=False,
            show_os=True,
            show_mnd=True,
            starred=[],
            selected_grid=None,
        )
        print(
            f"map layers={len(layers)} fallback_points={len(fallback)}"
        )
    blank_df = pd.DataFrame()
    _ = app_mod._watch_cells(blank_df, cutoff=0.35, only_mnd=True, top_k=5)
    print(
        f"snapshot_rows={len(snapshot_df)} risk_rows={len(snapshot_risk)} "
        f"history_days={len(history_days)}"
    )
    print("ui_smoke passed")


if __name__ == "__main__":
    main()
