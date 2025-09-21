"""Lightweight smoke checks for Streamlit helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import _filter_window, _prepare_heatmap, _watch_cells  # noqa: E402


def _load_latest() -> pd.DataFrame:
    enriched_dir = ROOT / "data" / "enriched"
    candidates = sorted(enriched_dir.glob("*.parquet"))
    if not candidates:
        return pd.DataFrame()
    return pd.read_parquet(candidates[-1])


def main() -> None:
    app_path = ROOT / "app.py"
    app_text = app_path.read_text(encoding="utf-8")
    if "Window:" not in app_text:
        raise SystemExit("Window header text missing")
    df = _load_latest()
    sub = _filter_window(df, 24)
    _ = _prepare_heatmap(sub)
    _ = _watch_cells(sub, cutoff=0.35, only_mnd=False, top_k=5)
    risk_path = ROOT / "data" / "enriched" / "daily_grid_risk.csv"
    if risk_path.exists():
        distinct = 0
        try:
            risk_df = pd.read_csv(risk_path)
            if "risk_score" in risk_df.columns:
                risk_series = pd.to_numeric(
                    risk_df["risk_score"], errors="coerce"
                ).dropna()
                distinct = int(risk_series.nunique())
        except Exception as exc:
            print(f"daily_grid_risk.csv read failed: {exc}")
        else:
            print(
                "daily_grid_risk distinct risk_score="
                f"{distinct}"
            )
    print("ui_smoke passed")


if __name__ == "__main__":
    main()
