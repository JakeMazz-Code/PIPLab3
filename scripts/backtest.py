#!/usr/bin/env python3
"""Lightweight backtest over historical enriched incidents."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

ENRICHED_DIR = Path("data/enriched")
EXAMPLES_DIR = Path("examples")


def _latest_parquet(directory: Path) -> Path | None:
    files = list(directory.glob("*.parquet"))
    if not files:
        return None
    files.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return files[0]


def _load_enriched() -> pd.DataFrame:
    parquet_path = _latest_parquet(ENRICHED_DIR)
    if parquet_path is None:
        raise FileNotFoundError("No enriched parquet files found")
    df = pd.read_parquet(parquet_path)
    df["dt"] = pd.to_datetime(df["dt"], utc=True, errors="coerce")
    df = df.dropna(subset=["dt"])
    return df


def _grid_sets(df: pd.DataFrame, day: datetime) -> tuple[set[str], dict[str, float]]:
    window_start = day - timedelta(days=3)
    window_end = day
    window = df[(df["dt"] >= window_start) & (df["dt"] < window_end)]
    window = window.copy()
    window["risk_score"] = pd.to_numeric(window.get("risk_score"), errors="coerce")
    forecast = (
        window.groupby("grid_id")["risk_score"].mean().dropna()
    )
    day_rows = df[(df["dt"] >= day) & (df["dt"] < day + timedelta(days=1))]
    observed = {
        grid for grid in day_rows[day_rows["source"] == "MND"]["grid_id"] if grid
    }
    return observed, forecast.to_dict()


def _select_hotspots(forecast: dict[str, float], cutoff: float | None, top_k: int | None) -> set[str]:
    if not forecast:
        return set()
    items = sorted(forecast.items(), key=lambda pair: pair[1], reverse=True)
    if top_k is not None:
        return {grid for grid, _ in items[:top_k]}
    threshold = cutoff if cutoff is not None else 0.35
    return {grid for grid, score in items if score >= threshold}


def _confusion(predicted: set[str], observed: set[str], universe: set[str]) -> tuple[int, int, int, int]:
    tp = len(predicted & observed)
    fp = len(predicted - observed)
    fn = len(observed - predicted)
    tn = len(universe - (predicted | observed))
    return tp, fp, fn, tn


def _brier(forecast: dict[str, float], observed: set[str], universe: set[str]) -> float:
    scores = []
    for grid in universe:
        prob = forecast.get(grid, 0.0)
        outcome = 1.0 if grid in observed else 0.0
        scores.append((prob - outcome) ** 2)
    return sum(scores) / len(scores) if scores else 0.0


def _safe_ratio(numer: float, denom: float) -> float:
    return numer / denom if denom else 0.0


def backtest(days: int, cutoff: float | None, top_k: int | None) -> dict[str, float]:
    df = _load_enriched()
    df = df.sort_values("dt")
    all_days = sorted({row.date() for row in df["dt"]})
    if len(all_days) < 4:
        raise ValueError("Not enough daily history for backtest")
    selected_days = all_days[-days:]
    metrics_accumulator = defaultdict(list)
    confusion_totals = [0, 0, 0, 0]

    for day_date in selected_days:
        day = datetime.combine(day_date, datetime.min.time(), tzinfo=timezone.utc)
        observed, forecast = _grid_sets(df, day)
        predicted = _select_hotspots(forecast, cutoff, top_k)
        if not forecast and not observed:
            continue
        universe = set(forecast.keys()) | observed
        tp, fp, fn, tn = _confusion(predicted, observed, universe)
        confusion_totals[0] += tp
        confusion_totals[1] += fp
        confusion_totals[2] += fn
        confusion_totals[3] += tn

        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        numerator = 2 * (tp * tn - fn * fp)
        denominator = ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
        hss = numerator / denominator if denominator else 0.0
        brier = _brier(forecast, observed, universe)

        metrics_accumulator["precision"].append(precision)
        metrics_accumulator["recall"].append(recall)
        metrics_accumulator["f1"].append(f1)
        metrics_accumulator["heidke_skill"].append(hss)
        metrics_accumulator["brier"].append(brier)

    metrics_summary = {
        key: sum(values) / len(values) if values else 0.0
        for key, values in metrics_accumulator.items()
    }
    metrics_summary.update({
        "true_positive": float(confusion_totals[0]),
        "false_positive": float(confusion_totals[1]),
        "false_negative": float(confusion_totals[2]),
        "true_negative": float(confusion_totals[3]),
    })
    return metrics_summary


def save_outputs(metrics: dict[str, float]) -> None:
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = EXAMPLES_DIR / "backtest_confusion_matrix.png"
    json_path = EXAMPLES_DIR / "skill_metrics.json"

    tp = metrics.get("true_positive", 0.0)
    fp = metrics.get("false_positive", 0.0)
    fn = metrics.get("false_negative", 0.0)
    tn = metrics.get("true_negative", 0.0)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.axis("off")
    table = [[f"TP
{int(tp)}", f"FP
{int(fp)}"], [f"FN
{int(fn)}", f"TN
{int(tn)}"]]
    ax.table(cellText=table, loc="center")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    summary = {key: round(value, 4) for key, value in metrics.items()}
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest grid risk forecasts over recent days",
    )
    parser.add_argument("--days", type=int, default=10, help="Number of days to score")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cutoff", type=float, default=0.35, help="Risk cutoff for hotspot selection")
    group.add_argument("--top-k", type=int, default=None, help="Top K grids to flag each day")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metrics = backtest(args.days, args.cutoff if args.top_k is None else None, args.top_k)
    except Exception as exc:
        print(f"Backtest failed: {exc}")
        return 1
    save_outputs(metrics)
    print("Backtest complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
