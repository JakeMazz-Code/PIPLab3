#!/usr/bin/env python3
"""Print the METRICS line from a log as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_METRIC_KEYS = [
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
]


def parse_metrics_line(path: Path) -> dict[str, str] | None:
    last_line = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("METRICS |"):
                last_line = line.strip()
    if not last_line:
        return None
    _, payload = last_line.split("|", maxsplit=1)
    fields = payload.strip().split()
    metrics: dict[str, str] = {}
    for item in fields:
        if "=" not in item:
            continue
        key, value = item.split("=", maxsplit=1)
        metrics[key] = value
    return metrics


def coerce_values(metrics: dict[str, str]) -> dict[str, int | float | str]:
    result: dict[str, int | float | str] = {}
    for key, value in metrics.items():
        try:
            if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                result[key] = int(value)
                continue
            result[key] = float(value)
            continue
        except ValueError:
            pass
        result[key] = value
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print METRICS line as JSON",
    )
    parser.add_argument(
        "--log",
        required=True,
        help="Log file containing the METRICS line",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Metrics log {log_path} not found")
        return 1
    metrics = parse_metrics_line(log_path)
    if metrics is None:
        print("METRICS line not found")
        return 1
    missing = [key for key in REQUIRED_METRIC_KEYS if key not in metrics]
    if missing:
        print(f"Missing keys: {', '.join(missing)}")
        return 1
    parsed = coerce_values(metrics)
    print(json.dumps(parsed, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
