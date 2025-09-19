#!/usr/bin/env python3
"""Full project verifier: baseline artifacts + UI policies.

This script runs a compact battery of checks and prints a PASS/FAIL/WARN
matrix. It is safe to run repeatedly. By default it does NOT call the
network; pass --run-pipeline to execute a short ETL.

Usage (PowerShell):
  python scripts/verify_all.py
  python scripts/verify_all.py --run-pipeline --hours 1 --prefix verify

Usage (bash):
  python scripts/verify_all.py --run-pipeline --hours 1 --prefix verify
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
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


@dataclass
class Line:
    kind: str  # PASS | FAIL | WARN | SKIP
    name: str
    note: str


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a command and return the CompletedProcess."""
    return subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )


def _latest_parquet(dirpath: Path) -> Path | None:
    files = list(dirpath.glob("*.parquet"))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def _read_metrics_line(text: str) -> dict[str, str] | None:
    last = None
    for line in text.splitlines():
        if line.startswith("METRICS |"):
            last = line.strip()
    if not last:
        return None
    _, payload = last.split("|", 1)
    metrics: dict[str, str] = {}
    for item in payload.strip().split():
        if "=" in item:
            k, v = item.split("=", 1)
            metrics[k] = v
    return metrics


def _contains_os_anom(value: Any) -> bool:
    if isinstance(value, str):
        return "OS_ANOM:" in value
    if isinstance(value, (list, tuple, set)):
        return any("OS_ANOM:" in str(v) for v in value)
    return "OS_ANOM:" in str(value)


def verify(run_pipeline: bool, hours: int, prefix: str) -> list[Line]:
    lines: list[Line] = []

    # Ensure dirs
    Path("logs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    # T01: import guards
    try:
        import importlib

        for mod in ("main", "deepseek_enrichment", "app"):
            importlib.import_module(mod)
        lines.append(Line("PASS", "T01 imports", "main/app/enrichment OK"))
    except Exception as exc:  # pragma: no cover - defensive
        lines.append(Line("FAIL", "T01 imports", f"{exc!s}"))

    # T02: optional pipeline run
    plog = Path("logs") / f"{prefix}.log"
    if run_pipeline:
        cmd = [sys.executable, "main.py", "--hours", str(hours), "--out-prefix",
               prefix]
        proc = _run(cmd)
        plog.write_text(proc.stdout, encoding="utf-8")
        if proc.returncode == 0:
            lines.append(Line("PASS", "T02 pipeline", f"log -> {plog}"))
        else:
            lines.append(Line("FAIL", "T02 pipeline", f"rc={proc.returncode}"))
    else:
        lines.append(Line("SKIP", "T02 pipeline", "use --run-pipeline to run"))

    # Locate a log to parse metrics (prefer the one we just made)
    log_text = plog.read_text(encoding="utf-8") if plog.exists() else ""
    if not log_text:
        # probe any .log under logs/
        probes = sorted(Path("logs").glob("*.log"),
                        key=lambda p: p.stat().st_mtime)
        if probes:
            log_text = probes[-1].read_text(encoding="utf-8")

    # T03: metrics line + fields
    metrics = _read_metrics_line(log_text) if log_text else None
    if metrics is None:
        lines.append(Line("FAIL", "T03 metrics",
                          "no 'METRICS |' line found in logs/"))
    else:
        missing = REQUIRED_METRIC_KEYS - set(metrics)
        if missing:
            lines.append(Line("FAIL", "T03 metrics",
                              f"missing keys: {sorted(missing)}"))
        else:
            lines.append(Line("PASS", "T03 metrics",
                              "all required keys present"))

    # T04: artifacts present and healthy
    enr_dir = Path("data/enriched")
    pq = _latest_parquet(enr_dir)
    if pq is None:
        lines.append(Line("FAIL", "T04 artifacts", "no parquet in data/enriched"))
        return lines
    try:
        df = pd.read_parquet(pq)
    except Exception as exc:  # pragma: no cover - defensive
        lines.append(Line("FAIL", "T04 artifacts", f"read {pq.name} -> {exc!s}"))
        return lines

    if "source" not in df.columns:
        lines.append(Line("FAIL", "T04 artifacts", "missing 'source' column"))
        return lines

    # T05: OS_ANOM present on MND rows (or sparse fallback in metrics)
    mnd = df[df["source"].eq("MND")]
    if mnd.empty:
        lines.append(Line("WARN", "T05 OS_ANOM", "no MND rows in latest parquet"))
    else:
        cnt = int(mnd["corroborations"].apply(_contains_os_anom).sum())
        if cnt >= 1:
            lines.append(Line("PASS", "T05 OS_ANOM", f"MND rows with tag: {cnt}"))
        else:
            msg = "none; expect validation_sparse_fallbacks > 0 in metrics"
            lines.append(Line("WARN", "T05 OS_ANOM", msg))

    # T06: daily_grid_risk non-flat
    risk_csv = enr_dir / "daily_grid_risk.csv"
    if not risk_csv.exists():
        risk_csv = Path("data/enriched/daily_grid_risk.csv")
    if not risk_csv.exists():
        lines.append(Line("FAIL", "T06 grid_risk", "daily_grid_risk.csv missing"))
    else:
        try:
            dg = pd.read_csv(risk_csv)
            distinct = pd.to_numeric(dg["risk_score"],
                                     errors="coerce").round(3).nunique()
            if distinct > 1:
                lines.append(Line("PASS", "T06 grid_risk",
                                  f"distinct_rounded3={distinct}"))
            else:
                lines.append(Line("WARN", "T06 grid_risk", "appears flat"))
        except Exception as exc:  # pragma: no cover
            lines.append(Line("FAIL", "T06 grid_risk", f"read -> {exc!s}"))

    # T07: UI static policy checks (grep app.py)
    app_py = Path("app.py")
    if not app_py.exists():
        lines.append(Line("FAIL", "T07 app.py", "app.py not found"))
    else:
        src = app_py.read_text(encoding="utf-8", errors="replace")
        # anomaly chart uses MND window call
        pat = r"render_anomaly_chart\(_filter_window\(df\[df\['source']\]\s*==\s*\"MND\"\s*\]"
        if re.search(pat, src):
            lines.append(Line("PASS", "T07 anomaly_call", "uses MND window"))
        else:
            lines.append(Line("WARN", "T07 anomaly_call", "MND grep not found"))
        # caching present
        if "st.cache_data" in src:
            lines.append(Line("PASS", "T07 caching", "st.cache_data present"))
        else:
            lines.append(Line("WARN", "T07 caching", "cache decorator missing"))
        # map layers (optional)
        hints = sum(x in src for x in
                    ("HeatmapLayer", "HexagonLayer", "ScatterplotLayer"))
        if hints >= 1:
            lines.append(Line("PASS", "T07 map_layers", "deck.gl layers present"))
        else:
            lines.append(Line("WARN", "T07 map_layers", "fallback map only"))

    # T08: headless UI helpers (_watch_cells non-empty fallback)
    try:
        import app as _ui

        if hasattr(_ui, "_watch_cells") and hasattr(_ui, "_filter_window"):
            sub = df.copy()
            sub["dt"] = pd.to_datetime(sub["dt"], utc=True, errors="coerce")
            sub = sub.dropna(subset=["dt"])
            sub = _ui._filter_window(sub, 24)
            # force strict cutoff to trigger top_k fallback
            try:
                w = _ui._watch_cells(sub, cutoff=0.95, only_mnd=False)
                rows = len(w) if hasattr(w, "__len__") else 0
                if rows >= 1 or sub.empty:
                    lines.append(Line("PASS", "T08 watch_cells", f"rows={rows}"))
                else:
                    lines.append(Line("FAIL", "T08 watch_cells", "empty result"))
            except TypeError:
                # older signature without top_k
                w = _ui._watch_cells(sub, cutoff=0.95, only_mnd=False)
                rows = len(w) if hasattr(w, "__len__") else 0
                if rows >= 1 or sub.empty:
                    lines.append(Line("PASS", "T08 watch_cells", f"rows={rows}"))
                else:
                    lines.append(Line("FAIL", "T08 watch_cells", "empty result"))
        else:
            lines.append(Line("WARN", "T08 watch_cells",
                              "helpers not exposed; skipped"))
    except Exception as exc:  # pragma: no cover - defensive
        lines.append(Line("WARN", "T08 watch_cells", f"import app -> {exc!s}"))

    # T09: temp hygiene
    tmp_hit = None
    for p in Path("data").rglob("*"):
        name = p.name.lower()
        if any(k in name for k in ("tmp", "temp", ".parq.tmp", ".tmp")):
            tmp_hit = str(p)
            break
    if tmp_hit:
        lines.append(Line("WARN", "T09 hygiene", f"temp residue: {tmp_hit}"))
    else:
        lines.append(Line("PASS", "T09 hygiene", "no temp residue"))

    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify artifacts and UI policies."
    )
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="Run a short pipeline before checks.",
    )
    parser.add_argument(
        "--hours", type=int, default=1, help="Hours for pipeline if run."
    )
    parser.add_argument(
        "--prefix",
        default="verify",
        help="Out prefix for pipeline log/artifacts.",
    )
    args = parser.parse_args()

    results = verify(args.run_pipeline, args.hours, args.prefix)

    # Print matrix and compute exit code
    tally = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0}
    for line in results:
        tally[line.kind] += 1
        print(f"{line.kind:4}  {line.name:16}  {line.note}")

    print(
        f"Summary: PASS={tally['PASS']} FAIL={tally['FAIL']} "
        f"WARN={tally['WARN']} SKIP={tally['SKIP']}"
    )
    return 1 if tally["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
