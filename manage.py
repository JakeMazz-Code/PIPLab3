#!/usr/bin/env python3
"""Convenience CLI wrapper for Gray-Zone maintenance tasks."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _run_subprocess(label: str, args: List[str]) -> int:
    """Run a subprocess and emit a one-line status message."""
    result = subprocess.run(args, check=False)
    code = result.returncode
    print(f"[{label}] exit code {code}")
    return code


def run_pipeline(args: argparse.Namespace) -> int:
    """Run the main pipeline over the requested time window."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        "--hours",
        str(args.hours),
    ]
    if args.bbox:
        cmd.extend(["--bbox", str(args.bbox)])
    return _run_subprocess("run", cmd)


def run_history(args: argparse.Namespace) -> int:
    """Backfill history partitions for the requested range."""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "backfill.py"),
        "--days",
        str(args.days),
        "--sleep",
        str(args.sleep),
    ]
    if args.resume:
        cmd.append("--resume")
    return _run_subprocess("history", cmd)


def run_ui(args: argparse.Namespace) -> int:
    """Launch the Streamlit UI with a safety guard for missing deps."""
    try:
        import streamlit  # type: ignore
    except ModuleNotFoundError:
        print(
            "[ui] streamlit is not installed. Run `pip install streamlit` first.",
            file=sys.stderr,
        )
        return 1
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(REPO_ROOT / "app.py"),
    ]
    return _run_subprocess("ui", cmd)


def run_smoke(args: argparse.Namespace) -> int:
    """Execute the headless Streamlit smoke test."""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "ui_smoke.py"),
    ]
    return _run_subprocess("smoke", cmd)


def run_metrics(args: argparse.Namespace) -> int:
    """Print metrics from a previously generated log file."""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "print_metrics.py"),
        "--log",
        str(args.log),
    ]
    return _run_subprocess("metrics", cmd)


def run_simulation(args: argparse.Namespace) -> int:
    """Run Monte Carlo simulations via the main pipeline."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        "--mode",
        "simulate",
        "--runs",
        str(args.runs),
    ]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    return _run_subprocess("simulate", cmd)


def run_backtest(args: argparse.Namespace) -> int:
    """Score historical predictions using the backtest harness."""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "backtest.py"),
        "--days",
        str(args.days),
    ]
    return _run_subprocess("backtest", cmd)


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        description="Gray-Zone management helper",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Execute the main pipeline",
    )
    run_parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Pipeline lookback window in hours",
    )
    run_parser.add_argument(
        "--bbox",
        type=str,
        default=None,
        help="Bounding box 'latmin,lonmin,latmax,lonmax'",
    )
    run_parser.set_defaults(func=run_pipeline)

    history_parser = subparsers.add_parser(
        "history",
        help="Backfill history partitions",
    )
    history_parser.add_argument(
        "--days",
        type=int,
        default=10,
        help="Number of days to backfill",
    )
    history_parser.add_argument(
        "--sleep",
        type=float,
        default=1.5,
        help="Delay between day runs (seconds)",
    )
    history_parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip existing outputs",
    )
    history_parser.set_defaults(func=run_history)

    ui_parser = subparsers.add_parser(
        "ui",
        help="Launch the Streamlit UI",
    )
    ui_parser.set_defaults(func=run_ui)

    smoke_parser = subparsers.add_parser(
        "smoke",
        help="Run the UI smoke test",
    )
    smoke_parser.set_defaults(func=run_smoke)

    metrics_parser = subparsers.add_parser(
        "metrics",
        help="Print metrics from a log file",
    )
    metrics_parser.add_argument(
        "--log",
        required=True,
        help="Path to the log file",
    )
    metrics_parser.set_defaults(func=run_metrics)

    simulate_parser = subparsers.add_parser(
        "simulate",
        help="Run Monte Carlo simulations",
    )
    simulate_parser.add_argument(
        "--runs",
        type=int,
        default=500,
        help="Number of simulation runs",
    )
    simulate_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    simulate_parser.set_defaults(func=run_simulation)

    backtest_parser = subparsers.add_parser(
        "backtest",
        help="Score historical predictions",
    )
    backtest_parser.add_argument(
        "--days",
        type=int,
        default=10,
        help="Number of days to score",
    )
    backtest_parser.set_defaults(func=run_backtest)

    return parser


def main(argv: List[str] | None = None) -> int:
    """Entry point for the manage.py helper."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
