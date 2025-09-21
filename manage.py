#!/usr/bin/env python3
"""Convenience CLI wrapper for Gray-Zone maintenance tasks."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

PRESETS = {
    "CORE": "118,20,123,26",
    "WIDE": "115,18,126,28",
    "MAX": "112,16,128,30",
}


def _env_with_overrides(clear_key: bool = False,
                        extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Build a subprocess env; optionally clear the DeepSeek key."""
    env = os.environ.copy()
    if clear_key and "DEEPSEEK_API_KEY" in env:
        env.pop("DEEPSEEK_API_KEY", None)
    if extra:
        env.update(extra)
    return env


def _run_subprocess(label: str, args: List[str],
                    log_path: Optional[Path] = None,
                    append: bool = False,
                    env: Optional[Dict[str, str]] = None) -> int:
    """Run a subprocess from repo root, optionally tee to a log file."""
    print(f"[{label}] exec:", " ".join(args))
    out_handle = None
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            out_handle = log_path.open(mode, encoding="utf-8")
            proc = subprocess.run(
                args,
                cwd=REPO_ROOT,
                env=env,
                text=True,
                stdout=out_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        else:
            proc = subprocess.run(
                args,
                cwd=REPO_ROOT,
                env=env,
                text=True,
                check=False,
            )
        code = proc.returncode
        msg = f"[{label}] exit code {code}"
        if log_path is not None:
            msg += f" | log: {log_path}"
        print(msg)
        if log_path is not None:
            print(f"[{label}] wrote {log_path}")
        return code
    finally:
        if out_handle is not None:
            out_handle.close()


def run_pipeline(args: argparse.Namespace) -> int:
    """Run the main pipeline over a lookback window."""
    # Decide hours: --days overrides --hours if provided.
    hours = args.hours
    if args.days is not None:
        hours = max(1, int(args.days) * 24)

    # Bbox from preset or explicit value (mutually exclusive).
    bbox = None
    if args.preset:
        bbox = PRESETS[args.preset]
    if args.bbox:
        bbox = args.bbox  # explicit overrides preset if user insists

    # Default out-prefix and log path.
    prefix = args.out_prefix or f"run_{hours}h"
    log_path = Path(args.log) if args.log else REPO_ROOT / "logs" / f"{prefix}.log"

    cmd = [sys.executable, str(REPO_ROOT / "main.py"), "--hours", str(hours)]
    if bbox:
        cmd.extend(["--bbox", bbox])
    if args.out_prefix:
        cmd.extend(["--out-prefix", args.out_prefix])

    env = _env_with_overrides(clear_key=args.no_key,
                              extra={"AUTO_RETRY_MAX": str(args.auto_retry).lower()
                                     if args.auto_retry is not None else
                                     os.environ.get("AUTO_RETRY_MAX", "false")})
    return _run_subprocess(
        "run",
        cmd,
        log_path=log_path,
        append=args.append,
        env=env,
    )


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
    log_path = Path(args.log) if args.log else None
    env = _env_with_overrides(clear_key=args.no_key)
    return _run_subprocess(
        "history",
        cmd,
        log_path=log_path,
        append=args.append,
        env=env,
    )


def run_ui(args: argparse.Namespace) -> int:
    """Launch the Streamlit UI with a safety guard for missing deps."""
    try:
        import streamlit  # type: ignore
    except ModuleNotFoundError:
        print("[ui] streamlit not installed. Run: pip install streamlit",
              file=sys.stderr)
        return 1
    cmd = [sys.executable, "-m", "streamlit", "run", str(REPO_ROOT / "app.py")]
    log_path = Path(args.log) if args.log else None
    return _run_subprocess(
        "ui",
        cmd,
        log_path=log_path,
        append=args.append,
    )


def run_smoke(args: argparse.Namespace) -> int:
    """Execute the headless Streamlit smoke test."""
    cmd = [sys.executable, str(SCRIPTS_DIR / "ui_smoke.py")]
    log_path = Path(args.log) if args.log else None
    return _run_subprocess(
        "smoke",
        cmd,
        log_path=log_path,
        append=args.append,
    )


def run_metrics(args: argparse.Namespace) -> int:
    """Print metrics from a previously generated log file."""
    cmd = [sys.executable, str(SCRIPTS_DIR / "print_metrics.py"),
           "--log", str(args.log)]
    return _run_subprocess("metrics", cmd)


def run_simulation(args: argparse.Namespace) -> int:
    """Run Monte Carlo simulations via the main pipeline."""
    cmd = [sys.executable, str(REPO_ROOT / "main.py"),
           "--mode", "simulate", "--runs", str(args.runs)]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    log_path = Path(args.log) if args.log else None
    return _run_subprocess(
        "simulate",
        cmd,
        log_path=log_path,
        append=args.append,
    )


def run_backtest(args: argparse.Namespace) -> int:
    """Score historical predictions using the backtest harness."""
    cmd = [sys.executable, str(SCRIPTS_DIR / "backtest.py"),
           "--days", str(args.days)]
    log_path = Path(args.log) if args.log else None
    return _run_subprocess(
        "backtest",
        cmd,
        log_path=log_path,
        append=args.append,
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        description="Gray-Zone management helper",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    run_parser = subparsers.add_parser("run", help="Execute the main pipeline")
    run_parser.add_argument("--hours", type=int, default=24,
                            help="Lookback window in hours (default 24)")
    run_parser.add_argument("--days", type=int, default=None,
                            help="Alternative to --hours (days*24)")
    run_parser.add_argument("--bbox", type=str, default=None,
                            help="Bounding box 'latmin,lonmin,latmax,lonmax'")
    run_parser.add_argument("--preset", type=str, choices=tuple(PRESETS),
                            help="BBox preset: CORE|WIDE|MAX")
    run_parser.add_argument("--out-prefix", type=str, default=None,
                            help="Filename prefix for artifacts/logs")
    run_parser.add_argument("--log", type=str, default=None,
                            help="Optional log path (default logs/<prefix>.log)")
    run_parser.add_argument("--append", action="store_true",
                            help="Append to the log path instead of clobbering")
    run_parser.add_argument("--no-key", action="store_true",
                            help="Run without DEEPSEEK_API_KEY (fallback)")
    run_parser.add_argument("--auto-retry", type=bool, default=None,
                            help="Set AUTO_RETRY_MAX=true/false for this run")
    run_parser.set_defaults(func=run_pipeline)

    # history
    history_parser = subparsers.add_parser("history",
                                           help="Backfill history partitions")
    history_parser.add_argument("--days", type=int, default=10,
                                help="Number of days to backfill")
    history_parser.add_argument("--sleep", type=float, default=1.5,
                                help="Delay between day runs (seconds)")
    history_parser.add_argument("--resume", action="store_true",
                                help="Skip existing outputs")
    history_parser.add_argument("--log", type=str, default=None,
                                help="Optional log path")
    history_parser.add_argument("--append", action="store_true",
                                help="Append to the log path instead of clobbering")
    history_parser.add_argument("--no-key", action="store_true",
                                 help="Run without DEEPSEEK_API_KEY")
    history_parser.set_defaults(func=run_history)

    # ui
    ui_parser = subparsers.add_parser("ui", help="Launch the Streamlit UI")
    ui_parser.add_argument("--log", type=str, default=None,
                           help="Optional log path")
    ui_parser.add_argument("--append", action="store_true",
                           help="Append to the log path instead of clobbering")
    ui_parser.set_defaults(func=run_ui)

    # smoke
    smoke_parser = subparsers.add_parser("smoke",
                                         help="Run the UI smoke test")
    smoke_parser.add_argument("--log", type=str, default=None,
                              help="Optional log path")
    smoke_parser.add_argument("--append", action="store_true",
                              help="Append to the log path instead of clobbering")
    smoke_parser.set_defaults(func=run_smoke)

    # metrics
    metrics_parser = subparsers.add_parser("metrics",
                                           help="Print metrics from a log")
    metrics_parser.add_argument("--log", required=True,
                                help="Path to the log file")
    metrics_parser.set_defaults(func=run_metrics)

    # simulate
    simulate_parser = subparsers.add_parser("simulate",
                                            help="Run Monte Carlo sims")
    simulate_parser.add_argument("--runs", type=int, default=500,
                                 help="Number of simulation runs")
    simulate_parser.add_argument("--seed", type=int, default=None,
                                 help="Random seed")
    simulate_parser.add_argument("--log", type=str, default=None,
                                 help="Optional log path")
    simulate_parser.add_argument("--append", action="store_true",
                                 help="Append to the log path instead of clobbering")
    simulate_parser.set_defaults(func=run_simulation)

    # backtest
    backtest_parser = subparsers.add_parser("backtest",
                                            help="Score historical preds")
    backtest_parser.add_argument("--days", type=int, default=10,
                                 help="Number of days to score")
    backtest_parser.add_argument("--log", type=str, default=None,
                                 help="Optional log path")
    backtest_parser.add_argument("--append", action="store_true",
                                 help="Append to the log path instead of clobbering")
    backtest_parser.set_defaults(func=run_backtest)

    return parser


def main(argv: List[str] | None = None) -> int:
    """Entry point for the manage.py helper."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
