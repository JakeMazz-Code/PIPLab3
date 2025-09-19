# Air & Sea Gray-Zone Monitor — Agents Guide

Purpose
This file gives AI coding agents (and humans) the exact context, rules,
and commands to work on this repository safely and productively. Treat it
like a contract: follow the invariants and the public API and the project
will remain stable and demo-ready.

---

## TL;DR (for agents)

- Do not change public function signatures or baseline file names.
- Two traditional sources only: OpenSky `/states/all` (API) and Taiwan MND
  daily "PLA activities" bulletins (scrape).
  Third source = AI (DeepSeek) for enrichment only.
- Style: PEP-8, max line length 79, docstrings, type hints, 4-space indents.
- I/O: UTC everywhere; requests + tenacity (timeouts + backoff); write files
  atomically (temp file then os.replace).
- Data dirs that must exist: `data/raw/`, `data/enriched/`, `examples/`.
- LLM JSON: strict schema; on invalid JSON retry once, then set
  `needs_review=True` with a safe fallback `risk_score`.
- Dependencies: You may add new libraries if they meet the checklist
  below. Keep the baseline minimal.

---

## Project overview

A minimal, robust ETL + LLM enrichment + simple simulation focused on
Taiwan gray-zone air/sea activity.

Outputs:
- Raw artifacts: OpenSky JSON, MND HTML.
- Enriched incidents: Parquet (CSV fallback).
- Daily grid risk: CSV heatmap table.
- Examples: 24-hour AI brief and a before/after enrichment sample.
- Simulation: risk-sensitive ETA/disruption Monte Carlo runs.

---

## Repository layout

If the repo root already contains these files, treat repo root as the
project root. If a `grayzone-tw/` directory is present, it contains the
same structure.

```
grayzone-tw/
├── main.py
├── deepseek_enrichment.py
├── data/
│   ├── raw/
├── examples/
├── requirements.txt
├── README.md
├── DEEPSEEK_USAGE.md
├── AI_USAGE.md
└── .gitignore
```

Do not rename these baseline files.
Streamlit/backtest extension adds new files only (see "Extensions").

---

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
. .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` in the project root:

```bash
# .env
DEEPSEEK_API_KEY=sk-...      # required for enrichment
OPENSKY_USER=your_user       # optional
OPENSKY_PASS=your_pass       # optional
AUTO_RETRY_MAX=false         # optional; gating for MAX bbox retry
```

---

## Run commands (baseline)

End-to-end ETL + enrich + examples:

```bash
python main.py --hours 24 --bbox 118,20,123,26 --out-prefix demo
```

Artifacts produced:
- data/raw/opensky_YYYYMMDDHH.json
- data/raw/mnd_YYYYMMDDHH.html
- data/enriched/incidents_enriched_*.parquet (CSV fallback)
- data/enriched/daily_grid_risk.csv
- examples/airops_brief_24h.md

Rebuild examples only (from latest enriched file):

```bash
python main.py --mode artifacts
```

Simulation runs:

```bash
python main.py --mode simulate --runs 500 --seed 42
# -> data/enriched/simulation_runs.csv
```

Dry-run mode (optional for tests; if implemented, no network):

```bash
python main.py --hours 6 --out-prefix dry --dry-run
```

---

## Public API (do not change signatures)

main.py
```python
extract_opensky(hours: int = 6, bbox: list | None = None) -> pd.DataFrame
scrape_mnd() -> list[dict]
clean_merge(os_df: pd.DataFrame, mnd_rows: list[dict]) -> pd.DataFrame
simulate_air_ops(
    df_enriched: pd.DataFrame,
    n_runs: int = 100,
    seed: int | None = None
) -> pd.DataFrame
```

deepseek_enrichment.py
```python
_call_deepseek_batch(texts: list[str]) -> list[dict]
enrich_incidents(df: pd.DataFrame) -> pd.DataFrame
summarize_theater(df: pd.DataFrame, horizon: str = "24h") -> str
```

---

## Dependency policy and checklist (adding libraries is OK)

We keep the baseline minimal for portability and grading, but you are
allowed to add dependencies if they meet this checklist:

Decision checklist (must meet all):
1) Justification: clear benefit (performance, UX, maintainability, or
   analytics) that the baseline cannot easily achieve.
2) Scope: the new lib is used in extensions (UI/backtest/analysis) or
   optional code paths; it must not be required for the baseline ETL to run.
3) No new traditional sources: do not add data sources beyond OpenSky + MND.
4) Version pin: keep requirements.txt for baseline; add optional deps to
   requirements-ui.txt or requirements-dev.txt.
5) Offline import: the repo must import without network. Use lazy imports
   and fallbacks:
   ```python
   try:
       import orjson as _json
   except Exception:
       import json as _json
   ```
6) License: permissive OSS (MIT/BSD/Apache-2.0) or otherwise compatible.
7) Security and size: avoid headless browsers and heavyweight stacks unless
   absolutely necessary. Prefer small, well-maintained libs.
8) Reproducibility: outputs, metrics, and public APIs remain stable. Add a
   short note in README if behavior changes.
9) Testability: add a small self-check or unit test or a CLI snippet.

Candidate optional dependencies (examples):
- Performance/parsing: orjson, ujson, python-rapidjson
- CLI & UX: typer or click, rich, tqdm
- Data & analysis: duckdb, fastparquet
- Metrics/backtest: scikit-learn (or compute metrics manually)
- UI/visualization: plotly (Streamlit only), folium (optional)
- Geo (sparingly): shapely, pyproj, geopandas (optional, heavy)

Not recommended (unless explicitly justified):
- Headless browsers (selenium, playwright) for MND scraping
- Full crawler frameworks (scrapy)
- Cloud SDKs or managed DB clients that require credentials
- Anything that introduces a new traditional data source

How to add (process):
1) Add the pinned lib to requirements-ui.txt or requirements-dev.txt.
2) Import lazily with fallbacks so baseline still runs without it.
3) Update README with install note:
   pip install -r requirements-ui.txt
4) Add a tiny self-check or test that exercises the new path.
5) Ensure the METRICS line and baseline ETL are unaffected.

---

## Data sources and extraction

OpenSky `/states/all`:
- Query with lamin, lomin, lamax, lomax
- Optional Basic Auth via OPENSKY_USER, OPENSKY_PASS
- Save raw JSON to data/raw/opensky_YYYYMMDDHH.json

Patch A behavior:
- If --bbox is not passed, default to WIDE for extraction.
  - CORE: 118,20,123,26 (display/scoring focus)
  - WIDE (default extraction): 115,18,126,28
  - MAX (optional fallback when enabled by env): 112,16,128,30
- Local anomaly scoring still uses CORE + neighboring grids.

Taiwan MND daily "PLA activities":
- Scrape daily post/table: date, aircraft/ship counts, crossings, notes
- Save raw HTML to data/raw/mnd_YYYYMMDDHH.html

---

## Transform and canonical schema

Normalize to these columns at minimum:

```
["dt","lat","lon","source","raw_text","country","grid_id"]
```

- dt: timezone-aware UTC timestamp
- lat/lon: floats or None
- source: "OS" (OpenSky) or "MND" (bulletin)
- raw_text: original source snippet
- country: best guess (ISO or descriptive string)
- grid_id: stable 0.5 degree bin like "R{row}C{col}"

Grid math (pure, unit-testable):
```python
latlon_to_grid(lat: float | None, lon: float | None, step: float = 0.5) -> str
```

If MND lacks coordinates, you may map AI where_guess to a coarse grid
("north/south/east/west/median line"). This stays within the AI-only rule.

---

## Enrichment (DeepSeek)

Read key from DEEPSEEK_API_KEY. Batch calls (<= 20 texts per batch).
Expose _call_deepseek_batch(texts) for monkeypatching in tests.

System prompt (must match exactly):
```
STRICT JSON only, keys exactly:
category (enum: Piracy|UAV/Drone|Missile|Boarding|Seizure|Mine|
Electronic Interference|Other|null)
actors (list[str])
weapon_class (str|null)
severity_0_5 (int 0-5)
risk_score (float 0-1)
summary_one_line (str)
where_guess (str|null)
geo_quality (high|medium|low|null)
Never add extra keys; use null when unsure.
```

On invalid JSON:
- Retry once.
- If still invalid, set needs_review=True and compute a safe fallback
  risk_score in [0, 1] from severity_0_5 or severity keywords.

Enriched fields added:
```
["category","severity_0_5","risk_score","actors","geo_quality",
 "weapon_class","summary_one_line","where_guess","needs_review"]
```

Summaries:
summarize_theater(df, horizon="24h") returns a concise 5-8 sentence
brief for the last 24h using categories, actors, and risk by grid.

---

## Validation (OpenSky corroboration)

Compute a per-grid, per-hour density from OpenSky points.

- Preferred: local z-score for each MND grid using a 1 degree neighborhood
  (+/- 2 steps at 0.5 degree) within the same hour across the window.
- If local neighbors sparse (< ~15 points): fallback to CORE bbox hourly
  deviation.
- If still sparse: set validation_score=1.0 and corroborations=[].

Attach to MND rows:
- validation_score (float; z or normalized deviation)
- corroborations (e.g., tag "OS_ANOM:<value>")

---

## Load layer

- Enriched table -> data/enriched/incidents_enriched_*.parquet
  (CSV fallback if pyarrow missing)
- Daily grid risk (mean risk per day x grid) ->
  data/enriched/daily_grid_risk.csv
- Examples:
  - examples/airops_brief_24h.md
  - examples/mnd_before_after_001.md

---

## Simulation contract

```python
simulate_air_ops(df_enriched, n_runs=100, seed=None) -> pd.DataFrame
# columns: ["run_id","eta_hours","disruption"]
```

- Deterministic when seed is set
- Risk-sensitive: higher mean risk_score must not yield lower average
  eta_hours or disruption across runs (monotonic trend)

---

## Metrics (log once per run)

Agents must keep or improve a final summary line:

```
METRICS | opensky_points=... mnd_rows=... merged_rows=... enriched_rows=...
llm_success=... llm_invalid_json=... llm_retries=... needs_review_count=...
validation_sparse_fallbacks=... os_anom_rows=... wall_ms=...
```

These counters support demos and regressions.

---

## Quick self-tests (no pytest required)

After a run (examples):

```python
import pandas as pd, pathlib as p
parq = sorted(p.Path("data/enriched").glob("*.parquet"))[-1]
df = pd.read_parquet(parq)
mnd = df[df["source"].eq("MND")]
nz = mnd["corroborations"].fillna("").str.contains("OS_ANOM")
print("MND:", len(mnd), "OS_ANOM rows:", int(nz.sum()), f"({nz.mean():.0%})")
```

Daily risk should not be flat:

```python
dg = pd.read_csv("data/enriched/daily_grid_risk.csv")
print("Grids:", dg["grid_id"].nunique(),
      "Distinct risk:", dg["risk_score"].round(3).nunique())
```

Monotonicity (informal):

```python
import numpy as np
dec = pd.qcut(df["risk_score"], 10, labels=False, duplicates="drop")
print(df.groupby(dec)["risk_score"].mean().round(3).to_list())
```

---

## Security and ethics

- Never hardcode secrets. Read from environment or `.env`.
- Respect source sites: set timeouts, retries, and limit scrape frequency.
- This system uses civil ADS-B as a proxy signal; not a military feed.
  Communicate limitations clearly in UI/docs.

---

## What agents may change

- Internal logic in main.py / deepseek_enrichment.py to improve robustness,
  JSON parsing, validation math, logging, or atomic writes.
- Add pure helpers (same files) provided public signatures stay intact.
- Add optional tests or small fixtures for offline runs.
- Add optional dependencies following the checklist and extras pattern.

## What agents must not change

- Public function names/signatures listed above.
- Baseline file names and repo structure.
- The DeepSeek system prompt keys and JSON contract.
- Introduce any new traditional data sources beyond OpenSky + MND.

---

## Extensions (add-only files)

Implement without editing baseline files:

1) Streamlit UI — app.py
   - Controls: bbox (display), time window (last N hours), risk cutoff.
   - Panels: heatmap (from daily_grid_risk.csv), watch cells,
     MND table, OpenSky anomaly chart, LLM brief.
   - Download buttons: latest incidents_enriched.parquet,
     daily_grid_risk.csv.
   - Read-only from data/enriched/ and examples/.

2) Backtest — scripts/backtest.py
   - Build D-day risk from D-3..D-1; evaluate vs D-day MND rows.
   - Outputs: examples/backtest_confusion_matrix.png,
     examples/skill_metrics.json.

Run:
```bash
streamlit run app.py
python scripts/backtest.py --days 10
```

---

## Patch references (for agents)

Keep a dev/codex_prompts.md with these intents:

- Patch A — OpenSky WIDE + Local Anomaly
  Extract with WIDE by default; score anomalies locally around CORE grids;
  log counters; optional one-shot retry with MAX when sparse.

- Patch A.1 — METRICS consolidation
  One final METRICS line including os_anom_rows and wall_ms.

- Patch S1 — Streamlit Core Demo UI
  Read-only from enriched artifacts; heatmap + KPIs + brief.

- Patch S2 — Scenario Player and Backtest
  6-hour playback; backtest over last N days; write confusion matrix and
  skill metrics; add UX polish (disclaimers, KPIs).

---

## Common pitfalls and remedies

- All grid_id = RNaNCNaN: MND lacks coords. Consider optional
  where_guess -> grid mapping.
- Flat daily_grid_risk: ensure severity/risk varies; confirm OS anomaly
  attaches to MND.
- Sparse validation: use WIDE extraction (default) but local anomaly for
  scoring with thresholds and fallbacks.
- Half-written files after crash: use atomic write pattern (temp +
  os.replace).
