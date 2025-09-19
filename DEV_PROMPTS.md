# DEV_PROMPTS.md
Guided prompts for AI coding agents (Codex/Copilot/etc.) to work on this
repo safely. Paste **System** + **User** blocks into your agent. Each patch
is **surgical** (small, targeted) and respects our public API and style.

> **Placement**: add this file to your repo at `dev/codex_prompts.md`  
> (agents will read the nearest AGENTS.md and this file).

---

## Global guardrails (paste as the System prompt for all patches)

```
You are a senior Python engineer. Produce minimal, robust, PEP‑8 compliant
code with max line length 79, full type hints, and docstrings. Do not change
any public function names or signatures unless explicitly instructed. Use
only approved baseline deps: pandas, requests, tenacity, pydantic,
python-dotenv, pyarrow, beautifulsoup4, lxml, matplotlib. For UI extension,
streamlit is allowed. Use requests with timeouts and tenacity retry/backoff.
Use UTC everywhere. Never hardcode secrets; read from environment/.env.
Write files atomically (temp file + os.replace). Comments explain "why".
All modules must import cleanly offline (no network on import).
```

**Public API (do not change)**

- `main.py`  
  `extract_opensky(hours: int = 6, bbox: list | None = None) -> pd.DataFrame`  
  `scrape_mnd() -> list[dict]`  
  `clean_merge(os_df: pd.DataFrame, mnd_rows: list[dict]) -> pd.DataFrame`  
  `simulate_air_ops(df_enriched: pd.DataFrame, n_runs: int = 100, seed: int | None = None) -> pd.DataFrame`

- `deepseek_enrichment.py`  
  `_call_deepseek_batch(texts: list[str]) -> list[dict]`  
  `enrich_incidents(df: pd.DataFrame) -> pd.DataFrame`  
  `summarize_theater(df: pd.DataFrame, horizon: str = "24h") -> str`

**Canonical columns (must exist where applicable)**

`["dt","lat","lon","source","raw_text","country","grid_id"]`  
Enrichment should add:  
`["category","severity_0_5","risk_score","actors","geo_quality","weapon_class","summary_one_line","where_guess","needs_review"]`

**Metrics line (end-of-run)**

```
METRICS | opensky_points=... mnd_rows=... merged_rows=... enriched_rows=...
llm_success=... llm_invalid_json=... llm_retries=... needs_review_count=...
validation_sparse_fallbacks=... os_anom_rows=... wall_ms=...
```

---

## Repo self‑awareness probes (sanity checks)

### Probe A — Can the agent see `AGENTS.md`?
**User**
```
List top-level files in the repo root. If AGENTS.md exists, read it and
summarize: (1) public APIs, (2) allowed deps & policy, (3) metrics line,
(4) extension constraints. Report any missing sections or violations
(e.g., non-PEP8 hints, signature drift). Propose concrete fixes (diff-like).
If AGENTS.md is missing, output a minimal AGENTS.md draft consistent with
the README and return a create-file plan.
```

### Probe B — Can the agent find this `dev/codex_prompts.md`?
**User**
```
Locate dev/codex_prompts.md (or DEV_PROMPTS.md). If found, summarize all
patch IDs and their acceptance criteria. Explain where the file lives in
the repo and propose the next two patches to run, with reasoning based on
current artifacts (data/enriched/*.parquet; daily_grid_risk.csv presence;
examples/*). If not found, propose the path and create the file with the
templates in your answer.
```

> Run these before any edits. Paste results into your PR description.

---

## Patch A — OpenSky “WIDE + Local Anomaly” (implemented in repo; reference)
**Purpose**: Reduce sparsity by extracting with a wider bbox while keeping
**local** anomaly scoring around CORE grids.

**Files allowed**: `main.py`, small utilities within it, README notes.

**User**
```
Goal
Upgrade OpenSky extraction and validation: use WIDE bbox by default when
--bbox is not supplied; compute local z-score anomalies per MND grid using
a 1° neighborhood (±2 cells at 0.5°). Keep all public signatures unchanged.

Behavior
- BBoxes (internal presets):
  CORE: 118,20,123,26
  WIDE: 115,18,126,28  (default extraction if bbox=None)
  MAX:  112,16,128,30  (optional fallback behind env flag AUTO_RETRY_MAX)

- Validation
  For each MND grid, compute hourly local z-score using neighbor grids.
  If neighbor points < 15, fallback to CORE-hour deviation; if still sparse,
  set validation_score=1.0, corroborations=[], increment counter.

- Logging
  Print the METRICS line with counters including os_anom_rows.

Acceptance
- No public signature change; PEP-8 79 cols.
- WIDE default used (no --bbox) and more OS_ANOM coverage than CORE run.
- Fallbacks logged but rare; pipeline stays within timeouts.
```

---

## Patch A.1 — Metrics consolidation (recommended)
**Purpose**: Ensure every run emits a single, parseable metrics line.

**Files allowed**: `main.py`

**User**
```
Add end-of-run METRICS line:
opensky_points, mnd_rows, merged_rows, enriched_rows, llm_success,
llm_invalid_json, llm_retries, needs_review_count,
validation_sparse_fallbacks, os_anom_rows, wall_ms per phase.

Implement a small helper to accumulate counters and render a flat string.
No signature changes. PEP-8 and 79 cols.
```

---

## Patch A.2 — README bbox defaults + auto-retry flag (recommended)
**Purpose**: Document extraction presets and optional MAX retry.

**Files allowed**: `README.md`

**User**
```
Add a "BBoxes & Validation" section:
- Extraction default when --bbox omitted: WIDE (115,18,126,28).
- Scoring: local anomalies computed around CORE grids even if extracting WIDE.
- Optional AUTO_RETRY_MAX env flag to retry extraction once with MAX if
  opensky_points < threshold (document default; do not enable by default).
Keep concise and demo-focused.
```

---

## Patch G — `where_guess` → grid mapping (optional but great for UI)
**Purpose**: Reduce RNaNCNaN grid_ids on MND rows using AI text hints.

**Files allowed**: `main.py`

**User**
```
Add pure helper mnd_where_to_grid(where_guess: str | None) -> str.
- Parse tokens: "north/south/east/west", "median line", "taipei", "kinmen",
  "pratas", "matsu", "bashi". Map to representative 0.5° grid cells inside
  CORE bbox; "median line" maps to a strip near 121E, 23–25N.
- If ambiguous or missing: return "RNaNCNaN".

In clean_merge(...), set grid_id for MND rows with missing lat/lon using
mnd_where_to_grid(where_guess). Keep lat/lon None.

Acceptance
- Distinct grid_ids in daily_grid_risk.csv increases (>1).
- No third traditional data source introduced.
```

---

## Patch J — DeepSeek JSON hardening (idempotent)
**Purpose**: Make enrichment robust to malformed JSON; preserve traceability.

**Files allowed**: `deepseek_enrichment.py`

**User**
```
Improve _call_deepseek_batch and parsing:
1) First pass: strict json.loads; if failure, attempt common repairs
   (strip backticks, remove trailing commas). Validate with pydantic model.
2) Retry API once if parse fails; on second failure:
   - needs_review=True
   - derive severity_0_5 from keywords if missing; clamp to 0..5
   - compute risk_score fallback in [0,1] (e.g., 0.18*severity + 0.02)
3) Store _raw_response for each item to aid audits (optional column).

Acceptance
- Invalid JSON rate after one retry <= 10%; needs_review <= 10%.
- No signature changes; PEP-8.
```

---

## Patch M — Validation tagging & counters
**Purpose**: Ensure corroborations carry a numeric anomaly and logs count them.

**Files allowed**: `main.py`

**User**
```
When a non-default validation result exists, append a corroboration tag
"OS_ANOM:<value>" with value rounded to 2 decimals. Count rows containing
this tag as os_anom_rows for METRICS. Keep current fallbacks. No signature
changes.
```

---

## Patch I — Atomic writes enforcement (quick)
**Purpose**: Avoid partial files on crash.

**Files allowed**: `main.py`

**User**
```
Refactor all file outputs to atomic writes:
- write to tmp path in same dir, fsync, os.replace(tmp, final)
- handle Parquet + CSV + examples/*.md
- add a tiny _atomic_write_text() and _atomic_write_df() helper.
No signature changes.
```

---

## Patch S1 — Streamlit Core Demo UI
**Purpose**: Presentation-ready demo reading only local artifacts.

**Files allowed**: **add** `app.py` (do not edit baseline files)

**User**
```
Create app.py. Read-only from:
- latest data/enriched/incidents_enriched_*.parquet (by mtime)
- data/enriched/daily_grid_risk.csv
- examples/airops_brief_24h.md (if present)

UI spec
- Sidebar: time window (6..48h, default 24), risk cutoff (0..1, default 0.35)
  note about display bbox "CORE: 118,20,123,26". Checkbox "Only MND cells".
- KPI row: incidents last N hours; grids over cutoff; LLM risk avg; MND %
  with corroboration.
- Map heatmap: pydeck GridLayer/HeatmapLayer using grid centroids derived
  from "R{row}C{col}" at 0.5°; filter to latest day in daily_grid_risk.csv.
- Watch Cells: top 10 grids by risk over window (risk >= cutoff) with
  grid_id, mean risk, last_seen, #incidents, sample actors.
- MND table: dt, grid_id, category, severity_0_5, risk_score, actors,
  summary_one_line, validation_score, corroborations.
- LLM Brief: show examples/airops_brief_24h.md if present else import and
  call summarize_theater on last 24h (no network).

- Downloads: two buttons to download latest parquet and daily_grid_risk.csv.
- Footer: "Prototype | Civil ADS‑B proxy + MoD bulletins + AI enrichment".

Acceptance
- streamlit run app.py launches with no external calls.
- Map + KPIs + tables render; handles missing examples gracefully.
```

---

## Patch S2 — Scenario Player + Backtest & Polish
**Purpose**: Add 6‑hour playback and a 10‑day backtest script.

**Files allowed**: **edit** `app.py`; **add** `scripts/backtest.py`

**User**
```
In app.py:
- Add "Scenario Player (6h)" expander with play/pause and a time slider in
  30-min steps over last 6 hours. On play, auto-advance using session_state
  and experimental_rerun.
- Recompute map/watch-cells windowed over the slider interval.
- Add a small matplotlib chart of mean validation_score for selected grids.

Add scripts/backtest.py:
- CLI: python scripts/backtest.py --days 10 [--cutoff 0.35 | --top-k 12]
- For each day D with data, build forecast risk from D-3..D-1 (mean or exp‑
  weighted). Select hotspots via cutoff or top-k; compare to D-day MND rows
  (same grid_id). Compute confusion matrix and metrics (precision, recall,
  F1, Heidke Skill Score, Brier).
- Outputs: examples/backtest_confusion_matrix.png, examples/skill_metrics.json

Acceptance
- streamlit run app.py shows scenario player and updates views.
- python scripts/backtest.py --days 10 writes both outputs.
```

---

## Patch T — Lightweight tests & self-checks (optional but helpful)
**Purpose**: Guard critical invariants without heavy frameworks.

**Files allowed**: `tests/` (new), small fixtures under `tests/fixtures/`

**User**
```
Add pytest-free (or minimal pytest) checks:
- latlon_to_grid unit tests (incl. edge cases, None -> RNaNCNaN).
- Simulation monotonicity: increasing risk deciles do not reduce mean
  eta_hours or disruption.
- JSON parse stub: monkeypatch _call_deepseek_batch to return one good and
  one malformed item; verify needs_review logic and fallback risk.
- Validation fallback: with a tiny OpenSky fixture, ensure local->CORE->
  default fallback chain produces expected validation_score/corroborations.
```

---

## Patch D — Docs polish (README/DEEPSEEK_USAGE/AI_USAGE)
**Purpose**: Keep demo narrative crisp.

**Files allowed**: `README.md`, `DEEPSEEK_USAGE.md`, `AI_USAGE.md`

**User**
```
- README: add "BBoxes & Validation" and "Demo Flow" sections (3-5 bullets).
- DEEPSEEK_USAGE: include the exact strict JSON system prompt, examples of
  success/failure payloads, the fallback mapping from severity->risk.
- AI_USAGE: what was AI-generated vs human-written, bugs fixed, lessons.
Keep each edit concise; retain existing sections.
```

---

## One-shot utility prompts

### “Run & compare” quick log probe
**User**
```
Run baseline twice:
1) python main.py --hours 6 --out-prefix core --bbox 118,20,123,26
2) python main.py --hours 6 --out-prefix wide
Collect the METRICS lines and compute deltas for opensky_points, os_anom_rows,
validation_sparse_fallbacks. Summarize which is better and by how much.
```

### “Artifacts sanity” probe
**User**
```
Load latest enriched parquet and daily_grid_risk.csv. Report:
- shape, column list, dtypes summary
- number of MND rows and % with corroborations containing 'OS_ANOM'
- top 10 grids by risk and a sample of actors per grid
- any mixed dtypes in severity_0_5 or risk_score (should be numeric)
Provide a short remediation plan if issues found.
```

---

## Using optional dependencies (policy summary)

You **may add** dependencies if they satisfy the checklist in AGENTS.md.
Prefer extras files (`requirements-ui.txt`, `requirements-dev.txt`), lazy
imports with fallbacks, permissive licenses, and stable outputs. Candidate
libs: `orjson`, `duckdb`, `typer`/`click`, `rich`, `tqdm`, `scikit-learn`,
`plotly` (UI only), `fastparquet`. Avoid heavy crawlers and headless
browsers unless explicitly justified.

---

## Maintainer acceptance gates (pre-UI and post-UI)

- Schema stable; enrichment fields present; numeric dtypes OK.
- Invalid-JSON after one retry ≤ 10%; needs_review ≤ 10%.
- ≥ 80% of MND rows have non-default validation or logged sparse fallback.
- Simulation monotonicity holds.
- Two same-seed runs are identical (ignoring timestamped filenames).
- Streamlit reads only local artifacts; backtest writes both outputs.
