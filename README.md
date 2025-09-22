# Gray‑Zone Air & Sea Monitor (Taiwan)

A reproducible **data pipeline + Streamlit dashboard** for tracking
**air/sea activity near Taiwan**. The system ingests **OpenSky** state
vectors and **Taiwan MND** daily bulletins, performs optional **LLM
enrichment** with DeepSeek, computes a per‑grid **risk surface**, writes
versioned artifacts, and renders an interactive **Monitor/History** UI.

---

## Table of contents

- [What this is](#what-this-is)
- [Data sources](#data-sources)
- [How it works (pipeline)](#how-it-works-pipeline)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Windows (PowerShell)](#windows-powershell)
  - [macOS / Linux (bash)](#macos--linux-bash)
  - [.env file format](#env-file-format)
- [Runbook](#runbook)
  - [Quickstart (24h run + UI)](#quickstart-24h-run--ui)
  - [Build N‑day history](#build-n%E2%80%91day-history)
  - [Common commands](#common-commands)
- [Environment variables](#environment-variables)
- [Artifacts and file conventions](#artifacts-and-file-conventions)
- [Metrics & logs](#metrics--logs)
- [Dashboard guide](#dashboard-guide)
- [Verification checklist](#verification-checklist)
- [Troubleshooting](#troubleshooting)
- [Development notes](#development-notes)
- [Security, privacy, and attribution](#security-privacy-and-attribution)
- [FAQ](#faq)

---

## What this is

- **Purpose.** Provide a simple, explainable view of gray‑zone activity
  (air and sea) in/around Taiwan by blending observed aircraft positions
  (OpenSky) with structured interpretations of official daily reports
  from the Taiwan MND (Ministry of National Defense).

- **Design goals.** Reproducibility, small/no external dependencies, two
  commands to usable output, and a **path‑stable** repo that behaves the
  same no matter where you launch it from (Windows/macOS/Linux).

- **What you get.** A local pipeline that produces human‑readable logs,
  versioned artifacts under `data/`, and a Streamlit app with a **Monitor**
  tab (fresh window) and a **History** tab (N‑day playback).

> The UI is read‑only (no network calls). All data is fetched by the
> pipeline and read from local files. pydeck is optional; maps fall back
> to simple points if hex/heat cannot be drawn.

---

## Data sources

1. **OpenSky** — state vectors (aircraft positions) inside a Taiwan‑centric
   bounding box. The code prefers the **official Python client** when
   available; otherwise it uses the **OAuth REST** API. You will need an
   OpenSky API client (client id/secret) for consistent results.

2. **Taiwan MND daily bulletins** — scraped across **multiple pages**
   (configurable), parsed into cards, and **deduped** by `(dt, title)`.
   These are public bulletins summarizing air/sea activity.

3. **DeepSeek (optional)** — an LLM used to convert raw MND text to
   **structured JSON** (severity `0–5`, risk proxy, actors, one‑line
   summary, validation hints). The pipeline runs without a key and
   falls back to safe defaults, but the structured view is far richer
   when a key is provided.

---

## How it works (pipeline)

```
            ┌────────────┐     ┌──────────────┐
            │  OpenSky   │     │    MND       │
            │ (client/   │     │  bulletins   │
            │  REST)     │     └──────┬───────┘
            └──────┬─────┘            │  scrape pages
                   │  state vectors    │  parse+dedupe
     bbox (CORE→WIDE→MAX retry)        ▼
                   │             ┌──────────────┐
                   ▼             │  MND (raw)   │
            ┌────────────┐       └──────┬───────┘
            │ OpenSky DF │              │  (optional)
            └──────┬─────┘              ▼
                   │             ┌──────────────┐
                   │             │  DeepSeek    │
                   │             │ enrichment   │
                   │             └──────┬───────┘
                   │                    │ JSON w/ severity, actors, brief
                   ▼                    ▼
            ┌────────────────────────────────────┐
            │          Merge & Risk              │
            │  - unify OS + MND                  │
            │  - grid risk & daily aggregates    │
            └───────────┬──────────────┬────────┘
                        │              │
             latest enriched parquet   │
                        │              │
                        ▼              ▼
         data/enriched/incidents_*.parquet
         data/enriched/daily_grid_risk.csv
                        │
                        ▼
           data/history/ (partitioned by day)
                        │
                        ▼
                 Streamlit UI (local-only)
```

- **Sparse protection.** If `AUTO_RETRY_MAX=true` and the initial pull is
  sparse, the extractor performs one **MAX‑bbox** retry.
- **Pagination.** `MND_MAX_PAGES` controls the number of pages scraped;
  default behavior retrieves the first page only; set a larger value for
  deeper coverage.
- **LLM controls.** `MAX_MND_ENRICH=0` means “unlimited”; set a small
  number to cap LLM calls for smoke tests.

---

## Repository layout

```
grayzone-tw/
  app.py                     # Streamlit app (Monitor + History), read-only
  main.py                    # pipeline: extract → enrich → risk → write
  manage.py                  # CLI wrapper (run/history/ui/metrics/...)
  deepseek_enrichment.py     # LLM helpers and schema validation
  scripts/
    backfill.py              # build day partitions under data/history
    backtest.py              # scoring harness (optional)
    ui_smoke.py              # headless UI smoke test
    print_metrics.py         # parse METRICS line → JSON
  data/
    raw/                     # pulled OpenSky JSON / MND HTML (by run)
    enriched/                # latest parquet + daily_grid_risk.csv
    history/
      incidents_enriched/    # partitioned enriched parquets
      daily_grid_risk/       # partitioned risk CSV
  examples/                  # samples or watchlists (optional)
  logs/                      # local logs (git-ignored)
  requirements.txt
  README.md (this file)
```

> The code anchors its working directory to the repo root so relative
> paths like `data/` and `scripts/` resolve correctly no matter where you
> launch commands from (VS Code, terminals, CI).

---

## Requirements

- Python **3.10+** (tested on 3.12)
- Internet connectivity for extraction (OpenSky + MND)
- Optional DeepSeek API key for enrichment
- `pip install -r requirements.txt`

---

## Installation

### Windows (PowerShell)

```powershell
# 1) Clone and enter
git clone <your-repo-url>
cd grayzone-tw

# 2) Virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3) Dependencies
pip install -r requirements.txt


## Why we set env **once per session** (and how to do it reliably)

Python does **not** auto-load `.env` files, and VS Code’s
`"python.envFile": ".env"` only applies to **Debug/Test**, not to commands
you run in the terminal (e.g., `python manage.py ...`). To make the CLI
and Streamlit behave the same way every time, we load the environment
**into the current terminal session** before running anything.

**What this guarantees**

- The pipeline and the UI see the **same variables** (`os.environ`).
- Works from any shell (no IDE coupling), on Windows/macOS/Linux.
- Avoids common `.env` pitfalls (wrong encoding/BOM, smart quotes).


'''



```powershell
$env:OPENSKY_CLIENT_ID     = "your_client_id"
$env:OPENSKY_CLIENT_SECRET = "your_client_secret"
$env:OPENSKY_TOKEN_URL     = "https://auth.opensky-network.org/oauth/token"
$env:DEEPSEEK_API_KEY      = "sk-..."      # optional
$env:AUTO_RETRY_MAX        = "true"
$env:MAX_MND_ENRICH        = "0"
$env:MND_MAX_PAGES         = "5"
```

### macOS / Linux (bash)
### NOT 100% SURE IF THIS WORKS, I AM ON PC USE THE LOGIC OF THE INSTRUCTIOSN ABOVE
```bash
git clone <your-repo-url>
cd grayzone-tw

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cat > .env <<'EOF'
OPENSKY_CLIENT_ID=your_client_id
OPENSKY_CLIENT_SECRET=your_client_secret
OPENSKY_TOKEN_URL=https://auth.opensky-network.org/oauth/token

DEEPSEEK_API_KEY=sk-...    # optional (enrichment)
AUTO_RETRY_MAX=true
MAX_MND_ENRICH=0
MND_MAX_PAGES=5
EOF
```
###verify
```python - <<'PY'
import os
def m(v): return (v[:4]+'…') if v else None
print("OPENSKY_CLIENT_ID:", m(os.getenv("OPENSKY_CLIENT_ID")))
print("DEEPSEEK_API_KEY :", m(os.getenv("DEEPSEEK_API_KEY")))
print("MND_MAX_PAGES   :", os.getenv("MND_MAX_PAGES"))
print("MAX_MND_ENRICH  :", os.getenv("MAX_MND_ENRICH"))
print("AUTO_RETRY_MAX  :", os.getenv("AUTO_RETRY_MAX"))
PY

```

### `.env` file format

- One `KEY=VALUE` per line, no quotes needed; avoid trailing spaces.
- The app reads `.env` at process start (and environment variables
  always win if both are present).

---

## Runbook

### Quickstart (24h run + UI)

**Windows (PowerShell):**

```powershell
# from repo root; venv active
if (!(Test-Path .\logs)) { New-Item -ItemType Directory -Path .\logs | Out-Null }
python .\manage.py run --hours 24 > .\logs\mon24.log
Select-String -Path .\logs\mon24.log -Pattern '^METRICS \|' | Select-Object -Last 1
python .\scripts\print_metrics.py --log .\logs\mon24.log

# launch the dashboard
streamlit run app.py
# Local URL: http://localhost:8501
```

**macOS / Linux (bash):**

```bash
mkdir -p logs
python ./manage.py run --hours 24 > ./logs/mon24.log
grep '^METRICS \|' ./logs/mon24.log | tail -n 1
python ./scripts/print_metrics.py --log ./logs/mon24.log

streamlit run app.py
```

### Build N‑day history

```powershell
# build 7 days of history (skips existing outputs)
python .\manage.py history --days 7 --resume
```

### Common commands

```powershell
# Parse a log into JSON metrics
python .\scripts\print_metrics.py --log .\logs\mon24.log

# Backtest (if implemented)
python .\manage.py backtest --days 7

# Quick UI smoke
python .\scripts\ui_smoke.py
```

- `manage.py run --hours N` accepts `--bbox "latmin,lonmin,latmax,lonmax"`
  if you want to override the default focus window.

---

## Environment variables

| Key                   | Purpose                                                     | Example / Default            |
|----------------------|--------------------------------------------------------------|------------------------------|
| `OPENSKY_CLIENT_ID`  | OAuth client id (OpenSky)                                    | _(required for client/REST)_ |
| `OPENSKY_CLIENT_SECRET` | OAuth client secret (OpenSky)                              | _(required for client/REST)_ |
| `OPENSKY_TOKEN_URL`  | OAuth token endpoint                                         | `https://auth.opensky-network.org/oauth/token` |
| `DEEPSEEK_API_KEY`   | Optional key for enrichment                                  | `sk-...`                     |
| `AUTO_RETRY_MAX`     | `true` → one retry with **MAX** bbox if initial pull sparse  | `true`/`false`               |
| `MAX_MND_ENRICH`     | Max number of MND rows to enrich (`0` = unlimited)           | `0`                          |
| `MND_MAX_PAGES`      | Number of MND pages to scrape (pagination)                   | `1` or more                  |

---

## Artifacts and file conventions

- **Raw pulls** → `data/raw/`
  - `opensky_YYYYMMDDHH.json`
  - `mnd_YYYYMMDDHH.html`
- **Latest enriched parquet** → `data/enriched/incidents_enriched_*.parquet`
- **Daily risk CSV** → `data/enriched/daily_grid_risk.csv`
- **History partitions** → `data/history/...`
  - `incidents_enriched/YYYY‑MM‑DD.parquet`
  - `daily_grid_risk/YYYY‑MM‑DD.csv`
- **Logs** → `logs/*.log`

> The dashboard reads **from disk only**. If a layer is missing (e.g. no
> risk yet), it falls back to point rendering and shows a caption with
> the reason.

---

## Metrics & logs

Every pipeline run ends with a **single METRICS line** and a JSON echo:

```
METRICS | opensky_points=85 mnd_rows=70 merged_rows=155 enriched_rows=70 \
llm_success=68 llm_invalid_json=2 llm_retries=3 needs_review_count=4 \
validation_sparse_fallbacks=5 os_anom_rows=12 wall_ms=78313
```

Use `scripts/print_metrics.py` to parse a log:

```powershell
python .\scripts\print_metrics.py --log .\logs\mon24.log
```

- `opensky_points` — OS rows in the window
- `mnd_rows` — MND rows after pagination + dedupe
- `merged_rows` — union rows in the joined set
- `enriched_rows` — rows sent to DeepSeek
- `llm_success/invalid_json/retries` — enrichment quality telemetry
- `needs_review_count` — LLM flagged rows for review
- `validation_sparse_fallbacks` — number of safe defaults used
- `os_anom_rows` — rows with anomaly tags (if computed)
- `wall_ms` — wall‑clock time for the run

---

## Dashboard guide

- **Monitor tab**
  - Window presets (24h/48h/N days) and `Show OpenSky` / `Show MND` toggles
  - **Risk hex** if risk is available; otherwise point fallback
  - Fixed, sensible view state (CARTO dark). No token required.
  - Watchlist of star grids (if present) and a focus grid selector

- **History tab**
  - Independent slider for N‑day playback
  - Snapshot toggle for 24h slices
  - Trend chart (points/day, OS_ANOM if available)
  - Uses partitioned history under `data/history/`

> Counters in the header reflect what is actually plotted. If a control
> changes the underlying filter, the counts update in‑place.

---

## Verification checklist

1. **Environment**  
   `python -V` prints 3.10+; `pip -V` ok; venv active.
2. **Credentials**  
   `.env` exists or session variables set (OpenSky + optional DeepSeek).
3. **24h run**  
   `manage.py run --hours 24` completes; **METRICS** present in log.
4. **Artifacts exist**  
   `data/raw/`, `data/enriched/` files created.
5. **UI smoke**  
   `scripts/ui_smoke.py` runs without exceptions.
6. **Dashboard**  
   `streamlit run app.py` shows points/hex and the counters align.

---

## Troubleshooting

- **OpenSky returns 0 states**
  - Verify OAuth values and network; try again with `AUTO_RETRY_MAX=true`.
  - If still 0, the airspace may be quiet—continue; the map will still
    render MND markers.

- **MND rows seem capped at 10**
  - Increase `MND_MAX_PAGES` (e.g., `5`, `10`, `12`); the site often
    lists ~10 per page.

- **LLM warnings**
  - If `DEEPSEEK_API_KEY` is missing, the pipeline warns and uses safe
    defaults. Set a key to enable full enrichment.

- **Blank/white map**
  - If `pydeck` is missing or risk is empty, the app falls back to point
    rendering (and shows a caption).

- **Widget collisions**
  - If you see "DuplicateElementId", you may be on an older branch.
    Pull latest `feat/ui-clean-root` (or your feature branch) where
    unique keys are set site‑wide.

- **History slider not changing map**
  - Ensure your history partitions exist in `data/history/` and match
    the date range you're selecting. The header shows the effective
    window used by the map.

---

## Development notes

- **Coding standards**: PEP‑8, 79‑char lines, minimal imports.
- **No new dependencies** without discussion.
- **UI is read‑only**: the Streamlit app never fetches data from the
  internet; all I/O is local file reads.
- **pydeck optional**: if unavailable, we render via `st.map`.
- **Small diffs**: keep patches surgical and include a short PASS/FAIL
  matrix (imports, `ui_smoke`, metrics present, basic counts).

---

## Security, privacy, and attribution

- Use of OpenSky and MND data is subject to their respective terms.
- This project is for research and demonstration. No warranties.
- Credit: OpenSky Network, Taiwan MND, and the open‑source Python
  ecosystem.

---

## FAQ

**Q: Do I need a DeepSeek key?**  
A: No. The pipeline runs without it; enrichment is simply skipped or
defaulted. With a key you get structured MND fields and better tooltips.

**Q: Why does the map sometimes show only points, not hex/heat?**  
A: Hex aggregation requires a risk surface. If risk isn't available for
the window (or pydeck isn't installed), the UI uses a point fallback.

**Q: Where do I change the geographic focus?**  
A: Pass `--bbox "latmin,lonmin,latmax,lonmax"` to `manage.py run` or
adjust internal defaults in `main.py` (advanced users only).

**Q: Can I run this entirely offline?**  
A: The UI can, but extraction cannot. Run the pipeline first to create
local artifacts, then the UI can operate offline.
