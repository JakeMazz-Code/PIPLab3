
# AI Usage — Prompts & Outcomes (Build Log)

This document captures the **prompts that drove the largest changes**, why
they mattered, and **exact, reproducible outcomes** in this repository.
Treat it as a **recipe** to reproduce, extend, or audit the project.

Project: **Gray‑Zone Air & Sea Monitor (Taiwan)**  
Repo root examples referenced here: `app.py`, `main.py`, `scripts/*`, `data/*`

> **Scope of this doc**
> - What we asked Codex/ChatGPT to do (prompt patterns, acceptance rules).
> - What changed in the codebase and why it was necessary.
> - How the Agents (see `AGENTS.md`) orchestrate multi‑step fixes.
> - How to verify with deterministic commands and a PASS/FAIL rubric.
> - Boundaries: what AI must **not** change without review.

---

## 0) North‑Star Acceptance (what “done” looks like)

**Prompt (operator intent)**  
> “I need a dashboard that, for a chosen window (24h, 48h, N‑days), shows
> risk‑weighted heat/hex over Taiwan, raw OpenSky points, and MND incidents.
> The UI must never crash, counts must match the window, and a metrics line
> must flush at the end of each run.”

**Why it mattered**  
This set the guardrails for every patch: no crashes, consistent filters,
auditable metrics, and a map that **always renders something** even when
weights are missing or data are sparse.

**Acceptance (quick)**

- A metric banner appears (no UnicodeDecodeError), e.g.:  
  `METRICS | opensky_points=85 mnd_rows=70 enriched_rows=70 ...`
- Map renders with no tiles? → still shows **points** (or `st.map`) fallback.
- “Last 24h/48h/N‑days” changes counts, tables, and layers **in sync**.
- Watch cells and incident lists match visible points for the same window.
- No Streamlit duplicate‑ID widget errors.

---

## 1) OpenSky client + OAuth REST fallback

**Prompt**  
> “Prefer the official Python OpenSky client when creds exist; otherwise use
> the OAuth REST (token) path. Fix bbox order for the client, handle
> `states=None`, write the raw JSON regardless, and never crash. Update
> `opensky_points` in METRICS.”

**Why it mattered**  
Anonymous REST can be sparse/unreliable. An environment‑driven client path
keeps the extractor robust without introducing hard runtime dependencies.

**Implementation outcomes**

- `main.py` optional import: `OpenSkyApi` with try/except.
- Unified `_fetch_opensky_states()` returns `(df, meta)`; meta notes path.
- **Correct bbox order** for client: `(min_lat, max_lat, min_lon, max_lon)`.
- **Sparse retry** obeys `AUTO_RETRY_MAX` (retry once with MAX bbox).  
- Atomic writes for raw JSON kept, even with 0 states.
- `opensky_points` added to METRICS from the filtered dataframe length.

**Verification**

```powershell
# No‑creds run (still succeeds)
python .\manage.py run --hours 6 > .\logs\nocreds6h.log
Select-String .\logs\nocreds6h.log '^METRICS \|' | Select-Object -Last 1
```

Files touched: `main.py` (extract), minor log/counter plumbing.

---

## 2) “10 incidents” root cause — MND pagination

**Prompt**  
> “Audit scraping for hidden caps. If we only pull the first bulletin page,
> add env‑gated pagination, dedupe by `(dt, title)`, and log scrape stats.
> Enrichment caps remain controlled by `MAX_MND_ENRICH`.”

**Why it mattered**  
We repeatedly saw “Enriching 10 incidents ...” because we never paged past
the **first page** of the MND bulletin site. This made the UI look flat and
misleading even when more rows existed.

**Implementation outcomes**

- `MND_MAX_PAGES` environment var (default `1`).
- Scraper logs: `pages=.. raw_cards=.. parsed_rows=.. after_dedupe=..`.
- No `.head(10)` or slices; enrichment limiter is **only** `MAX_MND_ENRICH`.
- Example run: `mnd_rows=70` with `MND_MAX_PAGES=7`.

**Verification**

```powershell
$env:MND_MAX_PAGES="7"; $env:MAX_MND_ENRICH="0"
python .\manage.py run --hours 168 > .\logs\paged7.log
Select-String .\logs\paged7.log 'MND scrape:' | Select-Object -Last 1
```

Files touched: `main.py` (scrape, dedupe, logging).

---

## 3) Streamlit stability — multiselect, selectbox keys, map fallback

**Prompt**  
> “Fix the multiselect crash when defaults include stale grid_ids; clear
> `selected_grid` if not in options; add unique `key=` for repeated widgets;
> guarantee map renders (pydeck → points → `st.map` fallback); set a fixed
> `ViewState` and height.”

**Why it mattered**  
Widget ID collisions and stale watchlists broke the UX. Fallback logic
ensured we never show a blank map if tiles or weights fail.

**Implementation outcomes**

- Guard defaults: `default = [g for g in saved if g in options]`.
- Reset invalid `st.session_state['selected_grid']` to `None`.
- For pydeck off/empty, autoswitch to points; otherwise `st.map`.
- Fixed `pdk.ViewState(latitude=23.5, longitude=121.0, zoom=5)`, height ~520.

Files touched: `app.py` (Arcade/Analyst → later superseded by Monitor).

---

## 4) Monitor tab — single, explicit map control

**Prompt**  
> “Replace confusing tabs with a **Monitor** page: _layer mode_
> (Points/Heatmap/Hexagons) and independent toggles for OpenSky & MND. Keep
> a stable dark map; don’t flip themes on toggle; counts above the map must
> match the filtered data. History uses the same slicer.”

**Why it mattered**  
It made the UX predictable, the counts honest, and the map controllable by a
few obvious switches.

**Implementation outcomes**

- Unified map builder: stable dark base + **layer mode** radio.
- Small dots for points (no giant translucent circles).
- Risk heat/hex use weights; auto‑fallback to points if weights missing.
- Header text reflects **actual** filtered window and row counts.
- Small explainer caption: what OS/MND/risk mean for this window.

Files touched: `app.py` (monitor plumbing, map helper).

---

## 5) Window slicing — one source of truth

**Prompt**  
> “Create a single `_window_bounds()` helper: compute start/end/label for
> 24h / 48h / N‑days. Clamp starts if snapshot < requested window; surface a
> ‘clamped’ hint. Ensure **all** frames (OS, MND, risk, watch cells, tables)
> slice with the same bounds.”

**Why it mattered**  
Fixed the “August rows in 24h view” confusion and kept cards/tables in sync
with the map.

**Implementation outcomes**

- `_window_bounds(now, mode, days)` returns `(start, end, label, clamped)`.
- All view dataframes slice with `(start <= dt < end)`.
- Header shows: `rows: 54 (OS: 53 / MND: 1)` for the exact filter.

Files touched: `app.py`.

---

## 6) UTF‑16 logs (PowerShell `>` redirection)

**Prompt**  
> “PowerShell writes UTF‑16 by default; make `_latest_metrics_line()` detect
> encoding (BOM or fallback attempts) and never crash. If nothing matches,
> return `None` and render a friendly banner.”

**Why it mattered**  
Streamlit crashed on `UnicodeDecodeError` when reading `.log` files created
with `>` redirection.

**Implementation outcomes**

- Open log **in binary**, detect BOM, decode with `utf-8`, `utf-8-sig`,
  `utf-16`, `utf-16le`, or `cp1252` fallbacks.
- Return last line matching `^METRICS \|` or `None`.

Files touched: `app.py`.

---

## 7) Caching & recompute hygiene

**Prompt**  
> “Add `st.cache_data(ttl≈60)` for file loaders keyed by `Path(...).stat()`
> mtime so charts don’t recompute on every rerender.”

**Why it mattered**  
Improved responsiveness while still reacting quickly to new files.

Files touched: `app.py` (loaders).

---

## 8) Environment & Secrets (operational notes)

- **OpenSky OAuth**  
  `OPENSKY_CLIENT_ID`, `OPENSKY_CLIENT_SECRET`, `OPENSKY_TOKEN_URL`
- **DeepSeek** (optional)  
  `DEEPSEEK_API_KEY`
- **Extractor behavior**  
  `AUTO_RETRY_MAX` (true/false), `MND_MAX_PAGES` (default 1)  
  `MAX_MND_ENRICH` (0 = unlimited)
- Use either `.env` (UTF‑8) or shell variables; **avoid quotes** around values
  in PowerShell **unless** the value contains spaces—then prefer **no trailing
  spaces** and straight quotes. The code tolerates both paths.

---

## Re‑usable Master Prompts (copy/paste)

### A) OpenSky client + REST fallback

> **Goal**: Prefer the official client with correct bbox `(min_lat, max_lat,
> min_lon, max_lon)`; otherwise OAuth REST. Write raw JSON even with 0. Retry
> MAX bbox once if sparse. Update `opensky_points` in METRICS. PEP‑8 diffs,
> no public‑API changes, `main.py` only.

**Accept when**: run shows `OpenSky (client|rest)` log, raw JSON file exists,
METRICS contains `opensky_points=N`.

---

### B) MND pagination and dedupe

> **Goal**: Add `MND_MAX_PAGES` (default 1). Scrape up to N pages, parse,
> **dedupe by (dt, title)**, and log `pages/raw/parsed/after_dedupe`. Keep
> enrichment limiter only in `MAX_MND_ENRICH`. No hidden `[:10]` slices.

**Accept when**: logs show `MND scrape: pages=.. raw=.. after_dedupe=..` and
`mnd_rows` grows with pages.

---

### C) Monitor map modes (Points/Heatmap/Hexagons)

> **Goal**: Radio layer mode with stable dark basemap; independent toggles for
> OS/MND; small crisp dots; heat/hex over risk weights; auto‑fallback to points
> if weights missing; header counts align with data.

**Accept when**: Theme never flips; counts match tables; toggles don’t reset
the view; caption explains layers.

---

### D) UTF‑16 log reader

> **Goal**: Robust `_latest_metrics_line()` that opens logs in binary, detects
> BOM, and decodes with utf‑8/utf‑8‑sig/utf‑16/utf‑16le/cp1252. Return the last
> `METRICS |` line or `None`. Never throw in UI.

**Accept when**: Running `python manage.py run --hours 24 > logs/x.log` then
opening Streamlit never crashes and shows the metrics banner or “unavailable”.

---

## Agents & Roles (see `AGENTS.md`)

- **Operator Agent**: orchestrates high‑level goals (e.g., “stabilize UI”),
  picks the next sub‑task, and asks Codex for diffs + a PASS/FAIL matrix.
- **Extractor Agent**: focuses on `main.py` (OpenSky/MND) with environment
  gates, retry policy, logging, and metrics flush.
- **UI Agent**: focuses on `app.py` widgets, fallbacks, cache, and layered
  maps. Adds unique keys, clamps sliders, and manages ViewState.
- **Doc Agent**: generates/editable `.md` drafts and PR summaries.

Each agent patch includes: a small diff, a **runnable test block**, and a
short PASS/FAIL table appended to the PR comment.

---

## What was AI‑generated vs human‑edited

**AI (Codex/ChatGPT) produced**  
- Draft diffs for OpenSky fallback, pagination, Streamlit keys, map layers,
  window slicing, and UTF‑16 log reader.  
- Test harness commands and PR checklists.  
- Documentation drafts (this file, `README.md`, `DEEPSEEK_USAGE.md`).

**Human review**  
- Credential strategy, bbox defaults, and Taiwan‑centric ViewState.  
- Safety gates around enrichment and retries.  
- Visual polish and wording in the UI.  
- Final smoke runs and decisions on acceptable defaults (e.g., page depth).

---

## Quick Verification Suite

```powershell
# Ensure venv active; set envs as needed
$env:AUTO_RETRY_MAX="true"
$env:MND_MAX_PAGES="5"
$env:MAX_MND_ENRICH="0"

# 1) Short run with metrics
python .\manage.py run --hours 24 > .\logs\utf16_test.log
Select-String .\logs\utf16_test.log '^METRICS \|' | Select-Object -Last 1

# 2) UI smoke
python .\scripts\ui_smoke.py

# 3) Snapshot sanity (latest parquet + risk CSV)
@'
import pathlib as p, pandas as pd
enr=sorted(p.Path("data/enriched").glob("*.parquet"))
print("parquet_count", len(enr))
if enr:
    df=pd.read_parquet(enr[-1])
    mnd=df[df["source"].eq("MND")]
    os_=df[df["source"].eq("OS")]
    print("latest", enr[-1].name, "rows", len(df), "MND", len(mnd), "OS", len(os_))
f=p.Path("data/enriched")/"daily_grid_risk.csv"
print("risk_csv", f.exists())
'@ | python -

# 4) Streamlit (check Monitor & History)
streamlit run app.py
```

PASS if:
- Metrics banner appears; no UnicodeDecodeError.  
- Map layer modes work; toggles don’t flip the theme; counts align.  
- History uses the same slicer and renders with no exceptions.

---

## Prompt Hygiene & Red Flags

- **Be explicit about acceptance** (commands + expected console snippets).  
- **Name files you will touch** and forbid others (“`app.py` only”).  
- **Ban scope creep**: “No new deps. No public API changes.”  
- **Guard UI widgets**: always provide stable `key=` when the same widget
  appears in multiple contexts.  
- **Watch for hidden caps**: search `.head(`, `[:N]`, `sample(` near enrich.  
- **Always surface env gates**: `MND_MAX_PAGES`, `MAX_MND_ENRICH` etc.  
- **Log what you do**: “MND scrape: pages=.. raw=.. after_dedupe=..”.

---

## Appendix — One‑liners you can paste in PRs

- “**UTF‑16 safe** metrics reader implemented; banner renders even if logs are
  redirected by PowerShell.”
- “**Map modes** now explicit (Points/Heatmap/Hexagons). Theme is stable. Dots
  are small; toggles don’t change basemap.”
- “**Window slicing** unified; tables and map use the same bounds; header
  counts derive from the same filtered frames.”
- “**MND pagination** enabled via `MND_MAX_PAGES` (default 1). Dedupe by
  `(dt, title)`. Logs include scrape stats for auditability.”

---

*Last updated: keep this file in sync with PRs that touch extraction,
slicing, metrics, or the map rendering path.*
