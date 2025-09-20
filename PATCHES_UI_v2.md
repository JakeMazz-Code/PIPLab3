# PATCHES_UI_v2.md
**Gray‑Zone TW — UI & Playback micro‑patches (v2)**  
Encoding: UTF‑8 (no BOM). Line endings: LF.

This file is a **reference playbook** for small, safe improvements to the
Streamlit dashboard and history playback. Each patch is self‑contained:
it lists the target files, edits, acceptance criteria, and quick tests.
Follow them **in order** unless the repo already contains the feature.

> Baseline constraints (do not change):  
> • No changes to public APIs in `main.py` / `deepseek_enrichment.py`.  
> • Streamlit reads from local artifacts only (no network).  
> • `pydeck` is optional (`try/except`).  
> • PEP‑8, max line length 79, two blank lines between top‑level defs.  
> • Use ASCII only in UI strings.


---

## 4.1 — KPI Cards (Analyst tab)

**Problem**  
Analysts need a quick, FlightAware‑style snapshot: incident count, mean
risk, mean OS_ANOM, top actor—with simple deltas vs. previous 24h.

**File**: `app.py`  

**Edits**  
1) Add a helper (or re‑use if present):
```python
def compute_kpis(df: pd.DataFrame, prev_df: pd.DataFrame | None) -> dict:
    # returns keys: mnd_count, mean_risk, mean_os_anom, top_actor,
    # and deltas: d_mnd, d_mean_risk, d_mean_os_anom
```
2) In Analyst, compute `window_df` once, `prev_24h_df` if available, then
   render four stat blocks with small green/red delta chips.

**Acceptance**  
- Four KPI blocks render.  
- Deltas display when the previous 24h window exists (else “—”).

**Quick tests**
```powershell
Select-String -Path .\app.py -Pattern "def compute_kpis" -n
python - <<'PY'
import pathlib as p, pandas as pd, app
files = sorted(p.Path('data/enriched').glob('*.parquet')); assert files
df = pd.read_parquet(files[-1]); df['dt']=pd.to_datetime(df['dt'],utc=True)
k = app.compute_kpis(df, None) if hasattr(app,'compute_kpis') else {}
print('kpis_keys:', sorted(k.keys()))
PY
```


---

## 4.3 — Date‑Range Header for long windows

**Problem**  
When `hours >= 24` the “last N hours” text is hard to parse. Show
absolute UTC start/end **and** keep the hours for context.

**File**: `app.py`

**Edits**  
- After `window_df` is computed, derive `start_utc` and `end_utc`, render:  
  `Window: {start} > {end} (~{hours}h) | rows: {len} (MND: {n_mnd})`.

**Acceptance**  
- Header appears on Analyst, Arcade, and History panes.  
- Counts change when the slider changes.

**Quick tests**
```powershell
Select-String -Path .\app.py -Pattern "Window:" -n
```


---

## H1 — History Slider Bounds (crash fix)

**Problem**  
`st.slider(min=3, max=available_days)` crashes when history < 3 days.

**File**: `app.py`

**Edits**  
- In `render_history_tab`:  
  ```python
  available = max(available_days, 0)
  if available == 0:
      st.info("No history partitions yet."); return
  lookback_min = 1
  lookback_max = min(30, max(available, 1))
  default = min(max(7, lookback_min), lookback_max)
  lookback = st.slider("Playback range (days)",
                       min_value=lookback_min,
                       max_value=lookback_max,
                       value=default)
  ```

**Acceptance**  
- No Streamlit slider exception when history < 3 days.  
- When there’s no history, the tab shows an info message.

**Quick tests**
```powershell
Select-String -Path .\app.py -Pattern "lookback_min|lookback_max" -n
```


---

## M1 — Map Provider + Safe Fallback (white map fix)

**Problem**  
pydeck draws layers but the base map is white because no tile provider
was set (defaults may require a Mapbox token).

**File**: `app.py`

**Edits**  
- When creating the deck, set:  
  `Deck(map_provider="carto", map_style="dark", layers=[...], initial_view_state=...)`  
- Wrap `st.pydeck_chart(deck)` with `try/except Exception`. On error, fall
  back to `st.map(centroids_df.rename(columns={"lat":"latitude","lon":"longitude"}))`
  and show `st.info("Map fell back to Streamlit base map")`.

**Acceptance**  
- Basemap shows “CARTO / OSM” credits and renders.  
- If pydeck errors, Streamlit fallback renders points.

**Quick tests**
```powershell
Select-String -Path .\app.py -Pattern "map_provider\\s*=\\s*['\\\"]carto" -n
```


---

## M2 — Layer Guards & Visibility (tiny datasets)

**Problem**  
Heatmap looks blank with small or zero weights. Points should be visible.

**File**: `app.py`

**Edits**  
- In heatmap prep: coerce `risk_score` numeric; drop NaNs; if all zeros,
  set `risk_score = 0.05` (epsilon).  
- If fewer than 5 rows after cleaning, draw `ScatterplotLayer` instead of
  `HeatmapLayer`.  
- Clip invalid lat/lon to bounds and drop if out of range.

**Acceptance**  
- Small datasets render as points; larger datasets render as heatmap/hex.

**Quick tests**
```powershell
Select-String -Path .\app.py -Pattern "ScatterplotLayer|risk_score.*=\\s*0\\.05" -n
```


---

## P1 — Caching & Performance

**Problem**  
UI re‑reads parquet/CSV and recomputes windows repeatedly.

**File**: `app.py`

**Edits**  
- Add `@st.cache_data(ttl=60)` to: latest parquet loader, history partition
  readers, and any deterministic helpers (e.g., grid centroid).  
- Compute the filtered `window_df` once per tab and pass it to render
  helpers (avoid recomputation).

**Acceptance**  
- Noticeable reduction in UI lag and CPU spikes.

**Quick tests**
```powershell
Select-String -Path .\app.py -Pattern "st\\.cache_data" -n
```


---

## U1 — Remove unintended 10‑row caps

**Problem**  
Some tables are artificially capped with `.head(10)`. Keep Top‑K only
for “watch cells”; do not cap MND incidents.

**File**: `app.py`

**Edits**  
- Remove `.head(10)` on the MND table.  
- Optional: set `MAX_TABLE_ROWS = 1000`; if over, display the first 1000
  plus a caption “table truncated for display”.

**Acceptance**  
- MND table shows all rows in window (or a high cap), not just 10.

**Quick tests**
```powershell
Select-String -Path .\app.py -Pattern "\\.head\\(10\\)" -n
```


---

## G1 — Wrapper CLI (`manage.py`) — *idempotent*

**Problem**  
Running separate commands is verbose; provide a single entry‑point without
changing baseline scripts.

**File**: `manage.py` (create or keep if present)

**Verbs**  
- `run --hours N [--bbox ...]` → calls pipeline  
- `history --days D [--resume]` → calls backfill  
- `ui` → launches Streamlit  
- `smoke` → runs `scripts/ui_smoke.py`  
- `metrics --log path` → pretty‑prints METRICS JSON  
- `simulate --runs R --seed S` → calls simulate mode  
- `backtest --days D` → calls backtest

**Acceptance**  
- Each verb exits 0 on success; prints a one‑line success message.

**Quick tests**
```powershell
python .\manage.py smoke
python .\manage.py run --hours 1
python .\manage.py history --days 3 --resume
```


---

## How to use this file with Codex

1) Open a new chat and paste the **Audit/Context prompt** below.  
2) Paste the patch prompt for the **next patch only** (4.1 → 4.3 → H1 → M1 → M2 → P1 → U1 → G1).  
3) After each patch, run the “Quick tests” from this file.  
4) Only proceed when the tests pass.

### Audit/Context (rehydrate) prompt
```
ROLE
You are a senior Python engineer. Audit the repo to regain context in ≤30 lines.

GOAL
Confirm the dashboard files and the PATCHES_UI_v2.md exist and that app.py
contains KPI + date header + history + map code paths.

TASKS
- Print: git rev-parse --show-toplevel
- Print last 6 commits: git log --oneline -n 6
- List presence of: app.py, manage.py, scripts/ui_smoke.py, data/history/
- Grep app.py for: compute_kpis, "Window:", map_provider, ScatterplotLayer,
  st.cache_data, ".head(10)"
- If any item is missing, list it and STOP (no edits). Otherwise say
  “Audit OK; ready for patch N”.
```

### Per‑patch Codex prompt template
```
You are a senior Python engineer. Produce minimal, PEP‑8 diffs (79 cols).
Do not change public function signatures. Touch ONLY the named file(s).
Streamlit UI is read‑only; pydeck is optional via try/except. ASCII only.

OBJECTIVE
Implement patch <ID> from PATCHES_UI_v2.md.

FILES TO EDIT
<file list>

EDITS
<summarised edits copied from this doc for the chosen patch>

ACCEPTANCE
<acceptance block>

OUTPUT
- Unified diff(s) only.
- One-line commit message.
```

---

## Full quick verification (after all patches)

```powershell
# Imports
python - <<'PY'
import importlib
for m in ('main','deepseek_enrichment','app'):
    importlib.import_module(m); print(m,'OK')
PY

# Pipeline + metrics
python .\main.py --hours 1 --out-prefix ui > .\logs\ui.log
python .\scripts\print_metrics.py --log .\logs\ui.log

# Backfill 7 days (for history)
python .\scripts\backfill.py --days 7 --resume

# UI smoke
python .\scripts\ui_smoke.py

# (Optional) Run manage.py shortcuts
python .\manage.py run --hours 1
python .\manage.py history --days 3 --resume
python .\manage.py ui
```