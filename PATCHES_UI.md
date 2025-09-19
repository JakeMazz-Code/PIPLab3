# UI Patch Playbook — Micro‑Patches (Post Phase‑1)
ASCII only • UTF‑8 (no BOM) • For Codex + Humans

## Purpose
This file tells **Codex** exactly what to do next (small, safe patches) and
tells **you** how to run checks after each step. It assumes Phase‑1 (core UX
fixes) already landed.

Keep baseline public APIs intact. **Edit `app.py` only** unless a patch
explicitly adds a tiny script. No network calls in Streamlit.

---

## Where to place this file
Put this file at the **repo root** as `PATCHES_UI.md`.
Codex will be instructed to read it for context.

Recommended layout (abridged):
```
grayzone-tw/
  app.py
  main.py
  deepseek_enrichment.py
  scripts/
    ui_smoke.py
    verify_all.py
  data/
  examples/
  PATCHES_UI.md        <-- this file
```

---

## How to direct Codex to use it
Start a **new Codex chat** and paste this bootstrap:

```
ROLE
You are a senior Python engineer. Minimal PEP‑8 diffs (79 cols). Touch only
files I name. No network in Streamlit. pydeck optional via try/except.

CONTEXT
Read ./PATCHES_UI.md and summarize the next micro‑patch to implement.
Confirm file presence and repo status in ≤20 lines, then implement only the
next patch. After each patch, run the provided checks and report PASS/FAIL.

APPROVED FILES
- app.py (primary)
- .streamlit/config.toml (optional theme; create if missing)
- scripts/ui_smoke.py (already present; do not modify unless instructed)
```

Then say (for each step):  
**“Implement: MP‑2.3 (sidebar filters). Use PATCHES_UI.md as the source of
truth.”**

---

## Status tracker (fill as you go)
- [x] Phase‑1 (core UX fixes)
- [ ] MP‑2.3 — Sidebar filters (source, severity, category)
- [ ] MP‑2.4 — Focus grid selectbox → map highlight
- [ ] MP‑3.1 — History summary strip (points/day + OS_ANOM mean/day)
- [ ] MP‑3.2 — Cached loaders (st.cache_data, keyed by mtime)
- [ ] MP‑3.3 — Timelapse controls polish (lookback slider + snapshot)
- [ ] MP‑4.1 — KPI cards (deltas vs previous 24h)
- [ ] MP‑4.2 — Optional dark theme (.streamlit/config.toml)

---

## Guardrails (do not violate)
- **No baseline API changes** (functions in main.py / deepseek_enrichment.py).
- **No new dependencies**; Streamlit, matplotlib, pandas only.
- **Read‑only**: app reads from `data/enriched/` and `data/history/` only.
- **PEP‑8/79**; docstrings for new helpers; 2 blank lines between top‑level defs.
- **pydeck optional**: if missing, fallback to `st.map` + captions.
- Be **defensive**: tolerate empty frames / missing columns.

---

## MP‑2.3 — Sidebar filters (source, severity, category)
**Scope**: `app.py` only.

**Goal**
Sidebar filters that affect **map, watch cells, anomaly chart, and table**:
- Sources: `[✓ MND] [✓ OpenSky]` (default both on).
- Severity range: `0..5` (inclusive).
- Category multiselect: use unique values present; may be empty.

**Implementation notes**
- Build a helper `_apply_filters(df, sources, sev_min, sev_max, categories)`.
- Ensure `risk_score`/`severity_0_5` numeric via `pd.to_numeric(..., coerce)`.
- If `category` missing, treat as empty and ignore category filter.
- Cache the filtered window in a local var so each panel reuses it.

**Acceptance**
- Toggling filters changes header counts and table rows.
- Map & anomaly chart reflect the filtered set.
- Never crashes if columns are missing (show info).

**Commands**
```powershell
# static grep proof of filters wired
Select-String -Path .\app.py -Pattern "Severity|Category|source" -n
python .\scripts\ui_smoke.py
```

---

## MP‑2.4 — Table→map focus (selected grid)
**Scope**: `app.py` only. Works with MP‑2.2 overlay.

**Goal**
Add a **“Focus grid”** selectbox above the MND table listing unique
`grid_id` in the current window. On change: set
`st.session_state['selected_grid'] = grid_id`.

**Behavior**
- If pydeck present, overlay highlights selection (MP‑2.2 scatter layer).
- Else, show a small caption: `Focused grid: <id>`.
- If no MND rows, disable/selectbox hidden.

**Acceptance**
- Picking a grid highlights it on the map (or shows caption).

**Commands**
```powershell
Select-String -Path .\app.py -Pattern "Focus grid" -n
python .\scripts\ui_smoke.py
```

---

## MP‑3.1 — History summary strip
**Scope**: `app.py` only.

**Goal**
Above the daily render in **History**, show a small summary chart:
- **Points/day** (reuse Arcade scoring; create a tiny `_row_points(row)`
  and `compute_points_per_day(df)` helper).
- **OS_ANOM mean/day** (use `validation_score` on MND rows).

**Implementation notes**
- Use matplotlib (single axes is fine); annotate days on x‑axis.
- If data too sparse, show `st.info`.

**Acceptance**
- With 7–14 days in `data/history/`, chart renders without errors.

**Commands**
```powershell
Select-String -Path .\app.py -Pattern "Playback summary" -n
python .\scripts\ui_smoke.py
```

---

## MP‑3.2 — Cached loaders
**Scope**: `app.py` only.

**Goal**
Use `@st.cache_data(ttl=60)` to speed up artifact loads:
- Latest enriched parquet loader.
- Per‑day incident parquet + risk CSV loader.

**Keying**
- Include file **mtime** in the cache key (or pass a tuple of mtimes).

**Acceptance**
- Re‑loading the same day in one session uses the cache (no errors).

**Commands**
```powershell
Select-String -Path .\app.py -Pattern "st\.cache_data" -n
python .\scripts\ui_smoke.py
```

---

## MP‑3.3 — Timelapse controls polish
**Scope**: `app.py` only.

**Goal**
- **Lookback slider**: `3..min(30, available_days)`, default 7.
- Button **Play timelapse** iterates selected days using `st.empty()`; no sleep.
- Checkbox **Show 24h snapshot** (default True) appends a snapshot of the
  latest enriched data after playback.

**Acceptance**
- With history present, playback iterates days and optionally shows snapshot.

**Commands**
```powershell
Select-String -Path .\app.py -Pattern "Play timelapse" -n
python .\scripts\ui_smoke.py
```

---

## MP‑4.1 — KPI cards
**Scope**: `app.py` only.

**Goal**
In **Analyst**, show four KPI cards:
- MND incident count
- Mean risk_score
- Mean OS_ANOM (mean `validation_score` on MND)
- Top actor (by frequency; empty if none)

Show small deltas vs previous 24h if window has enough data.

**Implementation notes**
- Helper `compute_kpis(df_window, df_prev=None) -> dict`.
- Use `st.columns(4)` to render; if delta unavailable, show `—`.

**Acceptance**
- Cards render; values change with slider/filters; no crashes if sparse.

**Commands**
```powershell
Select-String -Path .\app.py -Pattern "compute_kpis" -n
python .\scripts\ui_smoke.py
```

---

## MP‑4.2 — Optional dark theme
**Scope**: create file if missing.

**Goal**
Add `.streamlit/config.toml` with a subtle dark theme. Do not overwrite an
existing file.

**Content (suggestion)**
```
[theme]
base="dark"
primaryColor="#6ea8fe"
backgroundColor="#0f1116"
secondaryBackgroundColor="#161a23"
textColor="#e6e6e6"
```
**Commands**
```powershell
if (Test-Path .\.streamlit\config.toml) { Get-Content .\.streamlit\config.toml }
```

---

## Final sweeps — quick commands
```powershell
# (Optional) Fresh artifacts for nicer UI
python .\main.py --hours 2 --out-prefix ui > .\logs\ui.log

# History (7–14 days for playback)
python .\scripts\backfill.py --days 14 --resume

# Headless checks
python .\scripts\ui_smoke.py

# Launch UI
streamlit run app.py
```

---

## What Codex should ask you (if unclear)
- Do you want filters applied to **both** MND and OpenSky panels, or MND only?
  (Default: both; anomaly chart uses MND only.)
- If the **category** column is missing frequently, should the UI hide that
  control? (Default: hide if missing.)
- For **KPI deltas**, compare to the **previous 24h** or a **same‑day baseline**?
  (Default: previous 24h.)
- If no pydeck, is it OK to show a note “Install `pydeck` for advanced map
  layers”? (Default: yes, as a caption.)

---

## Verification one‑liners (PowerShell)
```powershell
python .\scripts\ui_smoke.py
Select-String -Path .\app.py -Pattern "Focus grid|st\.cache_data|Play timelapse|compute_kpis" -n
```

---

## Done criteria
- Each micro‑patch produces a small diff to `app.py` only (theme file aside).
- `ui_smoke.py` passes after each patch.
- Streamlit renders (with or without pydeck) and controls affect the data.
- No new dependencies; no network in the UI.
