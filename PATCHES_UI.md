# PATCHES_UI.md — Finishing Touches (MP-4.1 onward + Time Header + manage.py)
ASCII only • UTF-8 (no BOM) • Read by Codex for context

## Purpose
This playbook lists the last micro-patches to complete the UI and add a
simple wrapper CLI. It starts at MP-4.1 (skip dark mode), then adds a
small date-range header fix and a manage.py wrapper.

We keep public APIs unchanged. Streamlit UI is read-only (no network).
All edits must be PEP-8 with a 79-column limit. pydeck is optional.

---

## Guardrails (apply to every patch below)
- Files you may edit: `app.py` (UI). For the wrapper CLI, create
  `manage.py` at repo root.
- No new dependencies; do not edit `requirements.txt`.
- No network in the UI; read only from `data/enriched/` and
  `data/history/`.
- pydeck optional: try/except import; fallback to `st.map`.
- Whitespace rules (strict):
  - Exactly 2 blank lines between top-level defs/classes.
  - Exactly 1 blank line inside functions/methods between logical chunks.
  - Never emit 2+ consecutive blank lines elsewhere.
  - No trailing spaces; one newline at EOF.
  - After editing a file, collapse any excess blank lines before returning
    a diff.
- Be defensive: tolerate empty frames / missing columns.

---

## Remaining Work (ordered)
- MP-4.1 — Analyst KPI cards (with optional 24h deltas)  <- implement now
- ~~MP-4.2 — Optional dark theme (skip for now)~~
- MP-4.3 — Date-range header when hours >= 24  <- small polish
- TOOLS-1 — `manage.py` wrapper CLI  <- quality-of-life

---

## MP-4.1 — Analyst KPI cards (with optional 24h deltas)
Scope: `app.py` only

Goal
At the top of Analyst, render 4 KPI cards for the current window:
1) MND incidents count.
2) Mean risk (window).
3) Mean OS_ANOM (mean `validation_score` on MND rows).
4) Top actor (by frequency; empty if none).

Show deltas vs previous 24h if possible; otherwise show "—".

Implementation notes
- Add helper:
  ```python
  def compute_kpis(
      df_window: pd.DataFrame,
      df_prev: pd.DataFrame | None = None
  ) -> dict[str, str | float]:
      # Return dict with keys:
      # mnd_count, mean_risk, mean_os_anom, top_actor,
      # d_mnd, d_mean_risk, d_mean_os_anom
      ...
  ```
- df_window is the already filtered time window (respecting sidebar filters).
- Build df_prev by shifting the window back 24h using the same filters:
  - Compute end = df_window["dt"].max() (UTC).
  - Define prev_start = end - pd.Timedelta(hours=24) and prev_end = end.
  - Apply the same filter function to source/severity/category on the entire
    DataFrame, then cut prev_start <= dt < prev_end.
- Numeric coercion:
  `risk_score = pd.to_numeric(df["risk_score"], errors="coerce")`
  `validation_score = pd.to_numeric(df["validation_score"], errors="coerce")`
- Compute top actor by counting tokens across actors:
  accept strings or lists; defensively handle empty/missing.
- Render with st.columns(4); format numbers to 2 decimals; render deltas
  with a small "+/-" prefix or "—" when unavailable.
- Guard sparse or missing columns: if no data, show zeros/"—" and continue
  without exceptions.

Acceptance
- Cards render; values change with time slider and filters.
- No crashes when columns are missing or frames are empty.

Checks
```powershell
Select-String -Path .\app.py -Pattern "compute_kpis" -n
python .\scripts\ui_smoke.py
```

---

## MP-4.3 — Date-range header when hours >= 24
Scope: `app.py` only

Goal
Make long windows readable. If the selected hours >= 24, show an absolute
UTC range in addition to the hours count. Keep the short format for <24h.

Implementation notes
- In the header where you currently write "Window: last {hours}h …":
  - If hours < 24: keep existing text.
  - If hours >= 24: compute start = df["dt"].min() and end = df["dt"].max()
    for the current filtered window and render:
    "Window: {start:%Y-%m-%d %H:%MZ} -> {end:%Y-%m-%d %H:%MZ} (~{hours}h) | ..."
- Guard empty frames: if dt missing or NaT, fall back to the simple header.
- Do not mutate data; this is a display-only enhancement.

Acceptance
- For hours < 24, header is unchanged.
- For hours >= 24, header displays absolute UTC range plus approx hours.

Checks
```powershell
Select-String -Path .\app.py -Pattern "Window:" -n
python .\scripts\ui_smoke.py
```

---

## TOOLS-1 — manage.py wrapper CLI (quality-of-life)
Scope: new file `manage.py` at repo root

Goal
Wrap common commands into easy verbs:

```
python manage.py run --hours 6 [--bbox 118,20,123,26]
python manage.py history --days 14 --resume [--sleep 1.5]
python manage.py ui
python manage.py smoke
python manage.py metrics --log logs\demo.log
python manage.py simulate --runs 500 --seed 42
python manage.py backtest --days 10
```

Implementation notes
- Use argparse subcommands: run, history, ui, smoke, metrics, simulate,
  backtest.
- Call existing scripts with subprocess.run([...], check=False) and
  sys.executable:
  - main.py (run, simulate)
  - scripts/backfill.py (history)
  - scripts/ui_smoke.py (smoke)
  - scripts/print_metrics.py (metrics)
  - scripts/backtest.py (backtest)
  - app.py via streamlit run (ui); if streamlit missing, print a
    helpful message and exit 1.
- Windows-friendly: no shell=True; pass args as a list.
- Clear one-line success/failure per subcommand; exit with child’s code.
- Docstrings for the module and each subcommand function.
- ASCII only; PEP-8/79; 2 blank lines between top-level defs.

Acceptance
- python manage.py --help lists verbs.
- Each verb executes the corresponding command and returns the child’s code.

Checks
```powershell
python .\manage.py --help
python .\manage.py run --hours 1
python .\manage.py history --days 3 --resume
python .\manage.py smoke
```

---

## Final Verification (quick list)
```powershell
# Imports OK
python - <<'PY'
import importlib
for m in ('main','deepseek_enrichment','app'):
    importlib.import_module(m); print(m,'OK')
PY

# Short ETL + metrics
python .\main.py --hours 1 --out-prefix verify > .\logs\verify.log
python .\scripts\print_metrics.py --log .\logs\verify.log

# History for playback
python .\scripts\backfill.py --days 7 --resume

# Headless UI smoke
python .\scripts\ui_smoke.py

# (Optional) Wrapper CLI
python .\manage.py --help
```

---

## How Codex should use this file
1) Read this file, summarize which patch to do next.
2) Implement one patch per turn (minimal diffs).
3) Apply whitespace rules and collapse excess blank lines.
4) Run the listed Checks and report PASS/FAIL with one-line evidence.
5) Return a unified diff and a short commit message.
