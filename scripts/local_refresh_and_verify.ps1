Param(
  [ValidateSet("history","window")]
  [string]$Mode = "history",

  # History mode: number of days to backfill (default: 21 = 3 weeks)
  [int]$Days = 21,

  # Window mode: hours lookback (ignored in history mode)
  [int]$Hours = 504,

  # Whether to skip streamlit smoke at the end (default: false)
  [switch]$SkipUISmoke = $false,

  # Force AUTO_RETRY_MAX to help fill sparse windows
  [switch]$AutoRetryMax = $true,

  # Optional DeepSeek key override for this run (empty => no override)
  [string]$DeepseekKey = ""
)

$ErrorActionPreference = "Stop"

function Head($t){ Write-Host "=== $t ===" -ForegroundColor Cyan }

# 0) Canonical root guard
$root = (git rev-parse --show-toplevel).Trim()
Set-Location $root
Head "Repo root: $root"
"Branch: $(git rev-parse --abbrev-ref HEAD)"
"Latest commits:"
git log --oneline -n 3

# 1) Ensure python present (best effort)
try {
  $pyver = & python -c "import sys; print(sys.version)"
  "Python: $pyver"
} catch {
  Write-Error "python not on PATH. Activate your venv and re-run."
  exit 1
}

# 2) Session env wiring
if ($AutoRetryMax) { $env:AUTO_RETRY_MAX = "true" }
if ($DeepseekKey)  { $env:DEEPSEEK_API_KEY = $DeepseekKey }

# 3) Logs dir
$logs = Join-Path $root "logs"
if (-not (Test-Path $logs)) { New-Item -ItemType Directory -Path $logs | Out-Null }

# 4) Run selected mode
if ($Mode -eq "history") {
  Head "Backfill history ($Days days; --resume)"
  & python .\manage.py history --days $Days --resume `
    *> (Join-Path $logs "history_${Days}d.log")
} else {
  Head "Window run ($Hours hours)"
  & python .\manage.py run --hours $Hours `
    *> (Join-Path $logs "window_${Hours}h.log")
}

# 5) Metrics snapshot (parse the newest log if present)
$lastLog = Get-ChildItem $logs -File | Sort-Object LastWriteTime | Select-Object -Last 1
if ($null -ne $lastLog) {
  Head "Metrics from $($lastLog.Name)"
  & python .\scripts\print_metrics.py --log $lastLog.FullName
  "METRICS count: " + (Select-String -Path $lastLog.FullName -Pattern '^METRICS \|' | Measure-Object).Count
} else {
  Write-Warning "No log file found for metrics parse."
}

# 6) Enriched integrity (PowerShell -> Python via here-string)
Head "Enriched integrity"
@'
import pathlib as p, pandas as pd
enr = sorted(list(p.Path("data/enriched").glob("*.parquet")) +
             list(p.Path("data/enriched").glob("*.csv")))
if not enr:
    print("no enriched output found"); raise SystemExit(0)
f  = enr[-1]
df = pd.read_parquet(f) if f.suffix==".parquet" else pd.read_csv(f)
need = {"dt","source","grid_id","risk_score","corroborations"}
print("cols_ok:", need.issubset(df.columns), "rows:", len(df), "file:", f.name)
mnd = df[df["source"].eq("MND")]
cnt = int(mnd["corroborations"].astype(str).str.contains("OS_ANOM:").sum())
print("MND rows:", len(mnd), "OS_ANOM tagged:", cnt)
'@ | python -

# 7) Daily grid risk variance
Head "Daily grid risk variance"
@'
import pandas as pd, pathlib as p
path = p.Path("data/enriched")/"daily_grid_risk.csv"
if not path.exists():
    print("daily_grid_risk.csv missing")
else:
    dg = pd.read_csv(path)
    print("distinct risk rounded3:", dg["risk_score"].round(3).nunique())
'@ | python -

# 8) UI headless smoke (optional)
if (-not $SkipUISmoke) {
  Head "UI smoke"
  & python .\scripts\ui_smoke.py
}

Head "Done"
