# DeepSeek Usage and Enrichment Guide

This document explains exactly how the project uses **DeepSeek** to enrich
Taiwan gray‑zone incident bulletins (from Taiwan MND) with structured labels
and short summaries. It includes the prompt template, batch/validation
workflow, metrics mapping, tuning knobs, and a short runbook for operators.

---

## TL;DR

- The enrichment step adds **typed JSON fields** to each MND incident.
- We enforce **strict JSON output** with a small, stable schema.
- Calls are **batched (<=20)** with **retry + validation**; failures fall back
  to a conservative heuristic and are marked `needs_review=True`.
- Progress is visible in **METRICS**: `llm_success`, `llm_invalid_json`,
  `llm_retries`, and `needs_review_count`.
- To run: set `DEEPSEEK_API_KEY`, execute a pipeline command, and inspect the
  metrics line or the enriched parquet/csv artifacts.

---

## Where this lives in the pipeline

- Module: `deepseek_enrichment.py`
- Called by: extraction/merge routine in `main.py` after MND scraping.
- Output files:
  - `data/enriched/incidents_enriched_*.parquet` (most recent snapshot)
  - `data/history/incidents_enriched/*.parquet` (per‑day history, if backfill)
- Downstream consumers:
  - `app.py` Monitor/History tabs (tables, watch cells, map tooltips, charts)
  - Daily risk aggregation (`data/enriched/daily_grid_risk.csv`)

---

## Environment and knobs

- `DEEPSEEK_API_KEY` (required) — API key used for DeepSeek calls.
- `MAX_MND_ENRICH` (int, default **0** meaning unlimited) — hard cap on how
  many MND incidents are sent to DeepSeek in a single run. Useful to keep
  costs/pacing bounded during historical backfills or demos.

> The pipeline still runs without an API key. In that mode each incident is
> assigned a conservative heuristic severity/risk and flagged for review.

---

## Prompt: schema‑first JSON

We prompt DeepSeek to **only** return a single JSON object with the following
keys and contracts. The constraints are repeated both in the system prompt and
the client `response_format={"type": "json_object"}`.

### System prompt (exact)

```
You are an analyst supporting a Taiwan gray-zone monitor. Return STRICT JSON
with keys exactly: category, actors, weapon_class, severity_0_5, risk_score,
summary_one_line, where_guess, geo_quality. category is one of
[Piracy, UAV/Drone, Missile, Boarding, Seizure, Mine, Electronic
Interference, Other, null]. actors is a list of strings. weapon_class is str
or null. severity_0_5 is integer 0-5. risk_score is float 0-1.
summary_one_line is short string. where_guess is str or null. geo_quality is
high, medium, low, or null. No commentary.
```

### Call parameters

- `model="deepseek-chat"`
- `temperature=0.2` (pushes toward deterministic labels)
- `response_format={"type": "json_object"}` (server‑side JSON guard)

### Incident context injected

Each prompt includes:
- UTC timestamp of the incident
- Source tag (`MND`)
- Country tag (`TW`)
- Grid identifier (`R###C###`) used by the dashboard
- The original bulletin / headline text

> Rationale: adding timestamp/source/grid improves disambiguation without
> leaking any implementation details; we still get stable, human‑readable
> summaries that match UI filters.

---

## What the model must produce

The enrichment step appends the following **validated** columns to each MND
incident row:

- `category`: one of the listed enum values or `null`
- `actors`: list of strings (empty list is allowed)
- `weapon_class`: string or `null`
- `severity_0_5`: integer in `[0, 5]`
- `risk_score`: float in `[0.0, 1.0]`
- `summary_one_line`: short string; used directly in tables/tooltips
- `where_guess`: string or `null` (e.g., “Taiwan Strait, near Taichung”)
- `geo_quality`: `high` | `medium` | `low` | `null`
- `needs_review`: bool; `True` if parsing/validation failed
- `_raw_response`: raw JSON from the model (for traceability & audits)

### Normalization / coercion rules

- `actors` is coerced to a list (single strings wrapped as `[str]`).
- Non‑numerics in `severity_0_5` → best‑effort int; then clamp to `[0,5]`.
- `risk_score` → float; clamp to `[0.0,1.0]`.
- Empty strings for optional fields become `None`.
- On any irrecoverable parse error: set `needs_review=True` and populate a
  conservative default (`severity_0_5=1`, `risk_score≈0.2`).

---

## Batching, retries, and pacing

- **Batch size**: up to **20** incidents per API call (`_call_deepseek_batch`).
- **Retries**: wrapped in `tenacity` (max **2** attempts) when the HTTP
  response is missing `choices` or the JSON payload fails schema validation.
- **Throughput**: the small batch keeps UI‑latency reasonable and helps avoid
  rate spikes during backfills.

> We intentionally avoid parallelism to keep logs and per‑run metrics simple.
> If needed, horizontal scaling can be added later via process‑level sharding.

---

## Error handling and fallbacks (important)

Failure modes handled:
- Network / transport errors
- Rate limiting or timeouts
- Model returns non‑JSON text
- JSON is present but invalid by the schema rules above

Fallback path:
1. Mark the incident `needs_review=True`.
2. Apply a small keyword heuristic for a minimal label (e.g., set
   `category="Other"`, `severity_0_5=1`, `risk_score=0.2`).
3. Store the raw text we did receive (if any) in `_raw_response` for audits.
4. Continue the batch; **never** crash the pipeline on a single bad row.

---

## Metrics emitted (how to read success)

A single METRICS line is appended to the run log (see `logs/*.log`). DeepSeek
touches the following keys:

- `llm_success`: number of incidents successfully enriched and validated
- `llm_invalid_json`: LLM responses that failed JSON parsing/validation
- `llm_retries`: how many second‑attempt calls were made
- `needs_review_count`: incidents flagged for manual review after fallbacks

Other keys you will see (produced elsewhere in the pipeline):

- `mnd_rows`, `opensky_points`, `merged_rows`, `enriched_rows`
- `validation_sparse_fallbacks`, `os_anom_rows`, `wall_ms`

### Quick check

```powershell
# After running a pipeline command (see Runbook below)
Select-String -Path .\logs\*.log -Pattern '^METRICS \|' | Select-Object -Last 1

# Pretty-print the JSON payload portion
python scripts\print_metrics.py --log .\logs\LAST_RUN.log
```

---

## What works best (prompting strategies)

- **Strict JSON** + `response_format=json_object` + low `temperature`. This
  combination yields stable, schema‑conformant outputs with minimal post‑fixes.
- Keep the **schema small** and typed. Avoid long enumerations and nested
  objects; they increase invalid JSON risk without adding much value to the UI.
- Provide **minimal context** (timestamp/source/grid); more prose tends to
  degrade determinism and invite non‑JSON commentary.
- Normalize strictly and **clamp** numeric fields to valid ranges. Treat any
  non‑conformant output as “needs review” rather than guessing.

---

## Known challenges and our solutions

| Challenge | Symptom | Mitigation |
|---|---|---|
| Non‑JSON output despite JSON mode | “choices[0].message.content” is text | Retry once; if still invalid, mark `needs_review` and continue |
| Partially valid JSON (wrong types) | e.g., `actors` as string | Coerce types; if still invalid after coercion, mark `needs_review` |
| Overly verbose summaries | UI overflow / unreadable tooltips | Prompt asks for **short** `summary_one_line`; trim in UI if needed |
| Backfill spikes | API errors, long runs | Use `MAX_MND_ENRICH` to cap per‑run volume |
| Key not set | “DEEPSEEK_API_KEY is not configured.” in logs | Set env var (see Runbook) or run without enrichment for demos |

---

## Operator runbook

### 1) Configure the key (session‑scoped, reliable on Windows/PowerShell)

```powershell
# From repo root, with venv active
$env:DEEPSEEK_API_KEY = "sk-PASTE_YOUR_REAL_KEY"
```

*(If you prefer a `.env`, see README’s Environment section. For grading/demos,
the session variable above is the fastest and least error‑prone.)*

### 2) Run a small window and verify metrics

```powershell
# Fresh 24h snapshot (writes one enriched parquet + daily risk csv)
python .\manage.py run --hours 24 > .\logs\run_24h.log

# Show the last METRICS line
Select-String -Path .\logs\run_24h.log -Pattern '^METRICS \|' | Select-Object -Last 1
```

### 3) Backfill N days (optional)

```powershell
# Resume-friendly backfill; throttled in the script
python .\manage.py history --days 7 --resume > .\logs\backfill7d.log
```

### 4) Inspect artifacts in the app

```powershell
streamlit run app.py
```

- Monitor tab uses the **latest snapshot** in `data/enriched/`.
- History tab reads per‑day files from `data/history/incidents_enriched/` and
  matches them with `data/history/daily_grid_risk/` if present.

---

## Repro and tests (quick sanity)

```powershell
# Module syntax
python -m py_compile deepseek_enrichment.py

# UI smoke (map serialization; does not call the API)
python scripts\ui_smoke.py
```

---

## Extending the schema safely

1. Add the new key to the system prompt with a clear type (keep it simple).
2. Update the validator in `deepseek_enrichment.py` to coerce/clamp the type.
3. Add a column in the merge step that defaults to `None` on failure.
4. Surface the field in the UI tables only after the enrichment is stable.

> Keep additions rare and small. Stability in JSON structure is essential for
> consistent historical backfills and easy grading.

---

## Appendix: Example request and response

> Example values are synthetic; structure matches the real flow.

**Prompt suffix (dynamic data):**

```text
UTC: 2025-09-21T00:00:00Z
source: MND
country: TW
grid_id: R220C601

bulletin:
PLA activity reported in sea and air around Taiwan Strait.
```

**Expected model response (strict JSON):**

```json
{
  "category": "Other",
  "actors": ["People's Liberation Army"],
  "weapon_class": null,
  "severity_0_5": 1,
  "risk_score": 0.2,
  "summary_one_line": "PLA activity in sea and air near the Taiwan Strait.",
  "where_guess": "Taiwan Strait",
  "geo_quality": "low"
}
```

If the response is missing or invalid, the pipeline records:

```json
{
  "needs_review": true
}
```

…and continues without crashing.

---

*Last updated: 2025-09-22*
