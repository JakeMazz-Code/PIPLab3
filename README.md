# Gray-Zone TW Monitor

This project implements a minimal, PEP-8 compliant ETL workflow for Taiwan air and sea gray-zone monitoring. The pipeline ingests ADS-B telemetry from the OpenSky Network, scrapes the Taiwan Ministry of National Defense (MND) daily PLA activity bulletins, enriches incidents with DeepSeek, and produces risk-oriented analytics and simulations.

## Data sources

- **OpenSky Network REST API** (`/states/all`): current ADS-B states filtered to a Taiwan bounding box. Supports optional HTTP basic auth via the `OPENSKY_USER` and `OPENSKY_PASS` environment variables.
- **Taiwan MND daily PLA activity bulletins**: HTML table at `https://www.mnd.gov.tw/PublishTable.aspx?Types=%E5%8D%B3%E6%99%82%E8%BB%8D%E4%BA%8B%E5%8B%95%E6%85%8B&title=%E5%9C%8B%E9%98%B2%E6%B6%88%E6%81%AF`. The scraper captures the table rows (date + summary) and preserves the raw HTML for auditing.
- **DeepSeek API**: provides structured enrichment, categorisation, and risk scoring for MND incidents. Configure `DEEPSEEK_API_KEY` before running the pipeline.

## Installation

Create and activate a Python 3.11+ environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the full ETL + enrichment pipeline (default 24 h window):

```bash
python main.py --hours 24 --out-prefix daily
```

The pipeline uses fixed 0.5 degree bounding boxes:

- CORE: 118,20,123,26
- WIDE: 115,18,126,28 (default when --bbox is omitted)
- MAX: 112,16,128,30 (one retry when AUTO_RETRY_MAX is enabled)

This command will:

- Save raw OpenSky JSON under `data/raw/`.
- Save the scraped MND HTML page under `data/raw/`.
- Produce an enriched dataset (Parquet with CSV fallback) under `data/enriched/`.
- Compute `data/enriched/daily_grid_risk.csv` with mean grid risk per day.
- Generate example artifacts in `examples/` (before/after enrichment and a 24-hour textual brief).

Regenerate examples from the latest enriched file without fetching new data:

```bash
python main.py --mode artifacts
```

Run the stochastic air-ops simulation (reads the newest enriched file):

```bash
python main.py --mode simulate --runs 500 --seed 42
```

The simulation always writes `data/enriched/simulation_runs.csv`.

## Environment variables

- `DEEPSEEK_API_KEY` (required for enrichment).
- `OPENSKY_USER`, `OPENSKY_PASS` (optional basic auth for OpenSky).
- `AUTO_RETRY_MAX` (optional; set to true/1/y to retry once with the
  MAX bbox when fewer than 50 OpenSky points are returned).
- `GRAYZONE_LOG_LEVEL` (optional Python logging level, e.g. `DEBUG`).

## Metrics line

Each run prints a single `METRICS |` line with these fields in order:
opensky_points, mnd_rows, merged_rows, enriched_rows, llm_success,
llm_invalid_json, llm_retries, needs_review_count,
validation_sparse_fallbacks, os_anom_rows, wall_ms.
Example: `METRICS | opensky_points=90 mnd_rows=8 merged_rows=120 enriched_rows=24 llm_success=24 llm_invalid_json=0 llm_retries=1 needs_review_count=3 validation_sparse_fallbacks=2 os_anom_rows=5 wall_ms=2345`.

## Output overview

- `data/raw/opensky_*.json` - untouched OpenSky responses.
- `data/raw/mnd_*.html` - raw MND bulletin table HTML.
- `data/enriched/*_enriched_*.parquet` - combined and enriched incidents.
- `data/enriched/daily_grid_risk.csv` - grid-level mean risk per day.
- `examples/mnd_before_after_001.md` - LLM before/after illustration.
- `examples/airops_brief_24h.md` - 24-hour DeepSeek theatre summary.
- `data/enriched/simulation_runs.csv` - optional simulation outputs.

## Notes and limitations

- The MND site returns a paginated ASP.NET table. The scraper captures the published summary rows and preserves the raw HTML, but detailed counts are not exposed without additional postback automation. The DeepSeek prompt is therefore fed the bulletin summary text and metadata gathered from the table.
- If the OpenSky response is empty or the network is unavailable, the code falls back to empty DataFrames and continues so the pipeline remains importable without network access.
- Parquet serialisation requires `pyarrow`. If unavailable, the code writes a CSV file instead and logs a warning.

## Repository layout

```
grayzone-tw/
|-- main.py
|-- deepseek_enrichment.py
|-- data/
|   |-- raw/
|   `-- enriched/
|-- examples/
|-- requirements.txt
|-- README.md
|-- DEEPSEEK_USAGE.md
|-- AI_USAGE.md
`-- .gitignore
```
