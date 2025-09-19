# PR Checklist

Please confirm the following before requesting review:

- [ ] Patch scope is a single micro-patch from `PATCHES_UI_v2.md`
- [ ] Public API signatures unchanged (baseline)
- [ ] PEP-8, 79-col lines; ASCII-only strings
- [ ] Streamlit UI stays read-only; no network calls
- [ ] pydeck usage is wrapped in try/except (optional dependency)

## What this PR implements

- Patch ID:
- Files touched:

## Local checks

Paste one-line outputs (or `SKIP` if not applicable):

- T01 imports: `main OK`, `deepseek_enrichment OK`, `app OK`
- T07 ui_smoke: `ui_smoke passed`
- Notes:
