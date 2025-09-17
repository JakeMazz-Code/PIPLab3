# AI Usage Log

- **Authoring agent:** OpenAI ChatGPT (Codex CLI) acting as a senior Python
  engineer per assignment brief.
- **Generated files:** main.py, deepseek_enrichment.py, documentation, and
  support files were produced programmatically during this session. Manual
  adjustments were made to ensure ASCII-only source code and to align with
  repository conventions.
- **LLM prompts:** DeepSeek is only invoked at runtime via the helper module;
  no downstream LLMs were used while developing code in this repository
  outside of the mandated enrichment path.
- **Validation:** Static reasoning plus limited in-session inspection (no
  remote execution of external tests beyond those performed locally via the
  CLI instructions).
- **Bug review:** Issues encountered included encoding corrections (ensuring
  UTF-8/ASCII compliance) and verifying fallback logic for unavailable
  network resources; addressed directly in code.
