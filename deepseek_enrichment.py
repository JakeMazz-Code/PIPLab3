"""DeepSeek enrichment utilities for gray-zone monitoring."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Iterable, Iterator

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential


logger = logging.getLogger(__name__)


DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
MAX_BATCH_SIZE = 20

SYSTEM_PROMPT = (
    "You are an analyst supporting a Taiwan gray-zone monitor. "
    "Return STRICT JSON with keys exactly: "
    "category, actors, weapon_class, severity_0_5, risk_score, "
    "summary_one_line, where_guess, geo_quality. "
    "category is one of "
    "[Piracy, UAV/Drone, Missile, Boarding, Seizure, Mine, "
    "Electronic Interference, Other, null]. "
    "actors is a list of strings. weapon_class is str or null. "
    "severity_0_5 is integer 0-5. risk_score is float 0-1. "
    "summary_one_line is short string. where_guess is str or null. "
    "geo_quality is high, medium, low, or null. No commentary."
)


class DeepSeekError(Exception):
    """Raised when DeepSeek enrichment fails irrecoverably."""


def _chunked(iterable: Iterable[Any], size: int) -> Iterator[list[Any]]:
    """Yield successive chunks from *iterable* of length *size*."""
    chunk: list[Any] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _deepseek_headers() -> dict[str, str]:
    """Return HTTP headers for DeepSeek requests."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise DeepSeekError("DEEPSEEK_API_KEY is not configured.")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _strip_json_block(text: str) -> str:
    """Strip code fences around JSON if present."""
    clean = text.strip()
    if clean.startswith("```"):
        parts = clean.split("```")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part.startswith("json"):
                continue
            clean = part
            break
    return clean


def _safe_parse_json(text: str) -> dict[str, Any]:
    """Parse JSON content, raising ValueError on failure."""
    clean = _strip_json_block(text)
    try:
        return json.loads(clean)
    except json.JSONDecodeError as exc:
        raise ValueError("DeepSeek returned invalid JSON") from exc


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=8),
    stop=stop_after_attempt(2),
    reraise=True,
)
def _call_deepseek_once(text: str) -> tuple[dict[str, Any], str]:
    """Call DeepSeek for a single prompt and return parsed payload."""
    payload = {
        "model": "deepseek-chat",
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    }
    response = requests.post(
        DEEPSEEK_URL,
        headers=_deepseek_headers(),
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    try:
        message = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise DeepSeekError("DeepSeek payload missing choices") from exc
    parsed = _safe_parse_json(message)
    return parsed, message


def _fallback_payload(text: str) -> dict[str, Any]:
    """Generate a conservative fallback structure from raw text."""
    lowered = text.lower()
    severity = 1
    hints: list[tuple[str, int]] = [
        ("missile", 4),
        ("ballistic", 4),
        ("uav", 3),
        ("drone", 3),
        ("ship", 2),
        ("interference", 2),
        ("incursion", 3),
        ("exercise", 1),
    ]
    for keyword, value in hints:
        if keyword in lowered:
            severity = max(severity, value)
    risk = min(1.0, max(0.05, severity / 5.0))
    if severity >= 4:
        guess = "north median line"
    elif severity >= 3:
        guess = "east median line"
    elif severity >= 2:
        guess = "south median line"
    else:
        guess = "median line"
    return {
        "category": None,
        "actors": [],
        "weapon_class": None,
        "severity_0_5": severity,
        "risk_score": risk,
        "summary_one_line": "Review required: JSON parse failure.",
        "where_guess": guess,
        "geo_quality": None,
        "needs_review": True,
    }


def _ensure_types(payload: dict[str, Any]) -> dict[str, Any]:
    """Coerce DeepSeek payload types to safe defaults."""
    payload = dict(payload)
    payload.setdefault("category", None)
    actors = payload.get("actors")
    if not isinstance(actors, list):
        actors = []
    payload["actors"] = [str(actor) for actor in actors]
    payload.setdefault("weapon_class", None)
    severity = payload.get("severity_0_5")
    if not isinstance(severity, int):
        try:
            severity = int(severity)
        except (TypeError, ValueError):
            severity = 0
    payload["severity_0_5"] = max(0, min(5, severity))
    risk = payload.get("risk_score")
    try:
        risk_value = float(risk)
    except (TypeError, ValueError):
        risk_value = 0.0
    payload["risk_score"] = max(0.0, min(1.0, risk_value))
    payload.setdefault("summary_one_line", "")
    payload.setdefault("where_guess", None)
    quality = payload.get("geo_quality")
    if quality is not None:
        quality = str(quality).lower()
        if quality not in {"high", "medium", "low"}:
            quality = None
    payload["geo_quality"] = quality
    payload.setdefault("needs_review", False)
    return payload


def _call_deepseek_batch(texts: list[str]) -> list[dict[str, Any]]:
    """Call DeepSeek for each text, handling retries and fallbacks."""
    results: list[dict[str, Any]] = []
    for chunk in _chunked(texts, MAX_BATCH_SIZE):
        for text in chunk:
            try:
                parsed, raw = _call_deepseek_once(text)
            except (
                DeepSeekError,
                requests.RequestException,
                ValueError,
            ) as exc:
                logger.warning(
                    "DeepSeek call failed, using fallback: %s",
                    exc,
                )
                fallback = _fallback_payload(text)
                fallback["_raw_response"] = ""
                results.append(fallback)
                continue
            payload = _ensure_types(parsed)
            payload["_raw_response"] = raw
            results.append(payload)
    return results


def _render_prompt(row: pd.Series) -> str:
    """Render a prompt for the incident row."""
    fields = [
        f"Timestamp (UTC): {row.get('dt')}",
        f"Source: {row.get('source')}",
        f"Country: {row.get('country')}",
        f"Location: lat={row.get('lat')} lon={row.get('lon')}",
        f"Grid: {row.get('grid_id')}",
        f"Raw text: {row.get('raw_text', '')}",
    ]
    return "\n".join(str(field) for field in fields)


def enrich_incidents(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich incidents using DeepSeek, preserving original columns."""
    if df.empty:
        return df.copy()
    prompts = [_render_prompt(row) for _, row in df.iterrows()]
    responses = _call_deepseek_batch(prompts)
    enriched = df.copy()
    new_columns = [
        "category",
        "actors",
        "weapon_class",
        "severity_0_5",
        "risk_score",
        "summary_one_line",
        "where_guess",
        "geo_quality",
        "needs_review",
        "_raw_response",
    ]
    for column in new_columns:
        if column not in enriched.columns:
            enriched[column] = None
    for idx, row_index in enumerate(enriched.index):
        if idx < len(responses):
            response = responses[idx]
        else:
            response = _fallback_payload("Missing response")
            response["_raw_response"] = ""
        for key in new_columns:
            if key not in response:
                continue
            value = response[key]
            if key == "where_guess" and value in (None, "", "median line"):
                continue
            enriched.at[row_index, key] = value
    return enriched


def summarize_theater(df: pd.DataFrame, horizon: str = "24h") -> str:
    """Create a concise plain-text summary for the given horizon."""
    if df.empty:
        return (
            "No incidents available for the requested horizon; "
            "monitoring continues."
        )
    scoped = df.copy()
    scoped["dt"] = pd.to_datetime(scoped["dt"], errors="coerce", utc=True)
    scoped = scoped.dropna(subset=["dt"])
    if scoped.empty:
        return (
            "Incident timestamps missing or invalid for summarization. "
            "Analysts should review latest feeds."
        )
    scoped = scoped.sort_values("dt")
    latest = scoped["dt"].max()
    window_start = latest - pd.Timedelta(horizon)
    recent = scoped[scoped["dt"] >= window_start]
    if recent.empty:
        recent = scoped
    total = len(recent)
    if "risk_score" in recent.columns:
        risk_series = pd.to_numeric(
            recent["risk_score"],
            errors="coerce",
        )
    else:
        risk_series = pd.Series(dtype=float)
    if not risk_series.empty:
        high_count = int((risk_series >= 0.7).sum())
        avg_risk = float(risk_series.mean())
    else:
        high_count = 0
        avg_risk = float("nan")
    if "category" in recent.columns:
        categories = (
            recent["category"].dropna().astype(str).value_counts().head(3)
        )
        top_categories = ", ".join(
            f"{name} ({count})" for name, count in categories.items()
        )
    else:
        top_categories = ""
    avg_risk_str = f"{avg_risk:.2f}" if not pd.isna(avg_risk) else "n/a"
    high_txt = f", {high_count} high-risk" if high_count else ""
    category_txt = (
        f"; leading categories: {top_categories}" if top_categories else ""
    )
    return (
        "Taiwan gray-zone snapshot for the last "
        f"{horizon}: {total} incidents{high_txt} observed. "
        f"Mean risk score {avg_risk_str}{category_txt}."
    )
