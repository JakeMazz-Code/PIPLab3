# DeepSeek Usage Notes

## System prompt

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

The request payload uses `model="deepseek-chat"`, `temperature=0.2`, and
`response_format={"type": "json_object"}` to force JSON mode.

## Batched requests

- `_call_deepseek_batch` processes incidents in chunks of at most 20 prompts.
- Each prompt includes UTC timestamp, source, country tag, grid identifier,
  and the original bulletin text.
- Retries: the HTTP call is wrapped with `tenacity` (max two attempts). A
  second attempt is triggered if the response is missing `choices` or fails
  JSON validation.

## Error handling and fallbacks

- On persistent network or parsing failures the code flags the incident with
  `needs_review=True`, supplies a conservative severity/risk estimate derived
  from simple keyword heuristics, and records an empty `_raw_response`.
- All successful responses are type-checked (`actors` coerced to a list,
  numeric fields bounded) before merging back into the incident frame.

## Response columns

The enrichment appends the following fields to each MND incident:

- `category`
- `actors`
- `weapon_class`
- `severity_0_5`
- `risk_score`
- `summary_one_line`
- `where_guess`
- `geo_quality`
- `needs_review`
- `_raw_response` (stored for traceability)

## Rate limits and monitoring

No explicit rate limit calls are issued, but the 20-item batching imitates a
simple client-side budget. All DeepSeek interactions are logged via Python's
`logging` module at the WARNING level when failures occur.
