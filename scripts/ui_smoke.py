"""Tiny UI smoke check for map serialization."""

from __future__ import annotations

from app import MAP_LAT, MAP_LON, MAP_ZOOM, _as_pdk_layers

sanitized = _as_pdk_layers([[], None, (1, 2), "x"])
assert not sanitized and all(
    cls.__name__.endswith("Layer") for cls in map(type, sanitized)
)

try:
    import pydeck as pdk  # type: ignore
except Exception:
    print("pydeck unavailable; skipping ui smoke.")
else:
    layers = _as_pdk_layers([])
    view_state = pdk.ViewState(
        latitude=MAP_LAT,
        longitude=MAP_LON,
        zoom=MAP_ZOOM,
    )
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_provider="carto",
        map_style="dark",
    )
    deck.to_json()
    print("ui_smoke map OK")
