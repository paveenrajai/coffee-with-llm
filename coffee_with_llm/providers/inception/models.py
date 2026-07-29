"""Inception / Mercury model id helpers."""

from __future__ import annotations

# Short aliases → API model ids (after provider prefix strip).
_ALIASES: dict[str, str] = {
    "mercury": "mercury-2",
    "mercury-2": "mercury-2",
}


def normalize_inception_model_id(model: str) -> str:
    """Return the API model id for an Inception model string."""
    m = (model or "").strip()
    if not m:
        return m
    return _ALIASES.get(m.lower(), m)
