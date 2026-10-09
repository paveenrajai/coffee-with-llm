"""The model a provider says served a call."""

from __future__ import annotations

from typing import Optional


def served_model(value: object) -> Optional[str]:
    """``value`` as a model name, or ``None`` when the provider did not say.

    Only a non-empty string counts: a provider that leaves the field out must
    not have something else priced in its place.
    """
    if not isinstance(value, str):
        return None
    name = value.strip()
    if name.startswith("models/"):
        name = name[len("models/") :]
    return name or None
