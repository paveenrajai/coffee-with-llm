"""A provider's own reason for stopping, in the words every provider shares."""

from __future__ import annotations

from typing import Mapping, Optional

from ..types import Stop, StopReason


def stop_from(raw: object, reasons: Mapping[str, str]) -> Optional[Stop]:
    """``raw`` as a :class:`Stop`. ``None`` when the provider gave no reason.

    ``raw`` may be an enum (Gemini's ``FinishReason``) or a plain string. A
    word ``reasons`` does not know is kept as :attr:`StopReason.OTHER` with
    the word itself in ``raw``, never dropped: a provider adding a reason
    must not make a stop look like a finish.
    """
    if raw is None:
        return None
    word = str(getattr(raw, "name", None) or raw).strip()
    if not word:
        return None
    return Stop(reason=reasons.get(word, StopReason.OTHER), raw=word)
