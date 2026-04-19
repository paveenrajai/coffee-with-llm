"""Provider-agnostic reasoning_effort translation.

Public API exposes a single ``reasoning_effort`` string ("low" | "medium" |
"high"). Each provider maps it to its own extended-thinking configuration so
callers stay free of provider details.

Mapping (token budget for thinking):
    low    -> 1024
    medium -> 4096
    high   -> 16384

Unknown / None values yield ``None`` so callers can simply skip the param.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_BUDGETS = {
    "low": 1024,
    "medium": 4096,
    "high": 16384,
}


def normalize_effort(effort: Optional[str]) -> Optional[str]:
    """Lower-case, trim, and validate ``effort``. Returns None if absent/invalid."""
    if effort is None:
        return None
    key = str(effort).strip().lower()
    if not key:
        return None
    if key not in _BUDGETS:
        logger.warning(
            "Unknown reasoning_effort=%r; expected one of %s. Ignoring.",
            effort,
            sorted(_BUDGETS),
        )
        return None
    return key


def thinking_budget_tokens(effort: Optional[str]) -> Optional[int]:
    """Return the per-call thinking token budget for a normalized effort, or None."""
    key = normalize_effort(effort)
    if key is None:
        return None
    return _BUDGETS[key]
