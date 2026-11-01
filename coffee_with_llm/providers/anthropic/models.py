"""Anthropic model capability detection for Messages API requests.

Capability rules follow the Claude Platform docs (adaptive thinking, effort,
sampling restrictions). See:
https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking
"""

from __future__ import annotations

import re
from typing import Final

# Opus/Sonnet 4.x minor version (e.g. claude-opus-4-8 → 8).
_OPUS_SONNET_4_MINOR_RE = re.compile(r"(?:opus|sonnet)-4-(\d+)", re.IGNORECASE)

# Generation-5 families and short aliases (sonnet-5, claude-opus-5-5, …).
_GEN5_MODEL_RE = re.compile(
    r"(?:claude-)?(?:sonnet|fable|mythos|opus|haiku)-5(?:$|[-_])",
    re.IGNORECASE,
)

# Short model ids callers may pass after a provider prefix (anthropic/sonnet-5).
_MODEL_ALIASES: Final[dict[str, str]] = {
    "sonnet-5": "claude-sonnet-5",
    "fable-5": "claude-fable-5",
    "mythos-5": "claude-mythos-5",
}

# ``max_tokens`` floor when adaptive thinking is active (thinking + answer share budget).
ADAPTIVE_MIN_MAX_TOKENS: Final[dict[str, int]] = {
    "low": 8192,
    "medium": 12_000,
    "high": 16_000,
}


def normalize_anthropic_model_id(model: str) -> str:
    """Map known short aliases to canonical Claude API model ids."""
    m = (model or "").strip()
    if not m:
        return m
    return _MODEL_ALIASES.get(m.lower(), m)


def _model_key(model: str) -> str:
    return normalize_anthropic_model_id(model).lower()


def _opus_sonnet_4_minor(model: str) -> int | None:
    match = _OPUS_SONNET_4_MINOR_RE.search(_model_key(model))
    if match is None:
        return None
    return int(match.group(1))


def anthropic_supports_adaptive_thinking(model: str) -> bool:
    """Return True when the model accepts ``thinking: {type: adaptive}``."""
    key = _model_key(model)
    if not key:
        return False
    if _GEN5_MODEL_RE.search(key):
        return True
    if "mythos" in key:
        return True
    minor = _opus_sonnet_4_minor(model)
    return minor is not None and minor >= 6


def anthropic_thinking_always_on(model: str) -> bool:
    """Adaptive thinking cannot be turned off (Fable 5, Mythos 5)."""
    key = _model_key(model)
    return "fable-5" in key or "mythos-5" in key


def anthropic_thinking_defaults_on(model: str) -> bool:
    """Omitting ``thinking`` leaves adaptive thinking enabled on the API.

    Every generation-5 model (Opus 5 and 5.5, Sonnet 5 and 5.5, Haiku 5.5,
    Fable, Mythos) and Mythos Preview; Opus 4.8 and 4.7 run without it.
    """
    key = _model_key(model)
    if not key:
        return False
    if anthropic_thinking_always_on(model):
        return True
    if "mythos-preview" in key:
        return True
    return bool(_GEN5_MODEL_RE.search(key))


def anthropic_thinking_can_disable(model: str) -> bool:
    """Whether ``thinking: {type: disabled}`` is supported.

    Not on Fable 5, Mythos and Mythos Preview, whose thinking is always on;
    not on Opus 5.5, where effort is the only control; and not on Sonnet 5.5,
    which turns thinking off with ``between_tools`` instead.
    """
    if anthropic_thinking_always_on(model):
        return False
    key = _model_key(model)
    return not any(name in key for name in ("mythos-preview", "opus-5-5", "sonnet-5-5"))


def anthropic_thinking_off(model: str) -> dict[str, str] | None:
    """The ``thinking`` value that turns thinking off, or None where it cannot be."""
    if "sonnet-5-5" in _model_key(model):
        return {"type": "between_tools"}
    if anthropic_thinking_can_disable(model):
        return {"type": "disabled"}
    return None


def anthropic_rejects_sampling_params(model: str) -> bool:
    """Non-default temperature/top_p/top_k return 400 — omit them entirely."""
    key = _model_key(model)
    if not key:
        return False
    if _GEN5_MODEL_RE.search(key):
        return True
    if "mythos-preview" in key:
        return True
    minor = _opus_sonnet_4_minor(model)
    return minor is not None and minor >= 7


def adaptive_min_max_tokens(effort: str) -> int:
    """Recommended minimum ``max_tokens`` when adaptive thinking is active."""
    return ADAPTIVE_MIN_MAX_TOKENS.get(effort, 16_000)
