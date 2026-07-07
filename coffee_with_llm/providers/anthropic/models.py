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

# Generation-5 families and short aliases (sonnet-5, claude-sonnet-5, …).
_GEN5_MODEL_RE = re.compile(
    r"(?:claude-)?(?:sonnet|fable|mythos)-5(?:$|[-_])",
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
    """Omitting ``thinking`` leaves adaptive thinking enabled on the API."""
    key = _model_key(model)
    if not key:
        return False
    if anthropic_thinking_always_on(model):
        return True
    if "mythos-preview" in key:
        return True
    if _GEN5_MODEL_RE.search(key) and "sonnet" in key:
        return True
    return False


def anthropic_thinking_can_disable(model: str) -> bool:
    """Whether ``thinking: {type: disabled}`` is supported."""
    if anthropic_thinking_always_on(model):
        return False
    if "mythos-preview" in _model_key(model):
        return False
    return True


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
