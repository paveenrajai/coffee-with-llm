"""Shared tool execution utilities for all providers."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Max consecutive tool steps with only reasoning_log (no real tools) before breaking.
# Prevents infinite loops when model keeps "thinking" without calling tools.
MAX_CONSECUTIVE_REASONING_ONLY = 3


def normalize_tool_result(result: Any) -> Dict[str, Any]:
    """Normalize tool execution result to standard format."""
    try:
        if hasattr(result, "ok"):
            return {
                "ok": bool(getattr(result, "ok", False)),
                "result": getattr(result, "result", {}),
                "error": getattr(result, "error", None),
            }
        if isinstance(result, dict):
            return {
                "ok": bool(result.get("ok", False)),
                "result": result.get("result", {}),
                "error": result.get("error", None),
            }
    except Exception as e:
        logger.warning(f"Failed to normalize tool result: {e}")
    return {"ok": False, "result": {}, "error": None}


def extract_error_code(payload: Dict[str, Any]) -> Optional[str]:
    """Extract tool-defined error_code from payload. Tools may put it in result or top-level."""
    result = payload.get("result") or {}
    if isinstance(result, dict):
        code = result.get("error_code")
        if isinstance(code, str) and code:
            return code
    return payload.get("error_code") if isinstance(payload.get("error_code"), str) else None


def update_step_tracking(
    had_non_reasoning_tool: bool,
    effective_steps: int,
    consecutive_reasoning_only: int,
    max_effective_tool_steps: int,
) -> tuple[int, int]:
    """Update step tracking counters and return new values."""
    if had_non_reasoning_tool:
        return effective_steps + 1, 0
    return effective_steps, consecutive_reasoning_only + 1


def should_break_loop(
    effective_steps: int,
    consecutive_reasoning_only: int,
    max_effective_tool_steps: int,
) -> bool:
    """Check if the generation loop should break."""
    return (
        effective_steps >= max_effective_tool_steps
        or consecutive_reasoning_only >= MAX_CONSECUTIVE_REASONING_ONLY
    )
