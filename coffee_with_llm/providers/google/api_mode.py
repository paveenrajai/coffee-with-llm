"""Google Gemini API surface selection."""

from __future__ import annotations

from typing import Literal

#: ``generate_content`` — classic Models API (default, stable for JSON + search).
#: ``interactions`` — Interactions API (agent sessions, server-side state).
GoogleApiMode = Literal["generate_content", "interactions"]

DEFAULT_GOOGLE_API_MODE: GoogleApiMode = "generate_content"
