"""Configuration with env validation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

from .exceptions import ConfigurationError

DEFAULT_REQUEST_TIMEOUT = 60.0
MAX_REQUEST_TIMEOUT = 600.0


def _dotenv_values() -> dict[str, str]:
    """
    Read the project ``.env`` **without** touching ``os.environ``.

    A library must not export a ``.env`` into the caller's process: every other
    component then silently inherits whatever that file happens to contain, and
    the effect depends on whether a client was constructed first. Returning the
    values instead keeps the file's reach limited to this config object.
    """
    try:
        from dotenv import dotenv_values
    except ImportError:
        return {}
    # Prefer repo-root .env when imported from an installed package / scripts/.
    here = Path(__file__).resolve()
    for candidate in (Path.cwd() / ".env", here.parents[1] / ".env"):
        if candidate.is_file():
            return {k: v for k, v in dotenv_values(candidate).items() if v is not None}
    return {k: v for k, v in dotenv_values().items() if v is not None}


def _env_value(name: str, file_values: Mapping[str, str]) -> Optional[str]:
    """Real environment wins; ``.env`` is a fallback (matches ``override=False``)."""
    return os.environ.get(name) or file_values.get(name) or None


@dataclass
class Config:
    """Centralized configuration with env validation."""

    openai_api_key: Optional[str] = field(default=None)
    anthropic_api_key: Optional[str] = field(default=None)
    google_api_key: Optional[str] = field(default=None)
    inception_api_key: Optional[str] = field(default=None)
    request_timeout: Optional[float] = field(default=DEFAULT_REQUEST_TIMEOUT)

    @classmethod
    def from_env(cls) -> Config:
        """
        Load config from environment variables, falling back to a project ``.env``.

        The ``.env`` is read into a local mapping — calling this never mutates
        ``os.environ``.
        """
        file_values = _dotenv_values()
        timeout_str = _env_value("COFFEE_REQUEST_TIMEOUT", file_values) or "60"
        try:
            timeout = float(timeout_str) if timeout_str else None
            if timeout is not None and (timeout <= 0 or timeout > MAX_REQUEST_TIMEOUT):
                timeout = DEFAULT_REQUEST_TIMEOUT
        except (ValueError, TypeError):
            timeout = DEFAULT_REQUEST_TIMEOUT

        return cls(
            openai_api_key=_env_value("OPENAI_API_KEY", file_values),
            anthropic_api_key=_env_value("ANTHROPIC_API_KEY", file_values),
            google_api_key=_env_value("GOOGLE_API_KEY", file_values),
            inception_api_key=_env_value("INCEPTION_API_KEY", file_values),
            request_timeout=timeout,
        )

    def with_request_timeout(self, timeout: Optional[float]) -> Config:
        """Return a new Config with request_timeout overridden."""
        if timeout is None:
            return self
        return replace(self, request_timeout=timeout)

    def require_openai_key(self) -> str:
        """Return OpenAI key or raise ConfigurationError."""
        if not self.openai_api_key:
            raise ConfigurationError("OpenAI API key is not configured")
        return self.openai_api_key

    def require_anthropic_key(self) -> str:
        """Return Anthropic key or raise ConfigurationError."""
        if not self.anthropic_api_key:
            raise ConfigurationError("Anthropic API key is not configured")
        return self.anthropic_api_key

    def require_google_key(self) -> str:
        """Return Google key or raise ConfigurationError."""
        if not self.google_api_key:
            raise ConfigurationError("Google API key is not configured")
        return self.google_api_key

    def require_inception_key(self) -> str:
        """Return Inception key or raise ConfigurationError."""
        if not self.inception_api_key:
            raise ConfigurationError("Inception API key is not configured")
        return self.inception_api_key
