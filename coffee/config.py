"""Configuration with env validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Optional

from .exceptions import ConfigurationError

DEFAULT_REQUEST_TIMEOUT = 60.0
MAX_REQUEST_TIMEOUT = 600.0


@dataclass
class Config:
    """Centralized configuration with env validation."""

    openai_api_key: Optional[str] = field(default=None)
    anthropic_api_key: Optional[str] = field(default=None)
    google_api_key: Optional[str] = field(default=None)
    request_timeout: Optional[float] = field(default=DEFAULT_REQUEST_TIMEOUT)

    @classmethod
    def from_env(cls) -> Config:
        """Load config from environment variables."""
        timeout_str = os.getenv("COFFEE_REQUEST_TIMEOUT", "60")
        try:
            timeout = float(timeout_str) if timeout_str else None
            if timeout is not None and (timeout <= 0 or timeout > MAX_REQUEST_TIMEOUT):
                timeout = DEFAULT_REQUEST_TIMEOUT
        except (ValueError, TypeError):
            timeout = DEFAULT_REQUEST_TIMEOUT

        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            google_api_key=os.getenv("GOOGLE_API_KEY") or None,
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
