from __future__ import annotations


class AskLLMError(Exception):
    """Base exception for coffee library."""

    pass


class ConfigurationError(AskLLMError):
    """Raised when there's a configuration issue."""

    pass


class APIError(AskLLMError):
    """Raised when an API call fails."""

    pass


class ValidationError(AskLLMError):
    """Raised when input validation fails."""

    pass


class RateLimitError(AskLLMError):
    """Raised when API rate limit is exceeded."""

    pass
