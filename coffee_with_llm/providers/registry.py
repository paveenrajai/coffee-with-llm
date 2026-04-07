"""Provider registry – model-to-provider resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..exceptions import ValidationError
from .anthropic import AnthropicMessagesClient
from .google import GoogleTextClient
from .openai import OpenAIResponsesClient
from .protocol import ProviderProtocol

if TYPE_CHECKING:
    from ..config import Config

# First path segment must be one of these to treat ``provider/model`` as a prefix form.
_PROVIDER_PREFIXES = frozenset({"anthropic", "claude", "gemini", "google", "openai"})


def split_provider_model(model: str) -> tuple[str, str]:
    """
    Split optional ``provider/model`` id.

    Returns:
        (api_model_id, route_key): ``api_model_id`` is what provider APIs receive.
        ``route_key`` picks the client: explicit provider name, or the full legacy
        id lowercased when no recognized prefix is used.
    """
    m = (model or "").strip()
    if not m:
        return ("", "")

    if "/" in m:
        prefix, _, rest = m.partition("/")
        p_low = prefix.strip().lower()
        rest_stripped = rest.strip()
        if p_low in _PROVIDER_PREFIXES:
            if not rest_stripped:
                raise ValidationError(
                    "Model id is required after provider prefix (e.g. google/gemma-2-9b-it)"
                )
            return (rest_stripped, p_low)

    return (m, m.lower())


def _route_is_anthropic(route_key: str) -> bool:
    return route_key in ("anthropic", "claude") or route_key.startswith(
        ("claude", "anthropic")
    )


def _route_is_google(route_key: str) -> bool:
    return route_key in ("google", "gemini") or route_key.startswith(("gemini", "google"))


def get_provider(
    model: str,
    config: "Config",
    *,
    request_timeout: Optional[float] = None,
    google_explicit_cache: bool = True,
    google_inline_citations: bool = True,
) -> ProviderProtocol:
    """Return the appropriate provider client for the given model name."""
    _, route_key = split_provider_model(model)
    kwargs: dict = {"config": config}
    if request_timeout is not None:
        kwargs["request_timeout"] = request_timeout
    if _route_is_anthropic(route_key):
        return AnthropicMessagesClient(**kwargs)
    if _route_is_google(route_key):
        return GoogleTextClient(
            **kwargs,
            google_explicit_cache=google_explicit_cache,
            google_inline_citations=google_inline_citations,
        )
    return OpenAIResponsesClient(**kwargs)
