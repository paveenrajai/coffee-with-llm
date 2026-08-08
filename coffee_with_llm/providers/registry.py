"""Provider registry – model-to-provider resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..exceptions import ValidationError
from .anthropic import AnthropicMessagesClient
from .google import GoogleInteractionsClient, GoogleTextClient
from .google.api_mode import DEFAULT_GOOGLE_API_MODE, GoogleApiMode
from .inception import InceptionChatClient
from .openai import OpenAIResponsesClient
from .protocol import ProviderProtocol

if TYPE_CHECKING:
    from ..config import Config

# First path segment must be one of these to treat ``provider/model`` as a prefix form.
_PROVIDER_PREFIXES = frozenset(
    {"anthropic", "claude", "gemini", "google", "inception", "openai"}
)


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


def _route_is_inception(route_key: str) -> bool:
    return route_key in ("inception",) or route_key.startswith(("mercury", "inception"))


def get_google_interactions_client(
    model: str,
    config: "Config",
    *,
    request_timeout: Optional[float] = None,
    google_attach_search_tool: bool = True,
) -> GoogleInteractionsClient:
    """Interactions API client for a Google/Gemini model (independent of ``google_api_mode``)."""
    _, route_key = split_provider_model(model)
    if not _route_is_google(route_key):
        raise ValidationError(
            f"Interactions API is only available for Google/Gemini models, not {model!r}."
        )
    kwargs: dict = {"config": config}
    if request_timeout is not None:
        kwargs["request_timeout"] = request_timeout
    return GoogleInteractionsClient(
        **kwargs,
        google_attach_search_tool=google_attach_search_tool,
    )


def get_provider(
    model: str,
    config: "Config",
    *,
    request_timeout: Optional[float] = None,
    google_explicit_cache: bool = True,
    google_inline_citations: bool = True,
    google_attach_search_tool: bool = True,
    anthropic_prompt_cache: bool = True,
    google_api_mode: GoogleApiMode = DEFAULT_GOOGLE_API_MODE,
) -> ProviderProtocol:
    """Return the appropriate provider client for the given model name."""
    _, route_key = split_provider_model(model)
    kwargs: dict = {"config": config}
    if request_timeout is not None:
        kwargs["request_timeout"] = request_timeout
    if _route_is_anthropic(route_key):
        return AnthropicMessagesClient(
            **kwargs,
            anthropic_prompt_cache=anthropic_prompt_cache,
        )
    if _route_is_google(route_key):
        if google_api_mode == "interactions":
            return GoogleInteractionsClient(
                **kwargs,
                google_attach_search_tool=google_attach_search_tool,
            )
        return GoogleTextClient(
            **kwargs,
            google_explicit_cache=google_explicit_cache,
            google_inline_citations=google_inline_citations,
            google_attach_search_tool=google_attach_search_tool,
        )
    if _route_is_inception(route_key):
        return InceptionChatClient(**kwargs)
    return OpenAIResponsesClient(**kwargs)
