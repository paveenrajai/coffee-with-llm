from __future__ import annotations

import logging
import os
import hashlib
import time
from typing import Any, Dict, List, Optional
from collections import OrderedDict

from google import genai
from .utils.citations import (
    inject_inline_citations,
    collect_grounding_urls,
    async_resolve_urls,
)
import httpx

from ...exceptions import ConfigurationError, APIError

logger = logging.getLogger(__name__)


class GoogleTextClient:
    def __init__(self) -> None:
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise ConfigurationError("GOOGLE_API_KEY environment variable is not set")

        try:
            self._client = genai.Client(api_key=api_key)
        except ImportError as e:
            raise ConfigurationError(
                "Google GenAI package not installed. Install with: pip install google-genai"
            ) from e
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Google client: {e}") from e

        self._cached_contexts: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._max_cached_contexts = 10
        self._context_ttl_seconds = 3600

    def _build_config_dict(
        self,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build generation config as dict for Google Gemini API."""
        config_dict: Dict[str, Any] = {}

        is_json_response = (
            response_format
            and isinstance(response_format, dict)
            and response_format.get("type") == "json_schema"
            and response_format.get("json_schema")
        )

        if not is_json_response:
            config_dict["tools"] = [{"google_search": {}}]

        if max_tokens is not None:
            config_dict["max_output_tokens"] = max_tokens
        if temperature is not None:
            config_dict["temperature"] = temperature
        if top_p is not None:
            config_dict["top_p"] = top_p

        if response_format and isinstance(response_format, dict):
            if response_format.get("type") == "json_schema":
                json_schema = response_format.get("json_schema")
                if json_schema:
                    config_dict["response_mime_type"] = "application/json"
                    config_dict["response_json_schema"] = json_schema

        return config_dict

    def _get_system_prompt_hash(self, system_instruct: str) -> str:
        """Generate hash for system prompt to use as cache key."""
        return hashlib.sha256(system_instruct.encode()).hexdigest()

    async def _get_or_create_cached_context(
        self,
        system_instruct: str,
        model: str,
    ) -> Optional[str]:
        """Get or create cached context for static system prompt."""
        if not system_instruct or not system_instruct.strip():
            return None

        enable_explicit_cache = (
            os.getenv("GOOGLE_EXPLICIT_CACHE", "1") or "1"
        ).strip() not in ("0", "false", "False")

        if not enable_explicit_cache:
            return None

        model_lower = model.lower()
        if not ("2.5" in model_lower or "gemini-2" in model_lower):
            return None

        prompt_hash = self._get_system_prompt_hash(system_instruct)
        now = time.time()

        if prompt_hash in self._cached_contexts:
            context_name, created_at = self._cached_contexts[prompt_hash]
            if now - created_at < self._context_ttl_seconds:
                self._cached_contexts.move_to_end(prompt_hash)
                return context_name
            else:
                del self._cached_contexts[prompt_hash]

        try:
            static_content = system_instruct

            try:
                cached_context = await self._client.aio.cached_contents.create(
                    model=model,
                    contents=[static_content],
                    ttl=self._context_ttl_seconds,
                )
            except AttributeError:
                try:
                    cached_context = (
                        await self._client.aio.models.cached_contents.create(
                            model=model,
                            contents=[static_content],
                            ttl=self._context_ttl_seconds,
                        )
                    )
                except (AttributeError, Exception):
                    return None

            context_name = getattr(cached_context, "name", None)
            if context_name:
                if len(self._cached_contexts) >= self._max_cached_contexts:
                    self._cached_contexts.popitem(last=False)

                self._cached_contexts[prompt_hash] = (context_name, now)
                self._cached_contexts.move_to_end(prompt_hash)

                logger.debug(f"Google cache created: {context_name[:50]}...")
                return context_name
        except Exception as e:
            logger.debug(f"Google explicit caching unavailable: {e}")
            return None

        return None

    async def generate(
        self,
        *,
        prompt: str,
        model: str,
        system_instruct: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if not model or not model.strip():
            raise ValueError("Model name cannot be empty")

        cached_context_name = await self._get_or_create_cached_context(
            system_instruct, model
        )

        contents: List[Any] = []

        if cached_context_name:
            # Add conversation history if provided
            if messages:
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    # Google uses "user" and "model" roles
                    google_role = "model" if role == "assistant" else "user"
                    contents.append({"role": google_role, "parts": [{"text": content}]})
            contents.append(prompt)
        else:
            # Build conversation with system prompt merged into first message
            if messages:
                for i, msg in enumerate(messages):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    google_role = "model" if role == "assistant" else "user"
                    # Merge system instruct with first user message
                    if i == 0 and role == "user" and system_instruct:
                        content = f"{system_instruct}\n\n{content}"
                    contents.append({"role": google_role, "parts": [{"text": content}]})
                # Add current prompt
                contents.append({"role": "user", "parts": [{"text": prompt}]})
            else:
                merged_prompt = (
                    f"{system_instruct}\n\n{prompt}"
                    if (system_instruct or "").strip()
                    else prompt
                )
                contents.append(merged_prompt)

        config = self._build_config_dict(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            response_format=response_format,
        )

        request_kwargs: Dict[str, Any] = {
            "model": model,
            "contents": contents,
            "config": config,
        }

        if cached_context_name:
            request_kwargs["cached_content"] = cached_context_name

        try:
            resp = await self._client.aio.models.generate_content(**request_kwargs)
        except Exception as e:
            logger.error(f"Google API call failed: {e}")
            raise APIError(f"Google API request failed: {e}") from e

        text: str = str(
            getattr(resp, "text", None) or getattr(resp, "output_text", "") or ""
        )

        if not text.strip():
            raise APIError("Empty response received from Google API")

        try:
            enable_inline = (
                os.getenv("GOOGLE_INLINE_CITATIONS", "1") or "1"
            ).strip() not in ("0", "false", "False")
            if enable_inline:
                urls = collect_grounding_urls(resp)
                if urls:
                    async with httpx.AsyncClient(
                        follow_redirects=True, timeout=2
                    ) as http:
                        resolved = await async_resolve_urls(
                            urls, http, max_concurrency=4
                        )
                    text = inject_inline_citations(
                        text,
                        resp,
                        resolve_url=lambda u: resolved.get(u, u),
                    )
        except Exception as e:
            logger.debug(f"Failed to inject citations: {e}")

        return text
