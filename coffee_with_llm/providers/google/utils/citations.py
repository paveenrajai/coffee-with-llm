from __future__ import annotations

import asyncio
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import httpx

#: DeepGrasp-style inline citation markers produced by inject_inline_citations.
CITATION_MARKER_RE = re.compile(r"\s*\[cite:\s*([^\]]*)\]", re.IGNORECASE)
_HOOK_KEY_RE = re.compile(r'"hook"\s*:\s*"', re.IGNORECASE)


def extract_citations(resp: Any) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []

    def add(
        uri: Optional[str],
        title: Optional[str],
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> None:
        if not (uri or title):
            return
        citations.append(
            {
                "uri": uri,
                "title": title,
                "start_index": start_idx,
                "end_index": end_idx,
            }
        )

    try:
        gm = getattr(resp, "grounding_metadata", None) or getattr(resp, "groundingMetadata", None)
        if gm:
            atts = (
                getattr(gm, "grounding_attributions", None)
                or getattr(gm, "attributions", None)
                or []
            )
            for a in atts:
                web = getattr(a, "web", None) or getattr(a, "source", None) or {}
                uri = getattr(web, "uri", None) or getattr(web, "url", None)
                title = getattr(web, "title", None) or getattr(web, "site", None)
                add(uri, title)
    except Exception:
        pass

    try:
        cm = getattr(resp, "citation_metadata", None) or getattr(resp, "citationMetadata", None)
        if cm:
            sources = getattr(cm, "citation_sources", None) or getattr(cm, "sources", None) or []
            for s in sources:
                uri = getattr(s, "uri", None) or getattr(s, "url", None)
                title = getattr(s, "title", None)
                add(uri, title)
    except Exception:
        pass

    try:
        for cand in getattr(resp, "candidates", []) or []:
            gm = getattr(cand, "grounding_metadata", None) or getattr(
                cand, "groundingMetadata", None
            )
            if gm:
                atts = (
                    getattr(gm, "grounding_attributions", None)
                    or getattr(gm, "attributions", None)
                    or []
                )
                for a in atts:
                    web = getattr(a, "web", None) or getattr(a, "source", None) or {}
                    uri = getattr(web, "uri", None) or getattr(web, "url", None)
                    title = getattr(web, "title", None) or getattr(web, "site", None)
                    add(uri, title)

            cm = getattr(cand, "citation_metadata", None) or getattr(cand, "citationMetadata", None)
            if cm:
                sources = (
                    getattr(cm, "citation_sources", None) or getattr(cm, "sources", None) or []
                )
                for s in sources:
                    uri = getattr(s, "uri", None) or getattr(s, "url", None)
                    title = getattr(s, "title", None)
                    add(uri, title)
    except Exception:
        pass

    try:
        for cand in getattr(resp, "candidates", []) or []:
            for part in getattr(getattr(cand, "content", object()), "parts", []) or []:
                meta = getattr(part, "metadata", None)
                if not meta:
                    continue
                for c in getattr(meta, "citations", None) or []:
                    uri = getattr(c, "uri", None) or getattr(c, "url", None)
                    title = getattr(c, "title", None)
                    start_idx = getattr(c, "start_index", None)
                    end_idx = getattr(c, "end_index", None)
                    add(uri, title, start_idx, end_idx)
    except Exception:
        pass

    seen: set[tuple[Optional[str], Optional[str]]] = set()
    deduped: List[Dict[str, Any]] = []
    for c in citations:
        key = (c.get("uri"), c.get("title"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped


def resolve_vertex_redirect(url: str, client: httpx.Client, cache: Dict[str, str]) -> str:
    try:
        if url in cache:
            return cache[url]
        if "vertexaisearch.cloud.google.com" in (url or "") and "/grounding-api-redirect/" in (
            url or ""
        ):
            try:
                r = client.head(url)
                final_url = str(r.url)
            except Exception:
                r = client.get(url)
                final_url = str(r.url)
            cache[url] = final_url
            return final_url
        return url
    except Exception:
        return url


def resolve_citation_urls(
    citations: List[Dict[str, Any]], client: httpx.Client
) -> List[Dict[str, Any]]:
    cache: Dict[str, str] = {}
    out: List[Dict[str, Any]] = []
    for c in citations:
        u = c.get("uri")
        if isinstance(u, str) and u:
            c = {**c, "uri": resolve_vertex_redirect(u, client, cache)}
        out.append(c)
    return out


def _grounding_metadata(resp: Any) -> Any | None:
    gm = getattr(resp, "grounding_metadata", None) or getattr(resp, "groundingMetadata", None)
    if gm:
        return gm
    cands = getattr(resp, "candidates", []) or []
    if not cands:
        return None
    return getattr(cands[0], "grounding_metadata", None) or getattr(
        cands[0], "groundingMetadata", None
    )


def collect_grounding_urls(resp: Any) -> Set[str]:
    urls: Set[str] = set()
    try:
        gm = _grounding_metadata(resp)
        if gm:
            chunks = getattr(gm, "grounding_chunks", None) or getattr(gm, "chunks", None) or []
            for ch in chunks:
                try:
                    web = getattr(ch, "web", None) or {}
                    url = getattr(web, "uri", None) or getattr(web, "url", None)
                    if url:
                        urls.add(str(url))
                except Exception:
                    continue
        for item in extract_citations(resp):
            uri = item.get("uri")
            if uri:
                urls.add(str(uri))
        return urls
    except Exception:
        return urls


def describe_grounding(resp: Any) -> Dict[str, Any]:
    """Summarize grounding metadata for debugging (live scripts, logs)."""
    gm = _grounding_metadata(resp)
    if not gm:
        return {
            "has_grounding_metadata": False,
            "chunk_count": 0,
            "support_count": 0,
            "web_search_queries": [],
            "urls": [],
        }
    chunks = getattr(gm, "grounding_chunks", None) or getattr(gm, "chunks", None) or []
    supports = getattr(gm, "grounding_supports", None) or getattr(gm, "supports", None) or []
    queries = getattr(gm, "web_search_queries", None) or getattr(gm, "search_queries", None) or []
    return {
        "has_grounding_metadata": True,
        "chunk_count": len(chunks),
        "support_count": len(supports),
        "web_search_queries": list(queries) if queries else [],
        "urls": sorted(collect_grounding_urls(resp)),
    }


def merge_grounding_responses(responses: List[Any]) -> Any:
    """Combine grounding chunks/supports from multiple API responses for injection."""
    from types import SimpleNamespace

    merged_chunks: List[Any] = []
    merged_supports: List[Any] = []
    for resp in responses:
        gm = _grounding_metadata(resp)
        if not gm:
            continue
        chunks = getattr(gm, "grounding_chunks", None) or getattr(gm, "chunks", None) or []
        supports = getattr(gm, "grounding_supports", None) or getattr(gm, "supports", None) or []
        merged_chunks.extend(chunks)
        merged_supports.extend(supports)
    if not merged_chunks and not merged_supports:
        return responses[-1] if responses else SimpleNamespace()
    gm = SimpleNamespace(
        grounding_chunks=merged_chunks,
        grounding_supports=merged_supports,
        chunks=merged_chunks,
        supports=merged_supports,
    )
    return SimpleNamespace(grounding_metadata=gm, candidates=[])


async def async_resolve_urls(
    urls: Set[str], client: httpx.AsyncClient, max_concurrency: int = 4
) -> Dict[str, str]:
    cache: Dict[str, str] = {}

    sem = asyncio.Semaphore(max_concurrency)

    async def resolve_one(u: str) -> None:
        try:
            async with sem:
                if "vertexaisearch.cloud.google.com" in (
                    u or ""
                ) and "/grounding-api-redirect/" in (u or ""):
                    try:
                        r = await client.head(u)
                        final_url = str(r.url)
                    except Exception:
                        r = await client.get(u)
                        final_url = str(r.url)
                    cache[u] = final_url
                else:
                    cache[u] = u
        except Exception:
            cache[u] = u

    tasks = [resolve_one(u) for u in urls]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    return cache


def _chunk_index_to_url(chunks: List[Any]) -> Dict[int, str]:
    idx_to_url: Dict[int, str] = {}
    for idx, ch in enumerate(chunks):
        try:
            web = getattr(ch, "web", None) or {}
            url = getattr(web, "uri", None) or getattr(web, "url", None)
            if url:
                idx_to_url[idx] = str(url)
        except Exception:
            continue
    return idx_to_url


def _citation_urls_from_marker(marker: str) -> List[str]:
    out: List[str] = []
    for raw in marker.split(","):
        cleaned = raw.strip()
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out


def _citation_urls_in_text(text: str) -> List[str]:
    out: List[str] = []
    for match in CITATION_MARKER_RE.finditer(text):
        for url in _citation_urls_from_marker(match.group(1)):
            if url not in out:
                out.append(url)
    return out


_HTTPS_URL_RE = re.compile(r"https?://[^\s\]\)\"'<>]+")


def _normalize_citation_url(url: str) -> str:
    return url.strip().rstrip(".,;").rstrip("/").lower()


def _citation_domain(url: str) -> str:
    try:
        from urllib.parse import urlparse

        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def collect_research_citation_urls(text: str) -> List[str]:
    """URLs from ``[cite:…]`` markers in research notes, else plain https links."""
    urls = _citation_urls_in_text(text)
    if urls:
        return urls
    for match in _HTTPS_URL_RE.finditer(text or ""):
        cleaned = match.group(0).rstrip(".,;")
        if cleaned and cleaned not in urls:
            urls.append(cleaned)
    return urls


def extract_citation_urls_from_text(text: str) -> List[str]:
    """Every URL in ``[cite:…]`` markers, in order, de-duplicated."""
    return _citation_urls_in_text(text)


def _snap_citation_urls(
    cites: List[str],
    allowed_urls: List[str],
    *,
    max_urls: int | None = None,
    fallback: bool = True,
) -> List[str]:
    if not allowed_urls:
        return []

    allowed_exact = {_normalize_citation_url(url): url for url in allowed_urls}
    by_domain: Dict[str, str] = {}
    for url in allowed_urls:
        domain = _citation_domain(url)
        if domain:
            by_domain.setdefault(domain, url)

    kept: List[str] = []
    for cite in cites:
        normalized = _normalize_citation_url(cite)
        if normalized in allowed_exact:
            final = allowed_exact[normalized]
        else:
            domain = _citation_domain(cite)
            final = by_domain.get(domain)
        if final and final not in kept:
            kept.append(final)

    if not kept and fallback and allowed_urls:
        kept = [allowed_urls[0]]
    if max_urls is not None and len(kept) > max_urls:
        kept = kept[:max_urls]
    return kept


def _cite_in_allowlist(cite: str, allowed_urls: List[str]) -> bool:
    if not cite or not allowed_urls:
        return False
    normalized = _normalize_citation_url(cite)
    allowed_exact = {_normalize_citation_url(url) for url in allowed_urls}
    if normalized in allowed_exact:
        return True
    domain = _citation_domain(cite)
    return any(_citation_domain(url) == domain for url in allowed_urls)


def partition_citation_urls(
    urls: List[str],
    allowed_urls: List[str],
) -> tuple[List[str], List[str]]:
    """Split ``urls`` into those allowed by the research pass vs hallucinated."""
    matched: List[str] = []
    unmatched: List[str] = []
    for cite in urls:
        if _cite_in_allowlist(cite, allowed_urls):
            if cite not in matched:
                matched.append(cite)
        elif cite not in unmatched:
            unmatched.append(cite)
    return matched, unmatched


def restrict_inline_citations(text: str, allowed_urls: List[str]) -> str:
    """Snap every ``[cite:…]`` marker in prose/markdown to the research allowlist."""
    if not text or not allowed_urls:
        return text

    def replace_marker(match: re.Match[str]) -> str:
        cites = _citation_urls_from_marker(match.group(1))
        kept = _snap_citation_urls(cites, allowed_urls)
        if not kept:
            return ""
        return f" [cite:{', '.join(kept)}]"

    return CITATION_MARKER_RE.sub(replace_marker, text)


def restrict_hook_citation(hook: str, allowed_urls: List[str]) -> str:
    """Keep only allowed URLs in a hook's trailing cite marker."""
    body = _visible_text(hook).strip()
    if not allowed_urls:
        return body

    kept = _snap_citation_urls(_citation_urls_in_text(hook), allowed_urls, max_urls=1)
    return f"{body} [cite:{', '.join(kept)}]" if kept else body


def restrict_json_hook_citations(text: str, allowed_urls: List[str]) -> str:
    """Snap JSON card hook citations to URLs from the research pass."""
    if not text or not allowed_urls:
        return text

    import json as json_module

    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, flags=re.S)
    payload = (fenced.group(1) if fenced else text).strip()
    try:
        cards = json_module.loads(payload)
    except json_module.JSONDecodeError:
        return text
    if not isinstance(cards, list):
        return text

    changed = False
    for card in cards:
        if not isinstance(card, dict) or "hook" not in card:
            continue
        original = str(card.get("hook") or "")
        restricted = restrict_hook_citation(original, allowed_urls)
        if restricted != original:
            changed = True
        card["hook"] = restricted

    if not changed:
        return text

    new_payload = json_module.dumps(cards, indent=2)
    if fenced:
        return text[: fenced.start(1)] + new_payload + text[fenced.end(1) :]
    return new_payload


def _visible_text(text: str) -> str:
    return CITATION_MARKER_RE.sub("", text)


def _json_hook_regions(text: str) -> List[Tuple[int, int, int]]:
    """Return ``(value_start, quote_pos, assign_through)`` for each JSON hook field."""
    regions: List[Tuple[int, int, int]] = []
    for match in _HOOK_KEY_RE.finditer(text):
        value_start = match.end()
        index = value_start
        while index < len(text):
            char = text[index]
            if char == "\\":
                index += 2
                continue
            if char == '"':
                break
            index += 1
        quote_pos = index
        assign_through = quote_pos
        cursor = quote_pos + 1
        while cursor < len(text):
            tail = text[cursor:]
            gap = len(tail) - len(tail.lstrip())
            cite_match = CITATION_MARKER_RE.match(tail.lstrip())
            if not cite_match:
                break
            cursor += gap + cite_match.end()
            assign_through = cursor
        regions.append((value_start, quote_pos, assign_through))
    return regions


def _region_index_for_position(pos: int, regions: List[Tuple[int, int, int]]) -> int:
    for index, (start, _quote, end) in enumerate(regions):
        if start <= pos < end:
            return index
    best_index = 0
    best_distance = abs(pos - regions[0][0])
    for index, (start, quote, end) in enumerate(regions):
        if pos < start:
            distance = start - pos
        else:
            distance = pos - max(quote, end)
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


def _urls_for_hook_span(
    supports: List[Any],
    idx_to_url: Dict[int, str],
    resolve_url: Callable[[str], str],
    value_start: int,
    quote_pos: int,
) -> List[str]:
    urls: List[str] = []
    for support in supports:
        try:
            seg = getattr(support, "segment", None) or {}
            end_idx = getattr(seg, "end_index", None)
            if not isinstance(end_idx, int):
                continue
            if not (value_start <= end_idx <= quote_pos):
                continue
            idxs = (
                getattr(support, "grounding_chunk_indices", None)
                or getattr(support, "indices", None)
                or []
            )
            for raw_index in idxs or []:
                key = int(raw_index) if isinstance(raw_index, int) else raw_index
                raw_url = idx_to_url.get(key)
                if not raw_url:
                    continue
                resolved = resolve_url(raw_url)
                if resolved not in urls:
                    urls.append(resolved)
        except Exception:
            continue
    return urls


def _rewrite_hook_value(hook_text: str, urls: List[str]) -> str:
    merged = list(_citation_urls_in_text(hook_text))
    for url in urls:
        if url not in merged:
            merged.append(url)
    body = _visible_text(hook_text).strip()
    if not merged:
        return body
    return f"{body} [cite: {', '.join(merged)}]"


def _inject_json_hook_citations(
    text: str,
    *,
    supports: List[Any],
    idx_to_url: Dict[int, str],
    resolve_url: Callable[[str], str],
) -> str:
    regions = _json_hook_regions(text)
    if not regions:
        return text

    buckets: List[List[str]] = [[] for _ in regions]
    for match in CITATION_MARKER_RE.finditer(text):
        region_index = _region_index_for_position(match.start(), regions)
        for url in _citation_urls_from_marker(match.group(1)):
            if url not in buckets[region_index]:
                buckets[region_index].append(url)

    for index, (value_start, quote_pos, _assign_through) in enumerate(regions):
        hook_text = text[value_start:quote_pos]
        support_urls = _urls_for_hook_span(
            supports, idx_to_url, resolve_url, value_start, quote_pos
        )
        for url in support_urls:
            if url not in buckets[index]:
                buckets[index].append(url)
        for url in _citation_urls_in_text(hook_text):
            if url not in buckets[index]:
                buckets[index].append(url)

    chunk_urls = [resolve_url(idx_to_url[i]) for i in sorted(idx_to_url)]
    chunk_cursor = 0
    for index, bucket in enumerate(buckets):
        if bucket:
            continue
        if not chunk_urls:
            continue
        bucket.append(chunk_urls[chunk_cursor % len(chunk_urls)])
        chunk_cursor += 1

    for index in reversed(range(len(regions))):
        value_start, quote_pos, assign_through = regions[index]
        new_hook = _rewrite_hook_value(text[value_start:quote_pos], buckets[index])
        tail = text[quote_pos:assign_through]
        text = text[:value_start] + new_hook + tail + text[assign_through:]
    return text


def _inject_prose_citations(
    text: str,
    *,
    supports: List[Any],
    idx_to_url: Dict[int, str],
    resolve_url: Callable[[str], str],
) -> str:
    insertions: List[Dict[str, Any]] = []
    for support in supports:
        try:
            seg = getattr(support, "segment", None) or {}
            end_idx = getattr(seg, "end_index", None)
            idxs = (
                getattr(support, "grounding_chunk_indices", None)
                or getattr(support, "indices", None)
                or []
            )
            if not isinstance(end_idx, int):
                continue
            urls: List[str] = []
            seen_local: set[str] = set()
            for raw_index in idxs or []:
                key = int(raw_index) if isinstance(raw_index, int) else raw_index
                raw_url = idx_to_url.get(key)
                if not raw_url:
                    continue
                resolved = resolve_url(raw_url)
                if resolved not in seen_local:
                    seen_local.add(resolved)
                    urls.append(resolved)
            if urls:
                pos = max(0, min(len(text), end_idx))
                insertions.append({"pos": pos, "urls": urls})
        except Exception:
            continue

    if not insertions:
        return text

    merged: Dict[int, List[str]] = {}
    for ins in insertions:
        pos = int(ins["pos"])
        current = merged.get(pos, [])
        for url in ins["urls"]:
            if url not in current:
                current.append(url)
        merged[pos] = current

    for pos in sorted(merged.keys(), reverse=True):
        urls = merged[pos]
        marker = " [cite: " + ", ".join(urls) + "]"
        if 0 <= pos <= len(text):
            text = text[:pos] + marker + text[pos:]
    return text


def inject_inline_citations(text: str, resp: Any, resolve_url) -> str:
    """Inject ``[cite: url]`` markers using Gemini grounding metadata.

    For JSON card output with ``"hook"`` fields, citations are placed inside each
    hook string (before its closing quote) using grounding supports whose
    ``end_index`` falls within the hook prose, then any remaining grounding chunks
    are round-robin assigned to hooks still missing markers.

    For plain prose, falls back to offset-based insertion at each support end_index.
    """
    try:
        if not text:
            return text

        gm = _grounding_metadata(resp)
        if not gm:
            return text

        chunks = getattr(gm, "grounding_chunks", None) or getattr(gm, "chunks", None) or []
        supports = getattr(gm, "grounding_supports", None) or getattr(gm, "supports", None) or []

        idx_to_url = _chunk_index_to_url(chunks)
        if not idx_to_url:
            for index, url in enumerate(sorted(collect_grounding_urls(resp))):
                idx_to_url[index] = url
        if not idx_to_url:
            return text

        if _json_hook_regions(text):
            return _inject_json_hook_citations(
                text,
                supports=supports,
                idx_to_url=idx_to_url,
                resolve_url=resolve_url,
            )

        if supports:
            return _inject_prose_citations(
                text,
                supports=supports,
                idx_to_url=idx_to_url,
                resolve_url=resolve_url,
            )
        return text
    except Exception:
        return text
