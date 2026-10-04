# Changelog

All notable changes to `coffee_with_llm` are documented here.

## [0.8.2] - 2026-10-04

### Added

- **`AskResult.url_retrievals`**: for Gemini, each link URL context tried to open, as `UrlRetrieval(url, ok, status)`. A site that refuses the fetch is answered from search; this is how a caller can tell.

### Fixed

- **A Gemini tool loop that hits its step cap ends in an answer.** It stopped right after a tool call and raised "Empty response", losing every call it made. It now asks once more with tool calls off, as the Anthropic, OpenAI and Inception clients already did.
- **An empty response is asked again**, on calls without tools. A tool loop is not retried: that would run every tool again.
- **Tool result fields beside `result` reach the model.** They were dropped without a word, so a tool returning `{"ok": True, "answer": ...}` handed the model an empty result. They are now folded into `result`.

## [0.8.1] - 2026-10-04

### Changed

- **Gemini URL context**: with search attached, `ask()` (generate_content) now attaches the URL context tool beside Google Search, so a link in the prompt is opened and read. A page Google's index does not hold was reported missing before.

## [0.8.0] - 2026-08-08

### Added

- **Gemini Interactions API** — `AskLLM.ask_interaction()` with server-side session state (`previous_interaction_id` on `AskResult`).
- **`google_api_mode`** — choose `generate_content` (default) or `interactions` per client.
- **Grounded JSON flow** — `ask_with_grounded_json()` for two-pass search + JSON formatting (curator card hooks).
- **Grounded markdown flow** — `ask_with_grounded_markdown()` and `verify_markdown_citations()` (orchestrator-style web + markdown).
- **Citation helpers** — `extract_citation_urls_from_text`, `restrict_inline_citations`, `restrict_json_hook_citations`, `partition_citation_urls`.
- **Link verification** — `coffee_with_llm.link_check` treats 403/401 bot walls as reachable.
- **Live smoke scripts** — `test_json_hook_citations.py`, `test_interactions_api.py`, `test_markdown_web_citations.py` with per-stage timing (`test_timing.py`).

### Changed

- **`google-genai` ≥ 2.0.0** required for Interactions API (breaking vs 1.x schema).
- **JSON hook citations** — research notes and allowed URL list are passed into pass 2; hallucinated cites are stripped.
- **Interactions response parsing** — reads `output_text` / `steps` schema (not legacy `outputs`).

### Fixed

- `GoogleTextClient` import typo (`GoogleChatClient`) in `AskLLM._generate`.
- Flaky missing-key unit tests when a repo `.env` is present (patch `Config.from_env` instead of clearing `os.environ`).

[0.8.2]: https://github.com/paveenrajai/coffee-with-llm/compare/0.8.1...0.8.2
[0.8.1]: https://github.com/paveenrajai/coffee-with-llm/compare/0.8.0...0.8.1
[0.8.0]: https://github.com/paveenrajai/coffee-with-llm/compare/v0.7.1...v0.8.0
