# Changelog

All notable changes to `coffee_with_llm` are documented here.

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

[0.8.0]: https://github.com/paveenrajai/coffee-with-llm/compare/v0.7.1...v0.8.0
