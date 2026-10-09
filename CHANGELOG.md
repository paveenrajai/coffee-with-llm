# Changelog

All notable changes to `coffee_with_llm` are documented here.

## [0.11.0] - 2026-10-09

### Added

- **`StreamResult.usage_so_far`: the usage the chunks received so far carried**, priced like `usage`, and `None` while no chunk has carried any. Reading it never closes, drains or awaits the stream, so a caller can charge for a stream it is cancelling. Once the stream has ended it is `usage`.

### Changed

- **A Gemini stream stopped part way is closed, not read to its end.** A stream closed by its caller, or whose task was cancelled, read the rest of the generation to fill its usage: Google generated and billed it, nobody saw it, and a caller that closed a stream to retry waited for the whole tail first. The stream is now closed at once, which stops the generation, and its usage is the running total of the chunks received. Other providers are unchanged.

### Fixed

- **The Gemini usage sink no longer lags one chunk behind the text.** It took a chunk's usage after the chunk's text was handed out, so a caller holding chunk N's text read chunk N-1's total.

## [0.10.1] - 2026-10-09

### Changed

- **`input_tokens` leaves out cache reads for Google, OpenAI and Inception, as documented.** Each of them counts cache reads inside its own input count. `input_tokens` is now that count less `cached_tokens`, never below 0, so the buckets are disjoint for every provider, as they already were for Anthropic. The Interactions API is counted the same way.
- **`total_tokens` is `input_tokens + output_tokens` for every provider**, now with the uncached input. The Interactions API, OpenAI and Inception passed the provider's own total through, which counts the cache (and, for the Interactions API, other internal tokens); it is now the same sum as everywhere else.

### Fixed

- **Cached tokens are counted once in `prompt_tokens` and `billable_tokens`.** Google, the Interactions API and Inception counted a cache read in `input_tokens` and again in `cached_tokens`, and OpenAI would have once its reads were read. On 2026-10-09 a Gemini board with a 90,116-token prompt, 36,841 of them cached, reported `prompt_tokens=126957`. The call Inception makes to finalize an empty answer counted its cache reads as uncached input.
- **OpenAI cache reads are read.** They were looked for at `usage.cached_tokens`, which the Responses API does not send; they are at `usage.input_tokens_details.cached_tokens`. Every cache read was billed at the full input price.
- **Anthropic is no longer undercharged when its cache read is smaller than its uncached input.** `estimate_cost` guessed from which was larger whether `cached_tokens` was inside `input_tokens`, and took Anthropic's read, a bucket of its own, out of its input. It now bills each bucket once, at its own rate. Gemini's cost is unchanged: the board above still costs $0.091424.

## [0.10.0] - 2026-10-09

### Added

- **`TokenUsage.served_model`: the model that served the call**, as the provider named it in its reply: Gemini's `model_version`, and `model` from OpenAI, Anthropic, Inception and the Interactions API. For an alias, it is whatever the alias points to today.
- **Gemini 3 prices** from Google's pricing page: 3.8, 3.7 and 3.6 Flash ($0.75 / $3.75 / $0.075 cached through 2026, doubling from 2027-01-01), 3.5 Flash, 3.5 Flash-Lite and 3.1 Flash-Lite. `estimate_cost` takes `on=` for a price that changes on a date, and defaults to today.

### Changed

- **A call is priced at the model that served it.** `gemini-flash-lite-latest` was priced as 2.5 Flash-Lite while Google served 3.5 Flash-Lite, at three times the input price and six times the output; `gemini-flash-latest` as 2.5 Flash while it served 3.8 Flash. The alias rows are gone: an alias is swapped with every release and has no price of its own. The model asked for is priced only when the provider did not say what served it.
- **A model with no price is left unpriced and logged**, never priced as another model. `gemini-3.8-flash` had none, so every call to it had `cost_usd=None`.

### Fixed

- **2.5 Flash-Lite and 3.1 Flash-Lite are priced as Flash-Lite.** 2.5 Flash-Lite came after 2.5 Flash in the table and matched it first; 3.1 Flash-Lite without `-preview` matched 3.1 Flash.

## [0.9.0] - 2026-10-09

### Added

- **`AskResult.stop` and `StreamResult.stop`: why a call stopped**, for every provider: `Stop(reason, raw)`, where `reason` is one of `StopReason` (`end`, `max_tokens`, `content_filter`, `tool_use`, `other`) and `raw` is the provider's own word. A call cut off at `max_tokens` ended exactly like a finished one, so a caller could store half an answer as whole. `stop.truncated` is the check. A stream keeps its `Stop` and never yields it.
- **`TokenUsage.reasoning_tokens`**: how much of `output_tokens` was thinking, from Google, OpenAI and Inception. Anthropic counts thinking in `output_tokens` without saying how much, so it is `None` there.

### Changed

- **Gemini's `output_tokens` and `cost_usd` include its thinking**, which Google bills as output. They left it out, so a call that thought for 15,000 tokens and wrote 600 was reported and priced as 600. The Interactions API is counted the same way.
- **Gemini 3 and later are sent `thinking_level`, not `thinking_budget`.** `reasoning_effort` `low`, `medium` and `high` become the level of the same name, which Google recommends for Gemini 3; a request carrying both is refused. Gemini 2.x, and a model whose name does not say its generation (`gemini-flash-latest`), are still sent a budget, which Gemini 3 accepts and 2.x requires.
- **A provider's `generate` returns `(text, usage, stop)`.** One registered from outside that still returns `(text, usage)` keeps working, and says nothing about why it stopped.

## [0.8.4] - 2026-10-05

### Fixed

- **Inline `[cite: …]` markers land where their sentence ends.** Gemini gives a grounded segment's end in bytes, and it was used as a character position, so every non-ASCII character before it (a curly apostrophe is three bytes) pushed the marker further on: "Vin [cite: …]cent Bernat". Facts split at those markers credited the wrong page. Both plain answers and JSON `hook` fields now read it as a byte offset.

## [0.8.3] - 2026-10-05

### Fixed

- **Grounding redirects resolve from their own reply.** Each `vertexaisearch.cloud.google.com/grounding-api-redirect/…` citation is now resolved from the redirect's `Location` header instead of being followed to the cited page. Following it made the page's speed and manners decide: a site that failed the request left the redirect standing as the citation. It also no longer sends a request to every cited site.

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

[0.8.3]: https://github.com/paveenrajai/coffee-with-llm/compare/0.8.2...0.8.3
[0.8.2]: https://github.com/paveenrajai/coffee-with-llm/compare/0.8.1...0.8.2
[0.8.1]: https://github.com/paveenrajai/coffee-with-llm/compare/0.8.0...0.8.1
[0.8.0]: https://github.com/paveenrajai/coffee-with-llm/compare/v0.7.1...v0.8.0
