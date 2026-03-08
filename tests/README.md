# Tests

This directory contains unit tests for the `ask-llm` package.

## Running Tests

### Install test dependencies

```bash
pip install -e ".[dev]"
```

### Run all tests

```bash
pytest
```

### Run with coverage

```bash
pytest --cov=ask_llm --cov-report=html
```

### Run specific test file

```bash
pytest tests/test_llm.py
```

### Run specific test

```bash
pytest tests/test_llm.py::TestAskLLMInitialization::test_init_with_openai_model
```

## Test Structure

- `test_exceptions.py` - Tests for exception classes
- `test_llm.py` - Tests for the main AskLLM class
- `test_providers_openai.py` - Tests for OpenAI provider
- `test_providers_google.py` - Tests for Google provider
- `conftest.py` - Shared fixtures and pytest configuration

## Test Coverage

The tests cover:
- Exception handling and error messages
- Initialization and configuration validation
- Provider selection (OpenAI vs Google)
- Parameter validation (temperature, top_p, max_tokens, etc.)
- API call mocking and response handling
- Error propagation and wrapping

