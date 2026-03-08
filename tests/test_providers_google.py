"""Tests for Google provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ask_llm.exceptions import ConfigurationError
from ask_llm.providers.google import GoogleTextClient


class TestGoogleTextClientInitialization:
    """Tests for GoogleTextClient initialization."""

    def test_init_without_api_key(self):
        """Test that missing API key raises ConfigurationError."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError, match="GOOGLE_API_KEY"):
                GoogleTextClient()

    def test_init_with_api_key(self):
        """Test successful initialization with API key."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            with patch("ask_llm.providers.google.text_client.genai.Client"):
                client = GoogleTextClient()
                assert client._cached_contexts is not None

    def test_init_with_missing_google_package(self):
        """Test that missing Google package raises ConfigurationError."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            with patch(
                "ask_llm.providers.google.text_client.genai.Client",
                side_effect=ImportError("No module named 'google.genai'"),
            ):
                with pytest.raises(ConfigurationError, match="Google GenAI package not installed"):
                    GoogleTextClient()


class TestGoogleTextClientBuildConfigDict:
    """Tests for _build_config_dict method."""

    def test_build_config_basic(self):
        """Test building basic config dict."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            with patch("ask_llm.providers.google.text_client.genai.Client"):
                client = GoogleTextClient()
                config = client._build_config_dict()
                assert "tools" in config
                assert config["tools"] == [{"google_search": {}}]

    def test_build_config_with_max_tokens(self):
        """Test building config with max_tokens."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            with patch("ask_llm.providers.google.text_client.genai.Client"):
                client = GoogleTextClient()
                config = client._build_config_dict(max_tokens=100)
                assert config["max_output_tokens"] == 100

    def test_build_config_with_temperature(self):
        """Test building config with temperature."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            with patch("ask_llm.providers.google.text_client.genai.Client"):
                client = GoogleTextClient()
                config = client._build_config_dict(temperature=0.7)
                assert config["temperature"] == 0.7

    def test_build_config_with_top_p(self):
        """Test building config with top_p."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            with patch("ask_llm.providers.google.text_client.genai.Client"):
                client = GoogleTextClient()
                config = client._build_config_dict(top_p=0.9)
                assert config["top_p"] == 0.9

    def test_build_config_with_json_schema(self):
        """Test building config with JSON schema."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            with patch("ask_llm.providers.google.text_client.genai.Client"):
                client = GoogleTextClient()
                response_format = {
                    "type": "json_schema",
                    "json_schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                }
                config = client._build_config_dict(response_format=response_format)
                assert config["response_mime_type"] == "application/json"
                assert "response_json_schema" in config
                assert "tools" not in config  # Tools should not be included for JSON responses

    def test_build_config_without_tools_for_json(self):
        """Test that tools are not included for JSON schema responses."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            with patch("ask_llm.providers.google.text_client.genai.Client"):
                client = GoogleTextClient()
                response_format = {
                    "type": "json_schema",
                    "json_schema": {"type": "object"},
                }
                config = client._build_config_dict(response_format=response_format)
                assert "tools" not in config


class TestGoogleTextClientGetSystemPromptHash:
    """Tests for _get_system_prompt_hash method."""

    def test_hash_generation(self):
        """Test hash generation for system prompt."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            with patch("ask_llm.providers.google.text_client.genai.Client"):
                client = GoogleTextClient()
                hash1 = client._get_system_prompt_hash("test prompt")
                hash2 = client._get_system_prompt_hash("test prompt")
                assert hash1 == hash2  # Same input should produce same hash

                hash3 = client._get_system_prompt_hash("different prompt")
                assert hash1 != hash3  # Different input should produce different hash


class TestGoogleTextClientGenerate:
    """Tests for generate method."""

    @pytest.mark.asyncio
    async def test_generate_basic(self):
        """Test basic generation."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            with patch("ask_llm.providers.google.text_client.genai.Client") as mock_genai:
                mock_client_instance = MagicMock()
                mock_aio = MagicMock()
                mock_models = MagicMock()
                mock_response = MagicMock()
                mock_response.text = "Test response"

                async def mock_generate(*args, **kwargs):
                    return mock_response

                mock_models.generate_content = mock_generate
                mock_aio.models = mock_models
                mock_client_instance.aio = mock_aio
                mock_genai.return_value = mock_client_instance

                client = GoogleTextClient()
                result = await client.generate(
                    prompt="What is Python?", model="gemini-2.0-flash-exp"
                )
                assert result == "Test response"

    @pytest.mark.asyncio
    async def test_generate_with_system_instruct(self):
        """Test generation with system instruction."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            with patch("ask_llm.providers.google.text_client.genai.Client") as mock_genai:
                mock_client_instance = MagicMock()
                mock_aio = MagicMock()
                mock_models = MagicMock()
                mock_response = MagicMock()
                mock_response.text = "Test response"

                async def mock_generate(*args, **kwargs):
                    return mock_response

                mock_models.generate_content = mock_generate
                mock_aio.models = mock_models
                mock_client_instance.aio = mock_aio
                mock_genai.return_value = mock_client_instance

                client = GoogleTextClient()
                result = await client.generate(
                    prompt="What is Python?",
                    model="gemini-2.0-flash-exp",
                    system_instruct="You are a helpful assistant.",
                )
                assert result == "Test response"

    @pytest.mark.asyncio
    async def test_generate_with_max_tokens(self):
        """Test generation with max_tokens."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            with patch("ask_llm.providers.google.text_client.genai.Client") as mock_genai:
                mock_client_instance = MagicMock()
                mock_aio = MagicMock()
                mock_models = MagicMock()
                mock_response = MagicMock()
                mock_response.text = "Test response"

                async def mock_generate(*args, **kwargs):
                    return mock_response

                mock_models.generate_content = mock_generate
                mock_aio.models = mock_models
                mock_client_instance.aio = mock_aio
                mock_genai.return_value = mock_client_instance

                client = GoogleTextClient()
                result = await client.generate(
                    prompt="What is Python?", model="gemini-2.0-flash-exp", max_tokens=100
                )
                assert result == "Test response"

