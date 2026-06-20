"""Tests for the LLM client module."""

import os
import tempfile
from unittest.mock import MagicMock, patch

from underwriting.llm.llm_client import FALLBACK_MESSAGE, LLMClient


class TestLLMClientInit:
    """Test cases for LLM client initialization."""

    def test_init_loads_config_from_default_path(self) -> None:
        """Test that LLMClient loads config from ./config.yaml by default."""
        client = LLMClient()
        assert client.base_url == "http://192.168.1.59:1234/v1"
        assert client.model == "qwen/qwen3.6-35b-a3b"
        assert client.context_window == 200000
        assert client.wait_time == 3600
        assert client.temperature == 0.1
        assert client.max_tokens == 100000

    def test_init_loads_config_from_custom_path(self) -> None:
        """Test that LLMClient loads config from a custom YAML path."""
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False
        ) as tmp:
            tmp.write(
                "llm:\n"
                "  baseURL: http://localhost:8000/v1\n"
                "  model: test-model\n"
                "  context_window: 10000\n"
                "  wait_time: 60\n"
                "  temperature: 0.5\n"
                "  max_tokens: 500\n"
            )
            tmp_path = tmp.name

        try:
            client = LLMClient(config_path=tmp_path)
            assert client.base_url == "http://localhost:8000/v1"
            assert client.model == "test-model"
            assert client.context_window == 10000
            assert client.wait_time == 60
            assert client.temperature == 0.5
            assert client.max_tokens == 500
        finally:
            os.unlink(tmp_path)

    def test_init_handles_empty_llm_section(self) -> None:
        """Test that LLMClient handles an empty llm config section."""
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False
        ) as tmp:
            tmp.write("llm:\n")
            tmp_path = tmp.name

        try:
            client = LLMClient(config_path=tmp_path)
            assert client.base_url == ""
            assert client.model == ""
            assert client.context_window == 200000
            assert client.wait_time == 3600
            assert client.temperature == 0.1
            assert client.max_tokens == 2000
        finally:
            os.unlink(tmp_path)

    def test_init_handles_missing_llm_section(self) -> None:
        """Test that LLMClient handles a config with no llm section."""
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False
        ) as tmp:
            tmp.write("paths:\n  chroma_persist_dir: ./db\n")
            tmp_path = tmp.name

        try:
            client = LLMClient(config_path=tmp_path)
            assert client.base_url == ""
            assert client.model == ""
        finally:
            os.unlink(tmp_path)


class TestLLMClientGenerate:
    """Test cases for the generate method."""

    def test_generate_returns_fallback_when_not_configured(self) -> None:
        """Test that generate returns fallback when base_url is empty."""
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False
        ) as tmp:
            tmp.write("llm:\n  model: test-model\n")
            tmp_path = tmp.name

        try:
            client = LLMClient(config_path=tmp_path)
            result = client.generate("test prompt")
            assert result == FALLBACK_MESSAGE
        finally:
            os.unlink(tmp_path)

    def test_generate_returns_fallback_on_api_failure(self) -> None:
        """Test that generate returns fallback when the API call fails."""
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False
        ) as tmp:
            tmp.write(
                "llm:\n"
                "  baseURL: http://localhost:9999/v1\n"
                "  model: test-model\n"
                "  max_tokens: 100\n"
                "  temperature: 0.2\n"
            )
            tmp_path = tmp.name

        try:
            client = LLMClient(config_path=tmp_path)

            with patch(
                "openai.OpenAI"
            ) as mock_openai:
                mock_openai.side_effect = Exception("Connection refused")
                result = client.generate("test prompt")
                assert result == FALLBACK_MESSAGE
        finally:
            os.unlink(tmp_path)

    def test_generate_returns_fallback_on_no_choices(self) -> None:
        """Test that generate returns fallback when response has no choices."""
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False
        ) as tmp:
            tmp.write(
                "llm:\n"
                "  baseURL: http://localhost:9999/v1\n"
                "  model: test-model\n"
            )
            tmp_path = tmp.name

        try:
            client = LLMClient(config_path=tmp_path)

            mock_response = MagicMock()
            mock_response.choices = []

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response

            with patch(
                "openai.OpenAI",
                return_value=mock_client,
            ):
                result = client.generate("test prompt")
                assert result == FALLBACK_MESSAGE
        finally:
            os.unlink(tmp_path)

    def test_generate_returns_response_text_on_success(self) -> None:
        """Test that generate returns the response text on success."""
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False
        ) as tmp:
            tmp.write(
                "llm:\n"
                "  baseURL: http://localhost:9999/v1\n"
                "  model: test-model\n"
            )
            tmp_path = tmp.name

        try:
            client = LLMClient(config_path=tmp_path)

            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "Hello from LLM!"
            mock_response.choices = [mock_choice]

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response

            with patch(
                "openai.OpenAI",
                return_value=mock_client,
            ):
                result = client.generate("test prompt")
                assert result == "Hello from LLM!"
        finally:
            os.unlink(tmp_path)

    def test_generate_uses_custom_max_tokens(self) -> None:
        """Test that generate passes custom max_tokens to the API."""
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False
        ) as tmp:
            tmp.write(
                "llm:\n"
                "  baseURL: http://localhost:9999/v1\n"
                "  model: test-model\n"
            )
            tmp_path = tmp.name

        try:
            client = LLMClient(config_path=tmp_path)

            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "OK"
            mock_response.choices = [mock_choice]

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response

            with patch(
                "openai.OpenAI",
                return_value=mock_client,
            ):
                client.generate("test prompt", max_tokens=42)
                mock_client.chat.completions.create.assert_called_once()
                call_kwargs = (
                    mock_client.chat.completions.create.call_args
                )
                assert call_kwargs.kwargs["max_tokens"] == 42
        finally:
            os.unlink(tmp_path)

    def test_generate_uses_custom_temperature(self) -> None:
        """Test that generate passes custom temperature to the API."""
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False
        ) as tmp:
            tmp.write(
                "llm:\n"
                "  baseURL: http://localhost:9999/v1\n"
                "  model: test-model\n"
            )
            tmp_path = tmp.name

        try:
            client = LLMClient(config_path=tmp_path)

            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "OK"
            mock_response.choices = [mock_choice]

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response

            with patch(
                "openai.OpenAI",
                return_value=mock_client,
            ):
                client.generate("test prompt", temperature=0.7)
                mock_client.chat.completions.create.assert_called_once()
                call_kwargs = (
                    mock_client.chat.completions.create.call_args
                )
                assert call_kwargs.kwargs["temperature"] == 0.7
        finally:
            os.unlink(tmp_path)


class TestLLMClientAvailability:
    """Test cases for the is_available method."""

    def test_is_available_returns_true_when_configured(self) -> None:
        """Test is_available returns True when base_url and model are set."""
        client = LLMClient()
        assert client.is_available() is True

    def test_is_available_returns_false_when_not_configured(self) -> None:
        """Test is_available returns False when config is missing."""
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False
        ) as tmp:
            tmp.write("llm:\n")
            tmp_path = tmp.name

        try:
            client = LLMClient(config_path=tmp_path)
            assert client.is_available() is False
        finally:
            os.unlink(tmp_path)
