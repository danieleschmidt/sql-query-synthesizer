"""Tests for LLM interface and OpenAI adapter implementation."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from sql_synthesizer.llm_interface import (
    LLMProvider,
    ProviderAuthenticationError,
    ProviderError,
    ProviderTimeoutError,
)
from sql_synthesizer.openai_adapter import OpenAIAdapter


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, provider_name="mock", model_name="mock-model"):
        self.provider_name = provider_name
        self.model_name = model_name
        self.responses = {}

    def generate_sql(self, question, available_tables=None, **kwargs):
        """TODO: Add docstring"""
        return self.responses.get(question, "SELECT 1")

     """TODO: Add docstring"""
    def get_provider_name(self):
        return self.provider_name
            """TODO: Add docstring"""

    def get_model_name(self):
        return self.model_name


class TestLLMInterface:
    """Test the abstract LLM interface."""

    def test_mock_provider_implementation(self):
        """Test that mock provider implements interface correctly."""
        provider = MockLLMProvider("test", "test-model")

        assert provider.get_provider_name() == "test"
        assert provider.get_model_name() == "test-model"
        assert provider.validate_configuration() is True
        assert provider.get_capabilities() == {}

        # Test SQL generation
        result = provider.generate_sql("test question", ["users"])
        assert result == "SELECT 1"

    def test_provider_error_creation(self):
        """Test provider error exception creation."""
        error = ProviderError("Test error", "test-provider", "TEST_001")

        assert str(error) == "Test error"
        assert error.provider == "test-provider"
        assert error.error_code == "TEST_001"

    def test_provider_timeout_error(self):
        """Test provider timeout error creation."""
        error = ProviderTimeoutError("Timeout", "test-provider")

        assert str(error) == "Timeout"
        assert error.provider == "test-provider"
        assert error.error_code is None

    def test_provider_authentication_error(self):
        """Test provider authentication error creation."""
        error = ProviderAuthenticationError("Auth failed", "test-provider", "AUTH_001")

        assert str(error) == "Auth failed"
        assert error.provider == "test-provider"
        assert error.error_code == "AUTH_001"


class TestOpenAIAdapterInterface:
    """Test OpenAI adapter implementation of LLM interface."""

    def test_openai_adapter_implements_interface(self):
        """Test that OpenAI adapter properly implements LLM interface."""
        with patch("sql_synthesizer.openai_adapter.openai") as mock_openai:
            adapter = OpenAIAdapter("test-key", "gpt-4", 30.0)

            # Test interface methods
            assert adapter.get_provider_name() == "openai"
            assert adapter.get_model_name() == "gpt-4"
            assert adapter.validate_configuration() is True

            capabilities = adapter.get_capabilities()
            assert capabilities["model"] == "gpt-4"
            assert capabilities["timeout"] == 30.0
            assert "max_tokens" in capabilities

    def test_openai_adapter_without_package(self):
        """Test OpenAI adapter when package is not available."""
        with patch("sql_synthesizer.openai_adapter.openai", None):
            with pytest.raises(Exception):  # Should raise openai package missing error
                OpenAIAdapter("test-key")

    def test_openai_adapter_validation_without_key(self):
        """Test configuration validation without API key."""
        with patch("sql_synthesizer.openai_adapter.openai") as mock_openai:
            adapter = OpenAIAdapter("", "gpt-3.5-turbo")
            assert adapter.validate_configuration() is False

    def test_openai_adapter_validation_without_package(self):
        """Test configuration validation without openai package."""
        with patch("sql_synthesizer.openai_adapter.openai", None):
            try:
                adapter = OpenAIAdapter("test-key")
                assert False, "Should have raised exception"
            except Exception:
                pass  # Expected

    @patch("sql_synthesizer.openai_adapter.openai")
    def test_openai_adapter_generate_sql_success(self, mock_openai):
        """Test successful SQL generation with OpenAI."""
        # Mock the response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "SELECT * FROM users"
        mock_openai.chat.completions.create.return_value = mock_response

        adapter = OpenAIAdapter("test-key", "gpt-3.5-turbo")
        result = adapter.generate_sql("Show me users", ["users", "orders"])

        assert result == "SELECT * FROM users"
        mock_openai.chat.completions.create.assert_called_once()

        # Verify the call arguments
        call_args = mock_openai.chat.completions.create.call_args
        assert call_args[1]["model"] == "gpt-3.5-turbo"
        assert call_args[1]["temperature"] == 0
        assert "users" in call_args[1]["messages"][0]["content"]

    @patch("sql_synthesizer.openai_adapter.openai")
    def test_openai_adapter_timeout_error(self, mock_openai):
        """Test OpenAI adapter timeout handling."""

        class MockTimeout(Exception):
            pass

        mock_openai.chat.completions.create.side_effect = MockTimeout(
            "Request timed out"
        )

        adapter = OpenAIAdapter("test-key", "gpt-3.5-turbo", timeout=5.0)

        with pytest.raises(ProviderTimeoutError) as exc_info:
            adapter.generate_sql("Show me users", ["users"])

        assert "timed out" in str(exc_info.value)
        assert exc_info.value.provider == "openai"

    @patch("sql_synthesizer.openai_adapter.openai")
    def test_openai_adapter_auth_error(self, mock_openai):
        """Test OpenAI adapter authentication error handling."""

        class MockAuthenticationError(Exception):
            pass

        mock_openai.chat.completions.create.side_effect = MockAuthenticationError(
            "Invalid API key"
        )

        adapter = OpenAIAdapter("test-key", "gpt-3.5-turbo")

        with pytest.raises(ProviderAuthenticationError) as exc_info:
            adapter.generate_sql("Show me users", ["users"])

        assert "authentication failed" in str(exc_info.value)
        assert exc_info.value.provider == "openai"

    @patch("sql_synthesizer.openai_adapter.openai")
    def test_openai_adapter_generic_error(self, mock_openai):
        """Test OpenAI adapter generic error handling."""
        mock_openai.chat.completions.create.side_effect = RuntimeError("API Error")

        adapter = OpenAIAdapter("test-key", "gpt-3.5-turbo")

        with pytest.raises(ProviderError) as exc_info:
            adapter.generate_sql("Show me users", ["users"])

        assert "request failed" in str(exc_info.value)
        assert exc_info.value.provider == "openai"

    @patch("sql_synthesizer.openai_adapter.openai")
    def test_openai_adapter_capabilities_gpt35(self, mock_openai):
        """Test OpenAI adapter capabilities for GPT-3.5."""
        adapter = OpenAIAdapter("test-key", "gpt-3.5-turbo")
        capabilities = adapter.get_capabilities()

        assert capabilities["max_tokens"] == 4096
        assert capabilities["supports_json_mode"] is True
        assert capabilities["supports_function_calling"] is True
        assert capabilities["temperature_range"] == (0.0, 2.0)

    @patch("sql_synthesizer.openai_adapter.openai")
    def test_openai_adapter_capabilities_gpt4(self, mock_openai):
        """Test OpenAI adapter capabilities for GPT-4."""
        adapter = OpenAIAdapter("test-key", "gpt-4")
        capabilities = adapter.get_capabilities()

        assert capabilities["max_tokens"] == 8192
        assert capabilities["model"] == "gpt-4"

    @patch("sql_synthesizer.openai_adapter.openai")
    def test_openai_adapter_empty_question(self, mock_openai):
        """Test OpenAI adapter with empty question."""
        adapter = OpenAIAdapter("test-key", "gpt-3.5-turbo")

        with pytest.raises(Exception):  # Should raise empty question error
            adapter.generate_sql("", ["users"])

        with pytest.raises(Exception):  # Should raise empty question error
            adapter.generate_sql("   ", ["users"])

    @patch("sql_synthesizer.openai_adapter.openai")
    def test_openai_adapter_with_kwargs(self, mock_openai):
        """Test OpenAI adapter with additional kwargs."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "SELECT * FROM users"
        mock_openai.chat.completions.create.return_value = mock_response

        adapter = OpenAIAdapter("test-key", "gpt-3.5-turbo")
        result = adapter.generate_sql("Show me users", ["users"], custom_param="test")

        assert result == "SELECT * FROM users"
        # The kwargs are currently ignored but should not cause errors
