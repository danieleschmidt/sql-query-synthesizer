"""Tests for async OpenAI adapter functionality."""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from sql_synthesizer.async_openai_adapter import AsyncOpenAIAdapter
from sql_synthesizer.llm_interface import ProviderError, ProviderTimeoutError, ProviderAuthenticationError


class MockAsyncOpenAI:
    """Mock async OpenAI client."""
    def __init__(self):
        self.chat = Mock()
        self.chat.completions = Mock()
        self.chat.completions.create = AsyncMock()


class TestAsyncOpenAIAdapter:
    """Test async OpenAI adapter functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.api_key = "test-api-key"
        self.model = "gpt-3.5-turbo"
        
        # Mock the async OpenAI client
        self.mock_openai = MockAsyncOpenAI()
        
        # Create adapter with mocked client
        with patch('sql_synthesizer.async_openai_adapter.AsyncOpenAI') as mock_client_class:
            mock_client_class.return_value = self.mock_openai
            self.adapter = AsyncOpenAIAdapter(
                api_key=self.api_key,
                model=self.model,
                timeout=30.0
            )

    @pytest.mark.asyncio
    async def test_async_generate_sql_success(self):
        """Test successful async SQL generation."""
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "SELECT * FROM users WHERE age > 25"
        
        self.mock_openai.chat.completions.create.return_value = mock_response
        
        # Generate SQL
        result = await self.adapter.generate_sql(
            "Show me users older than 25",
            available_tables=["users", "orders"]
        )
        
        # Verify result
        assert result == "SELECT * FROM users WHERE age > 25"
        
        # Verify OpenAI was called with correct parameters
        self.mock_openai.chat.completions.create.assert_called_once()
        call_args = self.mock_openai.chat.completions.create.call_args
        
        assert call_args.kwargs["model"] == self.model
        assert call_args.kwargs["temperature"] == 0
        assert call_args.kwargs["timeout"] == 30.0
        assert len(call_args.kwargs["messages"]) == 1
        assert "Show me users older than 25" in call_args.kwargs["messages"][0]["content"]
        assert "users, orders" in call_args.kwargs["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_async_generate_sql_with_table_context(self):
        """Test async SQL generation with table context."""
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "SELECT COUNT(*) FROM products"
        
        self.mock_openai.chat.completions.create.return_value = mock_response
        
        # Generate SQL with specific tables
        result = await self.adapter.generate_sql(
            "How many products do we have?",
            available_tables=["products", "categories", "suppliers"]
        )
        
        # Verify result
        assert result == "SELECT COUNT(*) FROM products"
        
        # Verify table context was included in prompt
        call_args = self.mock_openai.chat.completions.create.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "products, categories, suppliers" in prompt

    @pytest.mark.asyncio
    async def test_async_generate_sql_without_tables(self):
        """Test async SQL generation without table context."""
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "SELECT 1"
        
        self.mock_openai.chat.completions.create.return_value = mock_response
        
        # Generate SQL without table context
        result = await self.adapter.generate_sql("Show me something")
        
        # Verify result
        assert result == "SELECT 1"
        
        # Verify no table context in prompt
        call_args = self.mock_openai.chat.completions.create.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "Available tables:" not in prompt

    @pytest.mark.asyncio
    async def test_async_generate_sql_empty_question(self):
        """Test async SQL generation with empty question."""
        # Test empty question - UserFriendlyError is raised, not ProviderError
        with pytest.raises(Exception, match="Please provide a question"):
            await self.adapter.generate_sql("")
        
        # Test whitespace-only question
        with pytest.raises(Exception, match="Please provide a question"):
            await self.adapter.generate_sql("   ")

    @pytest.mark.asyncio
    async def test_async_generate_sql_timeout_error(self):
        """Test async timeout error handling."""
        # Setup timeout exception
        class MockTimeoutError(Exception):
            pass
        MockTimeoutError.__name__ = "TimeoutError"
        
        self.mock_openai.chat.completions.create.side_effect = MockTimeoutError("Request timed out")
        
        # Verify timeout error is properly handled
        with pytest.raises(ProviderTimeoutError, match="OpenAI request timed out"):
            await self.adapter.generate_sql("Test question")
        
        # Verify circuit breaker recorded failure
        assert self.adapter.circuit_breaker.failure_count == 1

    @pytest.mark.asyncio
    async def test_async_generate_sql_auth_error(self):
        """Test async authentication error handling."""
        # Setup auth exception
        class MockAuthError(Exception):
            pass
        MockAuthError.__name__ = "AuthenticationError"
        
        self.mock_openai.chat.completions.create.side_effect = MockAuthError("Invalid API key")
        
        # Verify auth error is properly handled
        with pytest.raises(ProviderAuthenticationError, match="OpenAI authentication failed"):
            await self.adapter.generate_sql("Test question")
        
        # Verify circuit breaker recorded failure
        assert self.adapter.circuit_breaker.failure_count == 1

    @pytest.mark.asyncio
    async def test_async_generate_sql_generic_error(self):
        """Test async generic error handling."""
        # Setup generic exception
        self.mock_openai.chat.completions.create.side_effect = RuntimeError("API Error")
        
        # Verify generic error is properly handled
        with pytest.raises(ProviderError, match="OpenAI request failed"):
            await self.adapter.generate_sql("Test question")
        
        # Verify circuit breaker recorded failure
        assert self.adapter.circuit_breaker.failure_count == 1

    @pytest.mark.asyncio
    async def test_async_circuit_breaker_integration(self):
        """Test async circuit breaker integration."""
        # Setup repeated failures to trip circuit breaker
        self.mock_openai.chat.completions.create.side_effect = RuntimeError("API Error")
        
        # Make requests until circuit breaker opens
        for i in range(5):  # Default failure threshold is 5
            with pytest.raises(ProviderError):
                await self.adapter.generate_sql(f"Test question {i}")
        
        # Verify circuit breaker is now open
        assert not self.adapter.circuit_breaker.is_request_allowed()
        
        # Next request should be blocked by circuit breaker
        with pytest.raises(ProviderError, match="temporarily unavailable"):
            await self.adapter.generate_sql("Test question after circuit open")
        
        # Verify OpenAI was not called for the blocked request
        assert self.mock_openai.chat.completions.create.call_count == 5

    @pytest.mark.asyncio
    async def test_async_concurrent_requests(self):
        """Test concurrent async requests."""
        # Setup mock responses for concurrent requests
        responses = []
        for i in range(3):
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = f"SELECT * FROM table{i}"
            responses.append(mock_response)
        
        self.mock_openai.chat.completions.create.side_effect = responses
        
        # Make concurrent requests
        tasks = [
            self.adapter.generate_sql(f"Query {i}", available_tables=[f"table{i}"])
            for i in range(3)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all requests completed
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result == f"SELECT * FROM table{i}"
        
        # Verify all OpenAI calls were made
        assert self.mock_openai.chat.completions.create.call_count == 3

    @pytest.mark.asyncio
    async def test_async_provider_info_methods(self):
        """Test async adapter provider information methods."""
        # Test provider name
        assert self.adapter.get_provider_name() == "openai"
        
        # Test model name
        assert self.adapter.get_model_name() == self.model

    @pytest.mark.asyncio
    async def test_async_custom_timeout_configuration(self):
        """Test async adapter with custom timeout."""
        # Create adapter with custom timeout
        with patch('sql_synthesizer.async_openai_adapter.AsyncOpenAI') as mock_client_class:
            mock_client_class.return_value = self.mock_openai
            custom_adapter = AsyncOpenAIAdapter(
                api_key=self.api_key,
                model=self.model,
                timeout=60.0
            )
        
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "SELECT 1"
        
        self.mock_openai.chat.completions.create.return_value = mock_response
        
        # Generate SQL
        await custom_adapter.generate_sql("Test question")
        
        # Verify custom timeout was used
        call_args = self.mock_openai.chat.completions.create.call_args
        assert call_args.kwargs["timeout"] == 60.0

    @pytest.mark.asyncio
    async def test_async_circuit_breaker_recovery(self):
        """Test async circuit breaker recovery."""
        # Trip the circuit breaker
        self.mock_openai.chat.completions.create.side_effect = RuntimeError("API Error")
        
        for i in range(5):  # Trip circuit breaker
            with pytest.raises(ProviderError):
                await self.adapter.generate_sql(f"Test {i}")
        
        # Verify circuit is open
        assert not self.adapter.circuit_breaker.is_request_allowed()
        
        # Wait for recovery timeout (simulate with manual state change)
        self.adapter.circuit_breaker._state = "HALF_OPEN"
        
        # Setup successful response for recovery
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "SELECT 1"
        
        self.mock_openai.chat.completions.create.side_effect = None
        self.mock_openai.chat.completions.create.return_value = mock_response
        
        # Successful request should close circuit
        result = await self.adapter.generate_sql("Recovery test")
        
        # Verify success
        assert result == "SELECT 1"
        assert self.adapter.circuit_breaker.is_request_allowed()