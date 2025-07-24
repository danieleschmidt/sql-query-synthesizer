"""Tests for specific exception handling improvements.

This test module ensures that broad 'except Exception' clauses
are replaced with specific exception types for better error
diagnosis and handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.exc import SQLAlchemyError, OperationalError, TimeoutError as SQLTimeoutError
import openai

from sql_synthesizer.query_agent import QueryAgent
from sql_synthesizer import webapp
from sql_synthesizer.openai_adapter import OpenAIAdapter
from sql_synthesizer.async_openai_adapter import AsyncOpenAIAdapter
from sql_synthesizer.async_query_agent import AsyncQueryAgent
from sql_synthesizer.services.async_sql_generator_service import AsyncSQLGeneratorService
from sql_synthesizer.services.enhanced_query_validator import EnhancedQueryValidator


class TestSpecificExceptionHandling:
    """Test that specific exception types are caught instead of broad Exception."""

    def test_query_agent_cache_cleanup_handles_specific_exceptions(self):
        """Test that cache cleanup handles specific exceptions."""
        with patch('sql_synthesizer.query_agent.QueryAgent.__init__', return_value=None):
            agent = QueryAgent.__new__(QueryAgent)
            agent.cache = Mock()
            
            # Test CacheError handling
            from sql_synthesizer.cache import CacheError
            agent.cache.cleanup_expired.side_effect = CacheError("Cache backend unavailable")
            
            with patch('sql_synthesizer.query_agent.logger') as mock_logger:
                agent._perform_cache_cleanup()
                mock_logger.error.assert_called_once()
                assert "cache cleanup" in mock_logger.error.call_args[0][0].lower()

    def test_query_agent_row_count_handles_database_exceptions(self):
        """Test that row count fetching handles specific database exceptions."""
        with patch('sql_synthesizer.query_agent.QueryAgent.__init__', return_value=None):
            agent = QueryAgent.__new__(QueryAgent)
            agent.db_manager = Mock()
            
            # Mock connection that raises OperationalError
            connection = Mock()
            connection.execute.side_effect = OperationalError("statement", "params", "orig")
            agent.db_manager.get_connection.return_value.__enter__ = Mock(return_value=connection)
            agent.db_manager.get_connection.return_value.__exit__ = Mock(return_value=None)
            
            with patch('sql_synthesizer.query_agent.create_invalid_table_error') as mock_error:
                mock_error.return_value = "Table not found"
                result = agent.get_row_count("nonexistent_table")
                mock_error.assert_called_once_with("nonexistent_table")

    def test_webapp_query_handles_specific_exceptions(self):
        """Test that webapp query handling catches specific exceptions."""
        with patch('sql_synthesizer.webapp.QueryAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent
            
            # Test SQLAlchemy TimeoutError
            mock_agent.query.side_effect = SQLTimeoutError("statement", "params", "orig", "statement_timeout")
            
            with patch('sql_synthesizer.webapp.logger') as mock_logger, \
                 patch('sql_synthesizer.webapp.render_template') as mock_render:
                
                from sql_synthesizer.webapp import app
                with app.test_client() as client:
                    response = client.post('/', data={'question': 'test query'})
                    
                    # Verify timeout error is logged
                    mock_logger.error.assert_called()
                    assert "timeout" in str(mock_logger.error.call_args).lower()

    def test_webapp_api_handles_openai_specific_exceptions(self):
        """Test that API endpoints handle OpenAI-specific exceptions."""
        with patch('sql_synthesizer.webapp.QueryAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent
            
            # Test OpenAI RateLimitError
            mock_agent.query.side_effect = openai.RateLimitError("Rate limit exceeded")
            
            with patch('sql_synthesizer.webapp.logger') as mock_logger:
                from sql_synthesizer.webapp import app
                with app.test_client() as client:
                    response = client.post('/api/query', 
                                         json={'question': 'test query'},
                                         headers={'X-API-Key': 'test-key'})
                    
                    # Verify rate limit error is logged
                    mock_logger.error.assert_called()
                    assert "rate limit" in str(mock_logger.error.call_args).lower() or \
                           "openai" in str(mock_logger.error.call_args).lower()

    def test_openai_adapter_handles_specific_openai_exceptions(self):
        """Test that OpenAI adapter handles specific OpenAI exceptions."""
        adapter = OpenAIAdapter("test-key")
        
        with patch.object(adapter, 'client') as mock_client:
            mock_completion = Mock()
            mock_client.chat.completions.create = mock_completion
            
            # Test AuthenticationError
            mock_completion.side_effect = openai.AuthenticationError("Invalid API key")
            
            with patch('sql_synthesizer.openai_adapter.logger') as mock_logger:
                result = adapter.generate_sql("test prompt", [])
                
                # Should log authentication error specifically
                mock_logger.error.assert_called()
                assert "authentication" in str(mock_logger.error.call_args).lower() or \
                       "api key" in str(mock_logger.error.call_args).lower()

    def test_async_openai_adapter_handles_specific_exceptions(self):
        """Test that async OpenAI adapter handles specific exceptions."""
        adapter = AsyncOpenAIAdapter("test-key")
        
        with patch.object(adapter, 'client') as mock_client:
            mock_completion = Mock()
            mock_client.chat.completions.create = mock_completion
            
            # Test OpenAI Timeout
            mock_completion.side_effect = openai.APITimeoutError("Request timed out")
            
            with patch('sql_synthesizer.async_openai_adapter.logger') as mock_logger:
                import asyncio
                result = asyncio.run(adapter.generate_sql("test prompt", []))
                
                # Should log timeout error specifically
                mock_logger.error.assert_called()
                assert "timeout" in str(mock_logger.error.call_args).lower()

    def test_enhanced_query_validator_handles_encoding_exceptions(self):
        """Test that enhanced validator handles specific encoding exceptions."""
        validator = EnhancedQueryValidator()
        
        # Test with text that causes UnicodeDecodeError
        with patch('urllib.parse.unquote') as mock_unquote:
            mock_unquote.side_effect = UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid start byte')
            
            result = validator._decode_if_encoded("test%20input")
            # Should return original text when decoding fails
            assert result == "test%20input"

    def test_async_sql_generator_handles_circuit_breaker_exceptions(self):
        """Test that async SQL generator handles circuit breaker exceptions."""
        with patch('sql_synthesizer.services.async_sql_generator_service.AsyncSQLGeneratorService.__init__', return_value=None):
            service = AsyncSQLGeneratorService.__new__(AsyncSQLGeneratorService)
            service.llm_provider = Mock()
            
            # Test circuit breaker state check exception
            service.llm_provider.circuit_breaker.is_request_allowed.side_effect = AttributeError("No circuit breaker")
            
            # Should return False when circuit breaker check fails
            result = service._is_llm_available()
            assert result is False

    def test_async_query_agent_connection_cleanup_handles_specific_exceptions(self):
        """Test that async query agent handles specific connection cleanup exceptions."""
        with patch('sql_synthesizer.async_query_agent.AsyncQueryAgent.__init__', return_value=None):
            agent = AsyncQueryAgent.__new__(AsyncQueryAgent)
            agent.engine = Mock()
            
            # Test OperationalError during cleanup
            agent.engine.dispose.side_effect = OperationalError("statement", "params", "orig")
            
            with patch('sql_synthesizer.async_query_agent.logger') as mock_logger:
                import asyncio
                asyncio.run(agent.cleanup())
                
                # Should log specific database error
                mock_logger.error.assert_called()
                assert "database" in str(mock_logger.error.call_args).lower() or \
                       "connection" in str(mock_logger.error.call_args).lower()