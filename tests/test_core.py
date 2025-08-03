"""Tests for core functionality and utilities."""

import pytest
import time
from datetime import datetime
from unittest.mock import Mock

from sql_synthesizer.core import (
    SystemInfo, QueryMetadata, ResultFormatter, QueryTracker,
    TraceIDGenerator, ErrorHandler, get_system_info, create_query_metadata
)
from sql_synthesizer.types import QueryResult


class TestResultFormatter:
    """Test result formatting functionality."""
    
    def test_to_dict_basic(self):
        """Test basic result conversion to dictionary."""
        result = QueryResult(
            sql="SELECT * FROM users",
            data=[{"id": 1, "name": "John"}],
            explanation="Query retrieves all users",
            columns=["id", "name"]
        )
        
        result_dict = ResultFormatter.to_dict(result)
        
        assert result_dict['sql'] == "SELECT * FROM users"
        assert result_dict['data'] == [{"id": 1, "name": "John"}]
        assert result_dict['explanation'] == "Query retrieves all users"
        assert result_dict['columns'] == ["id", "name"]
        assert result_dict['row_count'] == 1
    
    def test_to_dict_empty_data(self):
        """Test dictionary conversion with empty data."""
        result = QueryResult(
            sql="SELECT * FROM empty_table",
            data=[],
            explanation="No data found",
            columns=["id", "name"]
        )
        
        result_dict = ResultFormatter.to_dict(result)
        assert result_dict['row_count'] == 0
        assert result_dict['data'] == []
    
    def test_to_csv_rows_basic(self):
        """Test CSV row conversion."""
        result = QueryResult(
            sql="SELECT * FROM users",
            data=[
                {"id": 1, "name": "John", "email": "john@example.com"},
                {"id": 2, "name": "Jane", "email": "jane@example.com"}
            ],
            explanation="",
            columns=["id", "name", "email"]
        )
        
        csv_rows = ResultFormatter.to_csv_rows(result)
        
        assert len(csv_rows) == 3  # Header + 2 data rows
        assert csv_rows[0] == ["id", "name", "email"]
        assert csv_rows[1] == ["1", "John", "john@example.com"]
        assert csv_rows[2] == ["2", "Jane", "jane@example.com"]
    
    def test_to_csv_rows_with_commas_and_quotes(self):
        """Test CSV row conversion with special characters."""
        result = QueryResult(
            sql="SELECT * FROM users",
            data=[{"name": 'John "Johnny" Doe', "company": "Acme, Inc."}],
            explanation="",
            columns=["name", "company"]
        )
        
        csv_rows = ResultFormatter.to_csv_rows(result)
        
        assert csv_rows[1] == ['"John ""Johnny"" Doe"', '"Acme, Inc."']
    
    def test_format_explanation(self):
        """Test explanation formatting."""
        formatted = ResultFormatter.format_explanation(
            "SELECT * FROM users", 
            "This query gets all users"
        )
        
        assert "SQL Query:" in formatted
        assert "SELECT * FROM users" in formatted
        assert "Explanation:" in formatted
        assert "This query gets all users" in formatted


class TestQueryTracker:
    """Test query execution tracking."""
    
    def test_initial_state(self):
        """Test tracker initial state."""
        tracker = QueryTracker()
        stats = tracker.get_statistics()
        
        assert stats['total_queries'] == 0
        assert stats['successful_queries'] == 0
        assert stats['failed_queries'] == 0
        assert stats['cache_hits'] == 0
        assert stats['cache_hit_rate'] == 0.0
        assert stats['success_rate'] == 0.0
    
    def test_record_successful_query(self):
        """Test recording successful queries."""
        tracker = QueryTracker()
        
        tracker.record_query(100.5, success=True, cache_hit=False)
        tracker.record_query(200.0, success=True, cache_hit=True)
        
        stats = tracker.get_statistics()
        assert stats['total_queries'] == 2
        assert stats['successful_queries'] == 2
        assert stats['failed_queries'] == 0
        assert stats['cache_hits'] == 1
        assert stats['cache_hit_rate'] == 50.0
        assert stats['success_rate'] == 100.0
        assert stats['average_duration_ms'] == 150.25
    
    def test_record_failed_query(self):
        """Test recording failed queries."""
        tracker = QueryTracker()
        
        tracker.record_query(50.0, success=False)
        tracker.record_query(100.0, success=True)
        
        stats = tracker.get_statistics()
        assert stats['total_queries'] == 2
        assert stats['successful_queries'] == 1
        assert stats['failed_queries'] == 1
        assert stats['success_rate'] == 50.0
    
    def test_reset_statistics(self):
        """Test statistics reset."""
        tracker = QueryTracker()
        
        tracker.record_query(100.0, success=True)
        tracker.reset_statistics()
        
        stats = tracker.get_statistics()
        assert stats['total_queries'] == 0
        assert stats['successful_queries'] == 0


class TestTraceIDGenerator:
    """Test trace ID generation."""
    
    def test_generate_basic(self):
        """Test basic trace ID generation."""
        trace_id = TraceIDGenerator.generate()
        
        assert trace_id.startswith("trace-")
        assert len(trace_id) == 14  # "trace-" + 8 hex chars
    
    def test_generate_with_prefix(self):
        """Test trace ID generation with custom prefix."""
        trace_id = TraceIDGenerator.generate("custom")
        
        assert trace_id.startswith("custom-")
        assert len(trace_id) == 15  # "custom-" + 8 hex chars
    
    def test_generate_query_id(self):
        """Test query ID generation."""
        query_id = TraceIDGenerator.generate_query_id()
        
        assert query_id.startswith("query-")
        assert len(query_id) == 18  # "query-" + 12 hex chars
    
    def test_uniqueness(self):
        """Test that generated IDs are unique."""
        ids = [TraceIDGenerator.generate() for _ in range(100)]
        assert len(set(ids)) == 100  # All unique


class TestErrorHandler:
    """Test error handling and formatting."""
    
    def test_format_database_error_connection(self):
        """Test database connection error formatting."""
        error = Exception("Connection refused to database")
        formatted = ErrorHandler.format_database_error(error)
        
        assert "connection" in formatted.lower()
        assert "check your connection settings" in formatted
    
    def test_format_database_error_timeout(self):
        """Test database timeout error formatting."""
        error = Exception("Query timeout after 30 seconds")
        formatted = ErrorHandler.format_database_error(error)
        
        assert "timeout" in formatted.lower()
        assert "simpler query" in formatted
    
    def test_format_database_error_permission(self):
        """Test database permission error formatting."""
        error = Exception("Access denied for user")
        formatted = ErrorHandler.format_database_error(error)
        
        assert "access" in formatted.lower()
        assert "permissions" in formatted
    
    def test_format_database_error_syntax(self):
        """Test database syntax error formatting."""
        error = Exception("Syntax error near SELECT")
        formatted = ErrorHandler.format_database_error(error)
        
        assert "syntax" in formatted.lower()
        assert "rephrase" in formatted
    
    def test_format_llm_error_timeout(self):
        """Test LLM timeout error formatting."""
        error = Exception("Request timeout after 30 seconds")
        formatted = ErrorHandler.format_llm_error(error)
        
        assert "timeout" in formatted.lower()
        assert "try again" in formatted
    
    def test_format_llm_error_authentication(self):
        """Test LLM authentication error formatting."""
        error = Exception("Invalid API key provided")
        formatted = ErrorHandler.format_llm_error(error)
        
        assert "authentication" in formatted.lower()
        assert "configuration" in formatted
    
    def test_format_llm_error_rate_limit(self):
        """Test LLM rate limit error formatting."""
        error = Exception("Rate limit exceeded for requests")
        formatted = ErrorHandler.format_llm_error(error)
        
        assert "rate limit" in formatted.lower()
        assert "try again later" in formatted


class TestSystemInfo:
    """Test system information functionality."""
    
    def test_get_system_info(self):
        """Test system info retrieval."""
        # Record some activity in global tracker
        from sql_synthesizer.core import query_tracker
        query_tracker.record_query(100.0, success=True, cache_hit=True)
        
        info = get_system_info()
        
        assert isinstance(info, SystemInfo)
        assert info.version == "0.2.2"
        assert info.uptime > 0
        assert info.total_queries >= 1
        assert info.successful_queries >= 1


class TestQueryMetadata:
    """Test query metadata creation."""
    
    def test_create_query_metadata(self):
        """Test query metadata creation."""
        metadata = create_query_metadata(
            query_id="test-123",
            duration_ms=150.5,
            cache_hit=True,
            user_agent="TestAgent/1.0",
            client_ip="192.168.1.1"
        )
        
        assert isinstance(metadata, QueryMetadata)
        assert metadata.query_id == "test-123"
        assert metadata.duration_ms == 150.5
        assert metadata.cache_hit is True
        assert metadata.user_agent == "TestAgent/1.0"
        assert metadata.client_ip == "192.168.1.1"
        assert isinstance(metadata.timestamp, datetime)
    
    def test_create_query_metadata_minimal(self):
        """Test query metadata creation with minimal parameters."""
        metadata = create_query_metadata("test-456", 75.0)
        
        assert metadata.query_id == "test-456"
        assert metadata.duration_ms == 75.0
        assert metadata.cache_hit is False
        assert metadata.user_agent is None
        assert metadata.client_ip is None