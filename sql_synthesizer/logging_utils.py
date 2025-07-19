"""Structured logging utilities with trace ID support."""

import json
import logging
import os
import uuid
from contextlib import contextmanager
from typing import Generator, Any, Dict


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields from the record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                          'pathname', 'filename', 'module', 'lineno', 
                          'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process',
                          'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                log_data[key] = value
        
        # Include exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


@contextmanager
def log_context(trace_id: str = None) -> Generator[str, None, None]:
    """Context manager for trace ID correlation."""
    if trace_id is None:
        trace_id = str(uuid.uuid4())
    
    # Store trace_id in context variable or thread local
    # For simplicity, we'll return it for manual propagation
    yield trace_id


def configure_logging(
    level: str = None,
    format_type: str = None,
    enable_json: bool = False
) -> None:
    """Configure application logging based on environment and parameters."""
    
    # Get configuration from environment variables
    log_level = level or os.getenv("QUERY_AGENT_LOG_LEVEL", "INFO")
    log_format = format_type or os.getenv("QUERY_AGENT_LOG_FORMAT", "standard")
    
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Get or create logger
    logger = logging.getLogger("sql_synthesizer")
    logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(numeric_level)
    
    # Set formatter based on configuration
    if log_format.lower() == "json" or enable_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False


def create_logger_with_trace_id(name: str, trace_id: str = None) -> logging.LoggerAdapter:
    """Create a logger adapter that automatically includes trace_id."""
    if trace_id is None:
        trace_id = str(uuid.uuid4())
    
    logger = logging.getLogger(name)
    
    class TraceIDAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            if 'extra' not in kwargs:
                kwargs['extra'] = {}
            kwargs['extra']['trace_id'] = self.extra['trace_id']
            return msg, kwargs
    
    return TraceIDAdapter(logger, {'trace_id': trace_id})


def get_trace_id() -> str:
    """Generate a new trace ID."""
    return str(uuid.uuid4())