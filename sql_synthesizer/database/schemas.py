"""Database schema definitions and data models."""

from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class QueryHistory:
    """Query history data model."""
    query_id: str
    user_question: str
    generated_sql: str
    execution_time_ms: float
    success: bool
    error_message: Optional[str] = None
    cache_hit: bool = False
    user_agent: Optional[str] = None
    client_ip: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'query_id': self.query_id,
            'user_question': self.user_question,
            'generated_sql': self.generated_sql,
            'execution_time_ms': self.execution_time_ms,
            'success': self.success,
            'error_message': self.error_message,
            'cache_hit': self.cache_hit,
            'user_agent': self.user_agent,
            'client_ip': self.client_ip,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


@dataclass
class SystemMetrics:
    """System metrics data model."""
    metric_name: str
    metric_value: float
    metric_type: str = 'gauge'  # gauge, counter, histogram
    tags: str = '{}'  # JSON string of tags
    recorded_at: Optional[datetime] = None
    id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'metric_type': self.metric_type,
            'tags': self.tags,
            'recorded_at': self.recorded_at.isoformat() if self.recorded_at else None
        }


# Database schema SQL definitions
QUERY_HISTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id VARCHAR(50) UNIQUE NOT NULL,
    user_question TEXT NOT NULL,
    generated_sql TEXT,
    execution_time_ms REAL,
    success BOOLEAN DEFAULT 1,
    error_message TEXT,
    cache_hit BOOLEAN DEFAULT 0,
    user_agent TEXT,
    client_ip VARCHAR(45),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_query_history_created_at ON query_history(created_at);
CREATE INDEX IF NOT EXISTS idx_query_history_success ON query_history(success);
CREATE INDEX IF NOT EXISTS idx_query_history_cache_hit ON query_history(cache_hit);
"""

SYSTEM_METRICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name VARCHAR(100) NOT NULL,
    metric_value REAL NOT NULL,
    metric_type VARCHAR(20) DEFAULT 'gauge',
    tags TEXT DEFAULT '{}',
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_system_metrics_name_time ON system_metrics(metric_name, recorded_at);
CREATE INDEX IF NOT EXISTS idx_system_metrics_recorded_at ON system_metrics(recorded_at);
"""

SCHEMA_MIGRATIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(20) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Complete schema initialization SQL
COMPLETE_SCHEMA = f"""
{SCHEMA_MIGRATIONS_SCHEMA}

{QUERY_HISTORY_SCHEMA}

{SYSTEM_METRICS_SCHEMA}
"""