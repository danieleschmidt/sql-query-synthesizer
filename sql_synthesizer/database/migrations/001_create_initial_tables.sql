-- Version: 20250803_001
-- Name: Create Initial Tables
-- Description: Initial database schema with query history and system metrics tables
-- Created: 2025-08-03T00:00:00

-- UP
-- Create schema_migrations table for tracking migrations
CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(20) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create query_history table for storing query execution history
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

-- Create indexes for query_history
CREATE INDEX IF NOT EXISTS idx_query_history_created_at ON query_history(created_at);
CREATE INDEX IF NOT EXISTS idx_query_history_success ON query_history(success);
CREATE INDEX IF NOT EXISTS idx_query_history_cache_hit ON query_history(cache_hit);
CREATE INDEX IF NOT EXISTS idx_query_history_query_id ON query_history(query_id);

-- Create system_metrics table for performance monitoring
CREATE TABLE IF NOT EXISTS system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name VARCHAR(100) NOT NULL,
    metric_value REAL NOT NULL,
    metric_type VARCHAR(20) DEFAULT 'gauge',
    tags TEXT DEFAULT '{}',
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for system_metrics
CREATE INDEX IF NOT EXISTS idx_system_metrics_name_time ON system_metrics(metric_name, recorded_at);
CREATE INDEX IF NOT EXISTS idx_system_metrics_recorded_at ON system_metrics(recorded_at);

-- DOWN
-- Drop all tables and indexes
DROP INDEX IF EXISTS idx_system_metrics_recorded_at;
DROP INDEX IF EXISTS idx_system_metrics_name_time;
DROP TABLE IF EXISTS system_metrics;

DROP INDEX IF EXISTS idx_query_history_query_id;
DROP INDEX IF EXISTS idx_query_history_cache_hit;
DROP INDEX IF EXISTS idx_query_history_success;
DROP INDEX IF EXISTS idx_query_history_created_at;
DROP TABLE IF EXISTS query_history;

DROP TABLE IF EXISTS schema_migrations;