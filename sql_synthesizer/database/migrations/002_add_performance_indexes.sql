-- Version: 20250803_002
-- Name: Add Performance Indexes
-- Description: Additional performance indexes and optimizations for query patterns
-- Created: 2025-08-03T00:00:00

-- UP
-- Add composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_query_history_success_time ON query_history(success, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_query_history_user_question_hash ON query_history(substr(user_question, 1, 100));

-- Add index for error analysis
CREATE INDEX IF NOT EXISTS idx_query_history_error_analysis ON query_history(success, error_message) 
WHERE success = 0 AND error_message IS NOT NULL;

-- Add partitioning-friendly index for cleanup operations
CREATE INDEX IF NOT EXISTS idx_query_history_cleanup ON query_history(created_at, success);

-- Add metrics aggregation indexes
CREATE INDEX IF NOT EXISTS idx_system_metrics_aggregation ON system_metrics(metric_name, metric_type, recorded_at);

-- Add user tracking indexes (for analytics)
CREATE INDEX IF NOT EXISTS idx_query_history_client_tracking ON query_history(client_ip, created_at) 
WHERE client_ip IS NOT NULL;

-- DOWN
-- Remove all performance indexes added in this migration
DROP INDEX IF EXISTS idx_query_history_client_tracking;
DROP INDEX IF EXISTS idx_system_metrics_aggregation;
DROP INDEX IF EXISTS idx_query_history_cleanup;
DROP INDEX IF EXISTS idx_query_history_error_analysis;
DROP INDEX IF EXISTS idx_query_history_user_question_hash;
DROP INDEX IF EXISTS idx_query_history_success_time;