"""Advanced Performance Optimizer for SQL Query Synthesizer.

This module implements sophisticated performance optimization techniques including
query plan analysis, index recommendations, and automatic performance tuning.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import statistics
import hashlib
import json
import re
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class OptimizationTechnique(Enum):
    """Performance optimization techniques."""
    QUERY_REWRITE = "query_rewrite"
    INDEX_SUGGESTION = "index_suggestion"
    CACHING_STRATEGY = "caching_strategy"
    EXECUTION_PLAN = "execution_plan"
    PARALLEL_EXECUTION = "parallel_execution"
    MATERIALIZED_VIEWS = "materialized_views"
    PARTITIONING = "partitioning"
    QUERY_BATCHING = "query_batching"


class PerformanceIssueType(Enum):
    """Types of performance issues."""
    SLOW_QUERY = "slow_query"
    HIGH_CPU = "high_cpu"
    HIGH_MEMORY = "high_memory"
    HIGH_IO = "high_io"
    LOCK_CONTENTION = "lock_contention"
    INDEX_MISSING = "index_missing"
    INEFFICIENT_JOIN = "inefficient_join"
    FULL_TABLE_SCAN = "full_table_scan"


@dataclass
class QueryPerformanceMetrics:
    """Performance metrics for a specific query."""
    query_hash: str
    sql_snippet: str
    execution_count: int
    total_execution_time_ms: float
    avg_execution_time_ms: float
    min_execution_time_ms: float
    max_execution_time_ms: float
    std_dev_execution_time_ms: float
    rows_examined: int = 0
    rows_returned: int = 0
    cpu_time_ms: float = 0.0
    io_operations: int = 0
    memory_usage_mb: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    last_executed: float = 0.0
    optimization_applied: bool = False


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation."""
    recommendation_id: str
    query_hash: str
    technique: OptimizationTechnique
    issue_type: PerformanceIssueType
    description: str
    expected_improvement: float  # Percentage improvement
    implementation_cost: str  # LOW, MEDIUM, HIGH
    sql_before: str
    sql_after: Optional[str] = None
    additional_notes: List[str] = field(default_factory=list)
    priority_score: float = 0.0


class QueryPatternAnalyzer:
    """Analyzes query patterns to identify optimization opportunities."""
    
    def __init__(self):
        self.query_patterns: Dict[str, Dict[str, Any]] = {}
        self.table_access_patterns: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.join_patterns: Dict[str, int] = defaultdict(int)
        
    def analyze_query_pattern(self, sql: str, execution_metrics: Dict[str, Any]) -> List[str]:
        """Analyze a query to identify patterns and potential issues."""
        patterns = []
        sql_upper = sql.upper()
        
        # Identify basic patterns
        if 'SELECT *' in sql_upper:
            patterns.append('select_star')
        
        if 'ORDER BY' in sql_upper and 'LIMIT' not in sql_upper:
            patterns.append('unbounded_sort')
        
        if sql_upper.count('JOIN') > 3:
            patterns.append('complex_joins')
        
        if 'GROUP BY' in sql_upper and 'HAVING' in sql_upper:
            patterns.append('group_having')
        
        if re.search(r'WHERE\s+.*LIKE\s+[\'"]%', sql_upper):
            patterns.append('leading_wildcard')
        
        if 'DISTINCT' in sql_upper:
            patterns.append('distinct_usage')
        
        if sql_upper.count('SELECT') > 1:
            patterns.append('subqueries')
        
        # Analyze execution characteristics
        exec_time = execution_metrics.get('execution_time_ms', 0)
        rows_examined = execution_metrics.get('rows_examined', 0)
        rows_returned = execution_metrics.get('rows_returned', 0)
        
        if exec_time > 1000:  # > 1 second
            patterns.append('slow_execution')
        
        if rows_examined > rows_returned * 10:  # Examining 10x more than returning
            patterns.append('inefficient_scan')
        
        return patterns
    
    def extract_table_relationships(self, sql: str) -> Dict[str, List[str]]:
        """Extract table relationships from JOIN statements."""
        relationships = defaultdict(list)
        
        # Simple regex-based extraction (can be enhanced with proper SQL parsing)
        join_pattern = r'(?i)(?:inner\s+join|left\s+join|right\s+join|join)\s+(\w+)\s+(?:as\s+\w+\s+)?on\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)'
        
        for match in re.finditer(join_pattern, sql):
            table1, col1, table2, col2 = match.groups()[1:]
            relationships[table1].append(f"{table2}.{col2}")
            relationships[table2].append(f"{table1}.{col1}")
        
        return dict(relationships)
    
    def identify_frequent_patterns(self, min_frequency: int = 5) -> List[Dict[str, Any]]:
        """Identify frequently occurring query patterns."""
        pattern_frequency = defaultdict(int)
        
        for query_hash, data in self.query_patterns.items():
            for pattern in data.get('patterns', []):
                pattern_frequency[pattern] += 1
        
        frequent_patterns = []
        for pattern, frequency in pattern_frequency.items():
            if frequency >= min_frequency:
                frequent_patterns.append({
                    'pattern': pattern,
                    'frequency': frequency,
                    'optimization_priority': self._get_optimization_priority(pattern)
                })
        
        return sorted(frequent_patterns, key=lambda x: x['optimization_priority'], reverse=True)
    
    def _get_optimization_priority(self, pattern: str) -> float:
        """Get optimization priority for a pattern."""
        priority_map = {
            'slow_execution': 10.0,
            'inefficient_scan': 9.0,
            'select_star': 7.0,
            'complex_joins': 8.0,
            'leading_wildcard': 6.0,
            'unbounded_sort': 5.0,
            'group_having': 4.0,
            'subqueries': 3.0,
            'distinct_usage': 2.0
        }
        return priority_map.get(pattern, 1.0)


class IndexOptimizer:
    """Provides intelligent index recommendations based on query patterns."""
    
    def __init__(self):
        self.column_access_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.where_clause_patterns: Dict[str, int] = defaultdict(int)
        self.join_patterns: Dict[Tuple[str, str], int] = defaultdict(int)
        self.order_by_patterns: Dict[str, int] = defaultdict(int)
        
    def analyze_for_index_opportunities(
        self, 
        sql: str, 
        execution_time_ms: float,
        table_schema: Dict[str, List[str]] = None
    ) -> List[OptimizationRecommendation]:
        """Analyze query for index optimization opportunities."""
        recommendations = []
        
        # Extract patterns for index analysis
        self._extract_access_patterns(sql)
        
        # Generate index recommendations
        recommendations.extend(self._recommend_where_clause_indexes(sql, execution_time_ms))
        recommendations.extend(self._recommend_join_indexes(sql, execution_time_ms))
        recommendations.extend(self._recommend_order_by_indexes(sql, execution_time_ms))
        recommendations.extend(self._recommend_composite_indexes(sql, execution_time_ms))
        
        return recommendations
    
    def _extract_access_patterns(self, sql: str):
        """Extract column access patterns from SQL."""
        sql_upper = sql.upper()
        
        # Extract WHERE clause patterns
        where_matches = re.finditer(r'WHERE\s+(.+?)(?:\s+(?:ORDER|GROUP|LIMIT|$))', sql_upper)
        for match in where_matches:
            where_clause = match.group(1)
            # Extract column references
            column_matches = re.finditer(r'(\w+)\.(\w+)\s*[=<>]', where_clause)
            for col_match in column_matches:
                table, column = col_match.groups()
                self.where_clause_patterns[f"{table}.{column}"] += 1
        
        # Extract JOIN patterns
        join_matches = re.finditer(r'JOIN\s+(\w+).*?ON\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)', sql_upper)
        for match in join_matches:
            table1, col1, table2, col2 = match.groups()[1:]
            self.join_patterns[(f"{table1}.{col1}", f"{table2}.{col2}")] += 1
        
        # Extract ORDER BY patterns
        order_matches = re.finditer(r'ORDER\s+BY\s+(.+?)(?:\s+(?:LIMIT|$))', sql_upper)
        for match in order_matches:
            order_clause = match.group(1)
            # Extract column references
            column_matches = re.finditer(r'(\w+)\.(\w+)', order_clause)
            for col_match in column_matches:
                table, column = col_match.groups()
                self.order_by_patterns[f"{table}.{column}"] += 1
    
    def _recommend_where_clause_indexes(self, sql: str, execution_time_ms: float) -> List[OptimizationRecommendation]:
        """Recommend indexes for WHERE clause conditions."""
        recommendations = []
        
        if execution_time_ms < 100:  # Skip for fast queries
            return recommendations
        
        # Find frequently used WHERE conditions
        frequent_conditions = [(k, v) for k, v in self.where_clause_patterns.items() if v >= 3]
        
        for column, frequency in frequent_conditions:
            recommendation = OptimizationRecommendation(
                recommendation_id=f"IDX_WHERE_{hashlib.md5(column.encode()).hexdigest()[:8]}",
                query_hash=hashlib.md5(sql.encode()).hexdigest(),
                technique=OptimizationTechnique.INDEX_SUGGESTION,
                issue_type=PerformanceIssueType.INDEX_MISSING,
                description=f"Create index on {column} for WHERE clause optimization",
                expected_improvement=min(80.0, frequency * 10),  # Up to 80% improvement
                implementation_cost="LOW",
                sql_before=sql,
                additional_notes=[
                    f"Column {column} used in WHERE clause {frequency} times",
                    "Single column index should improve query performance significantly"
                ],
                priority_score=frequency * (execution_time_ms / 1000)
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _recommend_join_indexes(self, sql: str, execution_time_ms: float) -> List[OptimizationRecommendation]:
        """Recommend indexes for JOIN operations."""
        recommendations = []
        
        if execution_time_ms < 200:  # Skip for reasonably fast queries
            return recommendations
        
        # Find frequently used JOIN patterns
        frequent_joins = [(k, v) for k, v in self.join_patterns.items() if v >= 2]
        
        for (col1, col2), frequency in frequent_joins:
            # Recommend index on both sides of the join
            for column in [col1, col2]:
                recommendation = OptimizationRecommendation(
                    recommendation_id=f"IDX_JOIN_{hashlib.md5(column.encode()).hexdigest()[:8]}",
                    query_hash=hashlib.md5(sql.encode()).hexdigest(),
                    technique=OptimizationTechnique.INDEX_SUGGESTION,
                    issue_type=PerformanceIssueType.INEFFICIENT_JOIN,
                    description=f"Create index on {column} for JOIN optimization",
                    expected_improvement=min(70.0, frequency * 15),  # Up to 70% improvement
                    implementation_cost="LOW",
                    sql_before=sql,
                    additional_notes=[
                        f"Column {column} used in JOIN condition {frequency} times",
                        "Index will improve JOIN performance by enabling efficient lookups"
                    ],
                    priority_score=frequency * (execution_time_ms / 1000) * 0.8
                )
                recommendations.append(recommendation)
        
        return recommendations
    
    def _recommend_order_by_indexes(self, sql: str, execution_time_ms: float) -> List[OptimizationRecommendation]:
        """Recommend indexes for ORDER BY clauses."""
        recommendations = []
        
        if execution_time_ms < 300:  # Skip for reasonably fast queries
            return recommendations
        
        # Find frequently used ORDER BY patterns
        frequent_sorts = [(k, v) for k, v in self.order_by_patterns.items() if v >= 2]
        
        for column, frequency in frequent_sorts:
            recommendation = OptimizationRecommendation(
                recommendation_id=f"IDX_SORT_{hashlib.md5(column.encode()).hexdigest()[:8]}",
                query_hash=hashlib.md5(sql.encode()).hexdigest(),
                technique=OptimizationTechnique.INDEX_SUGGESTION,
                issue_type=PerformanceIssueType.FULL_TABLE_SCAN,
                description=f"Create index on {column} for ORDER BY optimization",
                expected_improvement=min(60.0, frequency * 12),  # Up to 60% improvement
                implementation_cost="LOW",
                sql_before=sql,
                additional_notes=[
                    f"Column {column} used in ORDER BY clause {frequency} times",
                    "Index will eliminate need for filesort operation"
                ],
                priority_score=frequency * (execution_time_ms / 1000) * 0.6
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _recommend_composite_indexes(self, sql: str, execution_time_ms: float) -> List[OptimizationRecommendation]:
        """Recommend composite indexes for complex queries."""
        recommendations = []
        
        if execution_time_ms < 500:  # Only for slower queries
            return recommendations
        
        # Analyze for potential composite index opportunities
        # This is a simplified version - production would need more sophisticated analysis
        sql_upper = sql.upper()
        
        # Look for queries with multiple WHERE conditions on the same table
        where_conditions = re.findall(r'(\w+)\.(\w+)\s*[=<>]', sql_upper)
        table_columns = defaultdict(list)
        
        for table, column in where_conditions:
            table_columns[table].append(column)
        
        # Recommend composite indexes for tables with multiple conditions
        for table, columns in table_columns.items():
            if len(columns) >= 2:
                composite_columns = list(set(columns))[:3]  # Limit to 3 columns
                column_list = ', '.join(f"{table}.{col}" for col in composite_columns)
                
                recommendation = OptimizationRecommendation(
                    recommendation_id=f"IDX_COMP_{hashlib.md5(column_list.encode()).hexdigest()[:8]}",
                    query_hash=hashlib.md5(sql.encode()).hexdigest(),
                    technique=OptimizationTechnique.INDEX_SUGGESTION,
                    issue_type=PerformanceIssueType.INEFFICIENT_JOIN,
                    description=f"Create composite index on ({column_list}) for multi-condition optimization",
                    expected_improvement=min(85.0, len(columns) * 20),  # Up to 85% improvement
                    implementation_cost="MEDIUM",
                    sql_before=sql,
                    additional_notes=[
                        f"Query uses {len(columns)} conditions on table {table}",
                        "Composite index can satisfy multiple conditions with single lookup",
                        "Consider column order based on selectivity"
                    ],
                    priority_score=len(columns) * (execution_time_ms / 1000) * 1.2
                )
                recommendations.append(recommendation)
        
        return recommendations


class QueryRewriter:
    """Rewrites queries for better performance."""
    
    def __init__(self):
        self.rewrite_rules = [
            self._rewrite_select_star,
            self._rewrite_or_conditions,
            self._rewrite_subqueries_to_joins,
            self._rewrite_exists_to_joins,
            self._rewrite_unnecessary_distinct,
            self._rewrite_inefficient_like
        ]
    
    def suggest_query_rewrites(self, sql: str, execution_time_ms: float) -> List[OptimizationRecommendation]:
        """Suggest query rewrites for performance improvement."""
        recommendations = []
        
        for rule in self.rewrite_rules:
            try:
                rewrite_rec = rule(sql, execution_time_ms)
                if rewrite_rec:
                    recommendations.append(rewrite_rec)
            except Exception as e:
                logger.warning(f"Query rewrite rule failed: {e}")
        
        return recommendations
    
    def _rewrite_select_star(self, sql: str, execution_time_ms: float) -> Optional[OptimizationRecommendation]:
        """Suggest replacing SELECT * with specific columns."""
        if 'SELECT *' not in sql.upper() or execution_time_ms < 100:
            return None
        
        return OptimizationRecommendation(
            recommendation_id=f"QRW_STAR_{hashlib.md5(sql.encode()).hexdigest()[:8]}",
            query_hash=hashlib.md5(sql.encode()).hexdigest(),
            technique=OptimizationTechnique.QUERY_REWRITE,
            issue_type=PerformanceIssueType.HIGH_IO,
            description="Replace SELECT * with specific column names",
            expected_improvement=30.0,
            implementation_cost="LOW",
            sql_before=sql,
            sql_after=sql.replace('SELECT *', 'SELECT column1, column2, ...'),
            additional_notes=[
                "SELECT * retrieves all columns, increasing I/O and network transfer",
                "Specify only the columns you actually need",
                "This can reduce query execution time and memory usage"
            ],
            priority_score=execution_time_ms / 1000 * 0.3
        )
    
    def _rewrite_or_conditions(self, sql: str, execution_time_ms: float) -> Optional[OptimizationRecommendation]:
        """Suggest rewriting OR conditions to UNION."""
        if ' OR ' not in sql.upper() or execution_time_ms < 200:
            return None
        
        # Simple detection - production would need more sophisticated parsing
        or_count = sql.upper().count(' OR ')
        if or_count >= 2:
            return OptimizationRecommendation(
                recommendation_id=f"QRW_OR_{hashlib.md5(sql.encode()).hexdigest()[:8]}",
                query_hash=hashlib.md5(sql.encode()).hexdigest(),
                technique=OptimizationTechnique.QUERY_REWRITE,
                issue_type=PerformanceIssueType.INEFFICIENT_JOIN,
                description="Consider rewriting OR conditions to UNION for better index usage",
                expected_improvement=40.0,
                implementation_cost="MEDIUM",
                sql_before=sql,
                additional_notes=[
                    f"Query contains {or_count} OR conditions",
                    "OR conditions can prevent efficient index usage",
                    "UNION queries can utilize indexes better for each condition"
                ],
                priority_score=or_count * (execution_time_ms / 1000) * 0.4
            )
        
        return None
    
    def _rewrite_subqueries_to_joins(self, sql: str, execution_time_ms: float) -> Optional[OptimizationRecommendation]:
        """Suggest converting correlated subqueries to JOINs."""
        if sql.upper().count('SELECT') <= 1 or execution_time_ms < 300:
            return None
        
        # Detect potential correlated subqueries
        if re.search(r'(?i)WHERE\s+.*?\(\s*SELECT\s+.*?WHERE\s+.*?\.\w+\s*=\s*.*?\.\w+', sql):
            return OptimizationRecommendation(
                recommendation_id=f"QRW_SUBQ_{hashlib.md5(sql.encode()).hexdigest()[:8]}",
                query_hash=hashlib.md5(sql.encode()).hexdigest(),
                technique=OptimizationTechnique.QUERY_REWRITE,
                issue_type=PerformanceIssueType.SLOW_QUERY,
                description="Convert correlated subquery to JOIN for better performance",
                expected_improvement=60.0,
                implementation_cost="HIGH",
                sql_before=sql,
                additional_notes=[
                    "Correlated subqueries execute for each row of outer query",
                    "JOIN operations are typically more efficient",
                    "Modern query optimizers handle JOINs better than subqueries"
                ],
                priority_score=execution_time_ms / 1000 * 0.6
            )
        
        return None
    
    def _rewrite_exists_to_joins(self, sql: str, execution_time_ms: float) -> Optional[OptimizationRecommendation]:
        """Suggest converting EXISTS clauses to JOINs."""
        if 'EXISTS' not in sql.upper() or execution_time_ms < 250:
            return None
        
        return OptimizationRecommendation(
            recommendation_id=f"QRW_EXISTS_{hashlib.md5(sql.encode()).hexdigest()[:8]}",
            query_hash=hashlib.md5(sql.encode()).hexdigest(),
            technique=OptimizationTechnique.QUERY_REWRITE,
            issue_type=PerformanceIssueType.SLOW_QUERY,
            description="Consider rewriting EXISTS to INNER JOIN for better performance",
            expected_improvement=35.0,
            implementation_cost="MEDIUM",
            sql_before=sql,
            additional_notes=[
                "EXISTS clauses can often be rewritten as JOINs",
                "JOINs may provide better execution plans",
                "Use DISTINCT if necessary to maintain result set"
            ],
            priority_score=execution_time_ms / 1000 * 0.35
        )
    
    def _rewrite_unnecessary_distinct(self, sql: str, execution_time_ms: float) -> Optional[OptimizationRecommendation]:
        """Suggest removing unnecessary DISTINCT clauses."""
        if 'DISTINCT' not in sql.upper() or execution_time_ms < 150:
            return None
        
        # Check if DISTINCT might be unnecessary (basic heuristic)
        if 'GROUP BY' in sql.upper() or 'UNIQUE' in sql.upper():
            return OptimizationRecommendation(
                recommendation_id=f"QRW_DIST_{hashlib.md5(sql.encode()).hexdigest()[:8]}",
                query_hash=hashlib.md5(sql.encode()).hexdigest(),
                technique=OptimizationTechnique.QUERY_REWRITE,
                issue_type=PerformanceIssueType.HIGH_CPU,
                description="Review if DISTINCT is necessary - it may be redundant",
                expected_improvement=25.0,
                implementation_cost="LOW",
                sql_before=sql,
                additional_notes=[
                    "DISTINCT requires sorting/hashing to eliminate duplicates",
                    "If data is already unique, DISTINCT adds unnecessary overhead",
                    "Check if constraints or GROUP BY already ensure uniqueness"
                ],
                priority_score=execution_time_ms / 1000 * 0.25
            )
        
        return None
    
    def _rewrite_inefficient_like(self, sql: str, execution_time_ms: float) -> Optional[OptimizationRecommendation]:
        """Suggest alternatives for inefficient LIKE patterns."""
        if execution_time_ms < 200:
            return None
        
        # Check for leading wildcard LIKE patterns
        if re.search(r'(?i)LIKE\s+[\'"]%', sql):
            return OptimizationRecommendation(
                recommendation_id=f"QRW_LIKE_{hashlib.md5(sql.encode()).hexdigest()[:8]}",
                query_hash=hashlib.md5(sql.encode()).hexdigest(),
                technique=OptimizationTechnique.QUERY_REWRITE,
                issue_type=PerformanceIssueType.FULL_TABLE_SCAN,
                description="Consider alternatives to LIKE with leading wildcard",
                expected_improvement=70.0,
                implementation_cost="HIGH",
                sql_before=sql,
                additional_notes=[
                    "Leading wildcard LIKE patterns cannot use indexes",
                    "Consider full-text search indexes for text searching",
                    "Reverse indexes might help for suffix matching",
                    "Regular expressions might be more efficient for complex patterns"
                ],
                priority_score=execution_time_ms / 1000 * 0.7
            )
        
        return None


class PerformanceOptimizer:
    """Main performance optimizer that coordinates all optimization techniques."""
    
    def __init__(self):
        self.query_analyzer = QueryPatternAnalyzer()
        self.index_optimizer = IndexOptimizer()
        self.query_rewriter = QueryRewriter()
        
        # Performance tracking
        self.query_metrics: Dict[str, QueryPerformanceMetrics] = {}
        self.optimization_history: List[Dict[str, Any]] = []
        self._metrics_lock = threading.Lock()
        
        # Optimization thresholds
        self.slow_query_threshold_ms = 1000
        self.optimization_benefit_threshold = 20.0  # Minimum 20% improvement
        
    def record_query_execution(
        self, 
        sql: str, 
        execution_time_ms: float,
        additional_metrics: Dict[str, Any] = None
    ):
        """Record query execution for performance analysis."""
        query_hash = hashlib.md5(sql.encode()).hexdigest()
        additional_metrics = additional_metrics or {}
        
        with self._metrics_lock:
            if query_hash not in self.query_metrics:
                self.query_metrics[query_hash] = QueryPerformanceMetrics(
                    query_hash=query_hash,
                    sql_snippet=sql[:200] + "..." if len(sql) > 200 else sql,
                    execution_count=0,
                    total_execution_time_ms=0.0,
                    avg_execution_time_ms=0.0,
                    min_execution_time_ms=float('inf'),
                    max_execution_time_ms=0.0,
                    std_dev_execution_time_ms=0.0
                )
            
            metrics = self.query_metrics[query_hash]
            
            # Update metrics
            metrics.execution_count += 1
            metrics.total_execution_time_ms += execution_time_ms
            metrics.avg_execution_time_ms = metrics.total_execution_time_ms / metrics.execution_count
            metrics.min_execution_time_ms = min(metrics.min_execution_time_ms, execution_time_ms)
            metrics.max_execution_time_ms = max(metrics.max_execution_time_ms, execution_time_ms)
            metrics.last_executed = time.time()
            
            # Update additional metrics
            metrics.rows_examined = additional_metrics.get('rows_examined', metrics.rows_examined)
            metrics.rows_returned = additional_metrics.get('rows_returned', metrics.rows_returned)
            metrics.cpu_time_ms = additional_metrics.get('cpu_time_ms', metrics.cpu_time_ms)
            metrics.io_operations = additional_metrics.get('io_operations', metrics.io_operations)
            metrics.memory_usage_mb = additional_metrics.get('memory_usage_mb', metrics.memory_usage_mb)
            
            # Update cache metrics
            if additional_metrics.get('cache_hit'):
                metrics.cache_hits += 1
            else:
                metrics.cache_misses += 1
        
        # Record patterns for analysis
        patterns = self.query_analyzer.analyze_query_pattern(sql, additional_metrics)
        if patterns:
            self.query_analyzer.query_patterns[query_hash] = {
                'patterns': patterns,
                'last_updated': time.time()
            }
    
    def generate_optimization_recommendations(
        self, 
        limit: int = 50,
        min_execution_time_ms: float = None
    ) -> List[OptimizationRecommendation]:
        """Generate comprehensive optimization recommendations."""
        
        all_recommendations = []
        min_time = min_execution_time_ms or self.slow_query_threshold_ms
        
        # Get slow queries for optimization
        slow_queries = [
            (hash_val, metrics) for hash_val, metrics in self.query_metrics.items()
            if metrics.avg_execution_time_ms >= min_time and metrics.execution_count >= 2
        ]
        
        # Sort by priority (combination of execution time and frequency)
        slow_queries.sort(
            key=lambda x: x[1].avg_execution_time_ms * x[1].execution_count, 
            reverse=True
        )
        
        # Generate recommendations for top slow queries
        for query_hash, metrics in slow_queries[:limit]:
            sql = self._get_full_sql_for_hash(query_hash)
            if not sql:
                continue
            
            # Index optimization recommendations
            index_recs = self.index_optimizer.analyze_for_index_opportunities(
                sql, metrics.avg_execution_time_ms
            )
            all_recommendations.extend(index_recs)
            
            # Query rewrite recommendations
            rewrite_recs = self.query_rewriter.suggest_query_rewrites(
                sql, metrics.avg_execution_time_ms
            )
            all_recommendations.extend(rewrite_recs)
            
            # Caching recommendations
            if metrics.execution_count >= 5 and metrics.cache_misses > metrics.cache_hits:
                cache_rec = self._generate_caching_recommendation(query_hash, metrics, sql)
                if cache_rec:
                    all_recommendations.append(cache_rec)
        
        # Sort recommendations by priority score
        all_recommendations.sort(key=lambda x: x.priority_score, reverse=True)
        
        return all_recommendations[:limit]
    
    def _get_full_sql_for_hash(self, query_hash: str) -> Optional[str]:
        """Get full SQL for a query hash (placeholder - would need actual storage)."""
        # In production, you'd store full SQL statements or reconstruct them
        # For now, return the snippet from metrics
        metrics = self.query_metrics.get(query_hash)
        return metrics.sql_snippet if metrics else None
    
    def _generate_caching_recommendation(
        self, 
        query_hash: str, 
        metrics: QueryPerformanceMetrics, 
        sql: str
    ) -> Optional[OptimizationRecommendation]:
        """Generate caching recommendation for frequently executed queries."""
        
        if metrics.execution_count < 5 or metrics.avg_execution_time_ms < 100:
            return None
        
        return OptimizationRecommendation(
            recommendation_id=f"CACHE_{query_hash[:8]}",
            query_hash=query_hash,
            technique=OptimizationTechnique.CACHING_STRATEGY,
            issue_type=PerformanceIssueType.SLOW_QUERY,
            description="Enable aggressive caching for frequently executed query",
            expected_improvement=min(80.0, metrics.execution_count * 5),
            implementation_cost="LOW",
            sql_before=sql,
            additional_notes=[
                f"Query executed {metrics.execution_count} times",
                f"Average execution time: {metrics.avg_execution_time_ms:.1f}ms",
                "High cache hit rate potential due to frequency"
            ],
            priority_score=metrics.execution_count * (metrics.avg_execution_time_ms / 1000)
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self._metrics_lock:
            if not self.query_metrics:
                return {"total_queries": 0}
            
            total_queries = len(self.query_metrics)
            total_executions = sum(m.execution_count for m in self.query_metrics.values())
            
            # Calculate aggregate statistics
            all_avg_times = [m.avg_execution_time_ms for m in self.query_metrics.values()]
            overall_avg_time = statistics.mean(all_avg_times)
            overall_median_time = statistics.median(all_avg_times)
            
            # Identify slow queries
            slow_queries = [
                m for m in self.query_metrics.values() 
                if m.avg_execution_time_ms >= self.slow_query_threshold_ms
            ]
            
            # Top 10 slowest queries
            slowest_queries = sorted(
                self.query_metrics.values(),
                key=lambda x: x.avg_execution_time_ms,
                reverse=True
            )[:10]
            
            # Most frequent queries
            frequent_queries = sorted(
                self.query_metrics.values(),
                key=lambda x: x.execution_count,
                reverse=True
            )[:10]
            
            # Cache statistics
            total_cache_hits = sum(m.cache_hits for m in self.query_metrics.values())
            total_cache_misses = sum(m.cache_misses for m in self.query_metrics.values())
            cache_hit_rate = (
                total_cache_hits / (total_cache_hits + total_cache_misses) * 100
                if (total_cache_hits + total_cache_misses) > 0 else 0
            )
            
            return {
                "total_unique_queries": total_queries,
                "total_executions": total_executions,
                "overall_avg_execution_time_ms": overall_avg_time,
                "overall_median_execution_time_ms": overall_median_time,
                "slow_queries_count": len(slow_queries),
                "cache_hit_rate_percent": cache_hit_rate,
                "slowest_queries": [
                    {
                        "query_hash": q.query_hash,
                        "sql_snippet": q.sql_snippet,
                        "avg_time_ms": q.avg_execution_time_ms,
                        "execution_count": q.execution_count
                    } for q in slowest_queries
                ],
                "most_frequent_queries": [
                    {
                        "query_hash": q.query_hash,
                        "sql_snippet": q.sql_snippet,
                        "execution_count": q.execution_count,
                        "avg_time_ms": q.avg_execution_time_ms
                    } for q in frequent_queries
                ],
                "optimization_recommendations_available": len(slow_queries),
                "frequent_patterns": self.query_analyzer.identify_frequent_patterns()
            }
    
    def apply_optimization(self, recommendation_id: str) -> bool:
        """Apply an optimization recommendation (placeholder for actual implementation)."""
        # In production, this would actually apply the optimization
        # For now, just record that it was applied
        
        self.optimization_history.append({
            "recommendation_id": recommendation_id,
            "applied_at": time.time(),
            "status": "applied"
        })
        
        logger.info(f"Applied optimization recommendation: {recommendation_id}")
        return True
    
    def get_optimization_impact(self, days: int = 7) -> Dict[str, Any]:
        """Get the impact of applied optimizations."""
        cutoff_time = time.time() - (days * 24 * 3600)
        
        recent_optimizations = [
            opt for opt in self.optimization_history
            if opt.get('applied_at', 0) > cutoff_time
        ]
        
        return {
            "optimizations_applied": len(recent_optimizations),
            "time_period_days": days,
            "recent_optimizations": recent_optimizations[-10:],  # Last 10
            "estimated_performance_improvement": len(recent_optimizations) * 15  # Rough estimate
        }


# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()