"""
Intelligent Query Insights Engine
Provides advanced analysis and optimization suggestions for SQL queries using NLP and pattern analysis.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class QueryInsight:
    """Represents an insight or optimization suggestion for a query."""
    type: str  # performance, readability, security, best_practice
    severity: str  # low, medium, high, critical
    title: str
    description: str
    suggestion: str
    estimated_improvement: Optional[str] = None
    confidence: float = 0.0


@dataclass
class QueryComplexity:
    """Metrics representing query complexity."""
    total_score: int
    join_complexity: int
    subquery_depth: int
    aggregate_functions: int
    where_conditions: int
    table_count: int
    readability_score: float


class QueryPatternAnalyzer:
    """Advanced pattern analysis for SQL queries."""
    
    def __init__(self):
        self.common_patterns = {
            'n_plus_one': r'SELECT.*FROM.*WHERE.*IN\s*\(SELECT',
            'missing_index': r'WHERE.*LIKE\s*[\'"]%.*%[\'"]',
            'cartesian_product': r'FROM.*,.*WHERE.*=',
            'select_star': r'SELECT\s+\*\s+FROM',
            'unnecessary_distinct': r'SELECT\s+DISTINCT.*GROUP\s+BY',
            'inefficient_join': r'LEFT\s+JOIN.*WHERE.*IS\s+NOT\s+NULL',
            'subquery_to_join': r'WHERE.*IN\s*\(SELECT.*FROM',
            'function_in_where': r'WHERE\s+\w+\([^)]*\)\s*[=<>]',
        }
        
        self.performance_keywords = {
            'high_cost': ['CROSS JOIN', 'CARTESIAN', 'FULL OUTER JOIN'],
            'expensive_functions': ['SUBSTR', 'UPPER', 'LOWER', 'REGEXP'],
            'table_scan': ['LIKE \'%', 'NOT LIKE \'%', 'REGEXP'],
        }
    
    def analyze_patterns(self, sql: str) -> List[QueryInsight]:
        """Analyze SQL for performance and optimization patterns."""
        insights = []
        sql_upper = sql.upper()
        
        # Pattern-based analysis
        for pattern_name, pattern in self.common_patterns.items():
            if re.search(pattern, sql, re.IGNORECASE):
                insight = self._create_pattern_insight(pattern_name, sql)
                if insight:
                    insights.append(insight)
        
        # Keyword-based analysis
        for category, keywords in self.performance_keywords.items():
            for keyword in keywords:
                if keyword in sql_upper:
                    insight = self._create_keyword_insight(category, keyword)
                    if insight:
                        insights.append(insight)
        
        return insights
    
    def _create_pattern_insight(self, pattern_name: str, sql: str) -> Optional[QueryInsight]:
        """Create insight based on detected pattern."""
        pattern_insights = {
            'n_plus_one': QueryInsight(
                type='performance',
                severity='high',
                title='Potential N+1 Query Pattern',
                description='Detected nested SELECT in WHERE clause that may cause N+1 queries',
                suggestion='Consider using JOINs or EXISTS instead of subqueries in WHERE clauses',
                estimated_improvement='50-80% performance improvement',
                confidence=0.8
            ),
            'missing_index': QueryInsight(
                type='performance',
                severity='medium',
                title='Potential Missing Index',
                description='LIKE pattern with leading wildcard detected',
                suggestion='Consider full-text search or restructuring the query to avoid leading wildcards',
                estimated_improvement='10-50x performance improvement',
                confidence=0.7
            ),
            'select_star': QueryInsight(
                type='best_practice',
                severity='low',
                title='SELECT * Usage',
                description='Using SELECT * can impact performance and maintainability',
                suggestion='Specify only the columns you need to reduce network overhead and improve clarity',
                estimated_improvement='5-20% network reduction',
                confidence=0.9
            ),
            'unnecessary_distinct': QueryInsight(
                type='performance',
                severity='medium',
                title='Unnecessary DISTINCT',
                description='DISTINCT with GROUP BY may be redundant',
                suggestion='Remove DISTINCT when using GROUP BY as it already provides unique results',
                estimated_improvement='10-30% performance improvement',
                confidence=0.8
            )
        }
        
        return pattern_insights.get(pattern_name)
    
    def _create_keyword_insight(self, category: str, keyword: str) -> Optional[QueryInsight]:
        """Create insight based on detected keyword."""
        keyword_insights = {
            'high_cost': QueryInsight(
                type='performance',
                severity='high',
                title=f'High-Cost Operation: {keyword}',
                description=f'Detected potentially expensive operation: {keyword}',
                suggestion='Consider alternative approaches or ensure proper indexing',
                confidence=0.7
            ),
            'expensive_functions': QueryInsight(
                type='performance',
                severity='medium',
                title=f'Expensive Function: {keyword}',
                description=f'Function {keyword} in WHERE clause may prevent index usage',
                suggestion='Consider functional indexes or restructuring to allow index usage',
                confidence=0.6
            ),
            'table_scan': QueryInsight(
                type='performance',
                severity='medium',
                title='Potential Table Scan',
                description=f'Pattern {keyword} may cause full table scan',
                suggestion='Consider alternative search strategies or full-text indexing',
                confidence=0.7
            )
        }
        
        return keyword_insights.get(category)


class QueryComplexityAnalyzer:
    """Analyzes query complexity using multiple metrics."""
    
    def analyze_complexity(self, sql: str) -> QueryComplexity:
        """Calculate comprehensive complexity score for SQL query."""
        sql_upper = sql.upper()
        
        # Count joins
        join_count = len(re.findall(r'\b(?:INNER|LEFT|RIGHT|FULL|CROSS)\s+JOIN\b', sql_upper))
        join_complexity = min(join_count * 2, 10)  # Cap at 10
        
        # Count subqueries
        subquery_depth = self._count_subquery_depth(sql)
        
        # Count aggregate functions
        agg_functions = len(re.findall(r'\b(?:COUNT|SUM|AVG|MAX|MIN|GROUP_CONCAT)\s*\(', sql_upper))
        
        # Count WHERE conditions (approximation)
        where_conditions = len(re.findall(r'\bAND\b|\bOR\b', sql_upper)) + (1 if 'WHERE' in sql_upper else 0)
        
        # Count tables
        # Simple heuristic: count FROM and JOIN clauses
        table_count = len(re.findall(r'\bFROM\s+\w+|\bJOIN\s+\w+', sql_upper))
        
        # Calculate readability score (inverse of complexity)
        line_count = len(sql.split('\n'))
        avg_line_length = sum(len(line) for line in sql.split('\n')) / max(line_count, 1)
        readability_score = max(0, 10 - (avg_line_length / 20) - (line_count / 10))
        
        # Total complexity score
        total_score = (
            join_complexity + 
            (subquery_depth * 3) + 
            (agg_functions * 2) + 
            where_conditions +
            table_count
        )
        
        return QueryComplexity(
            total_score=total_score,
            join_complexity=join_complexity,
            subquery_depth=subquery_depth,
            aggregate_functions=agg_functions,
            where_conditions=where_conditions,
            table_count=table_count,
            readability_score=readability_score
        )
    
    def _count_subquery_depth(self, sql: str) -> int:
        """Count maximum nesting depth of subqueries."""
        depth = 0
        max_depth = 0
        in_string = False
        string_char = None
        
        for i, char in enumerate(sql):
            # Handle string literals
            if char in ['"', "'"] and (i == 0 or sql[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
                continue
            
            if in_string:
                continue
            
            if char == '(':
                # Check if this is likely a subquery
                prev_context = sql[max(0, i-20):i].upper()
                if any(keyword in prev_context for keyword in ['SELECT', 'FROM', 'WHERE', 'HAVING']):
                    depth += 1
                    max_depth = max(max_depth, depth)
            elif char == ')':
                depth = max(0, depth - 1)
        
        return max_depth


class QueryInsightsEngine:
    """Main engine for generating query insights and recommendations."""
    
    def __init__(self):
        self.pattern_analyzer = QueryPatternAnalyzer()
        self.complexity_analyzer = QueryComplexityAnalyzer()
        self.query_history = []
    
    def analyze_query(self, sql: str, execution_time_ms: Optional[float] = None) -> Dict:
        """Comprehensive analysis of a single query."""
        try:
            # Pattern analysis
            pattern_insights = self.pattern_analyzer.analyze_patterns(sql)
            
            # Complexity analysis
            complexity = self.complexity_analyzer.analyze_complexity(sql)
            
            # Performance insights based on execution time
            performance_insights = []
            if execution_time_ms:
                performance_insights = self._analyze_performance(sql, execution_time_ms, complexity)
            
            # Security insights
            security_insights = self._analyze_security(sql)
            
            # Combine all insights
            all_insights = pattern_insights + performance_insights + security_insights
            
            # Sort by severity and confidence
            severity_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
            all_insights.sort(
                key=lambda x: (severity_order.get(x.severity, 0), x.confidence), 
                reverse=True
            )
            
            return {
                'insights': [asdict(insight) for insight in all_insights],
                'complexity': asdict(complexity),
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'sql_hash': hash(sql.strip()),
                'total_insights': len(all_insights),
                'critical_issues': len([i for i in all_insights if i.severity == 'critical']),
                'high_priority_issues': len([i for i in all_insights if i.severity == 'high'])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            return {
                'insights': [],
                'complexity': None,
                'error': str(e),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
    
    def _analyze_performance(self, sql: str, execution_time_ms: float, complexity: QueryComplexity) -> List[QueryInsight]:
        """Analyze performance based on execution time and complexity."""
        insights = []
        
        # Slow query detection
        if execution_time_ms > 1000:  # 1 second
            insights.append(QueryInsight(
                type='performance',
                severity='high' if execution_time_ms > 5000 else 'medium',
                title=f'Slow Query Detected ({execution_time_ms:.0f}ms)',
                description=f'Query executed in {execution_time_ms:.0f}ms, which is slower than recommended',
                suggestion='Consider adding indexes, optimizing joins, or refactoring complex operations',
                confidence=0.9
            ))
        
        # Complexity-based insights
        if complexity.total_score > 20:
            insights.append(QueryInsight(
                type='readability',
                severity='medium',
                title='High Complexity Query',
                description=f'Query complexity score is {complexity.total_score}, which may impact maintainability',
                suggestion='Consider breaking down into smaller queries or using views for complex logic',
                confidence=0.8
            ))
        
        # Join-heavy queries
        if complexity.join_complexity > 6:
            insights.append(QueryInsight(
                type='performance',
                severity='medium',
                title='Join-Heavy Query',
                description=f'Query contains many joins ({complexity.join_complexity//2} joins)',
                suggestion='Verify that all joins are necessary and properly indexed',
                confidence=0.7
            ))
        
        return insights
    
    def _analyze_security(self, sql: str) -> List[QueryInsight]:
        """Analyze potential security issues in SQL."""
        insights = []
        
        # Check for potential SQL injection patterns (basic)
        dangerous_patterns = [
            (r"'.*'.*'", "Potential string concatenation in query"),
            (r"\+.*\+", "Potential string concatenation operator"),
            (r"EXEC\s*\(", "Dynamic SQL execution detected"),
            (r"EXECUTE\s*\(", "Dynamic SQL execution detected"),
        ]
        
        for pattern, description in dangerous_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                insights.append(QueryInsight(
                    type='security',
                    severity='high',
                    title='Potential Security Risk',
                    description=description,
                    suggestion='Use parameterized queries and avoid dynamic SQL construction',
                    confidence=0.6
                ))
        
        return insights
    
    def batch_analyze_queries(self, queries: List[Dict]) -> Dict:
        """Analyze multiple queries and provide aggregate insights."""
        results = []
        aggregate_stats = {
            'total_queries': len(queries),
            'total_insights': 0,
            'avg_complexity': 0,
            'common_issues': Counter(),
            'performance_distribution': {'fast': 0, 'medium': 0, 'slow': 0}
        }
        
        for query_data in queries:
            sql = query_data.get('sql', '')
            execution_time = query_data.get('execution_time_ms', 0)
            
            analysis = self.analyze_query(sql, execution_time)
            results.append(analysis)
            
            # Update aggregate stats
            if analysis.get('complexity'):
                aggregate_stats['avg_complexity'] += analysis['complexity']['total_score']
            
            aggregate_stats['total_insights'] += len(analysis.get('insights', []))
            
            # Categorize query performance
            if execution_time < 100:
                aggregate_stats['performance_distribution']['fast'] += 1
            elif execution_time < 1000:
                aggregate_stats['performance_distribution']['medium'] += 1
            else:
                aggregate_stats['performance_distribution']['slow'] += 1
            
            # Count common issues
            for insight in analysis.get('insights', []):
                aggregate_stats['common_issues'][insight.get('title', 'Unknown')] += 1
        
        if queries:
            aggregate_stats['avg_complexity'] /= len(queries)
        
        return {
            'individual_analyses': results,
            'aggregate_statistics': aggregate_stats,
            'recommendations': self._generate_aggregate_recommendations(aggregate_stats),
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
    
    def _generate_aggregate_recommendations(self, stats: Dict) -> List[str]:
        """Generate system-wide recommendations based on aggregate statistics."""
        recommendations = []
        
        if stats['performance_distribution']['slow'] > stats['total_queries'] * 0.2:
            recommendations.append(
                "Consider implementing query optimization review process - "
                f"{stats['performance_distribution']['slow']} slow queries detected"
            )
        
        if stats['avg_complexity'] > 15:
            recommendations.append(
                f"Average query complexity is high ({stats['avg_complexity']:.1f}). "
                "Consider query refactoring guidelines and complexity limits"
            )
        
        common_issues = stats['common_issues'].most_common(3)
        if common_issues:
            recommendations.append(
                f"Most common issues: {', '.join([issue[0] for issue in common_issues])}. "
                "Consider creating coding standards to address these patterns"
            )
        
        return recommendations