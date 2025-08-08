"""
Adaptive Learning Engine for SQL Query Synthesis
Learns from user patterns, query performance, and feedback to improve recommendations.
"""

import json
import logging
import pickle
import hashlib
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QueryPattern:
    """Represents a learned pattern from query history."""
    pattern_id: str
    pattern_type: str  # syntax, semantic, performance, domain
    description: str
    frequency: int
    success_rate: float
    avg_performance_ms: float
    user_feedback_score: float
    confidence: float
    examples: List[str]
    created_at: str
    last_seen: str


@dataclass
class LearningInsight:
    """Insight generated from adaptive learning."""
    type: str
    confidence: float
    description: str
    recommendation: str
    supporting_evidence: List[str]
    impact_score: float


class QueryFeatureExtractor:
    """Extracts features from SQL queries for pattern recognition."""
    
    def __init__(self):
        self.common_keywords = {
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP', 'ORDER', 'HAVING',
            'UNION', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP'
        }
        
        self.aggregate_functions = {
            'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'STDDEV', 'VARIANCE'
        }
        
        self.join_types = {
            'INNER', 'LEFT', 'RIGHT', 'FULL', 'CROSS'
        }
    
    def extract_features(self, sql: str) -> Dict[str, Any]:
        """Extract comprehensive features from SQL query."""
        sql_upper = sql.upper()
        tokens = sql_upper.split()
        
        features = {
            # Structural features
            'query_length': len(sql),
            'token_count': len(tokens),
            'line_count': len(sql.split('\\n')),
            
            # Keyword frequency
            'keyword_counts': {kw: tokens.count(kw) for kw in self.common_keywords},
            'aggregate_usage': {func: tokens.count(func) for func in self.aggregate_functions},
            'join_types': {jtype: sql_upper.count(f'{jtype} JOIN') for jtype in self.join_types},
            
            # Complexity indicators
            'parentheses_depth': self._calculate_max_depth(sql, '(', ')'),
            'subquery_count': sql_upper.count('SELECT') - 1,  # Main query + subqueries
            'where_conditions': sql_upper.count(' AND ') + sql_upper.count(' OR ') + (1 if 'WHERE' in sql_upper else 0),
            
            # Semantic features
            'has_aggregation': any(func in sql_upper for func in self.aggregate_functions),
            'has_join': 'JOIN' in sql_upper,
            'has_subquery': sql_upper.count('SELECT') > 1,
            'has_grouping': 'GROUP BY' in sql_upper,
            'has_ordering': 'ORDER BY' in sql_upper,
            'has_limit': 'LIMIT' in sql_upper,
            
            # Pattern-specific features
            'table_references': self._extract_table_references(sql),
            'column_references': self._extract_column_references(sql),
            'literal_values': self._extract_literals(sql),
        }
        
        return features
    
    def _calculate_max_depth(self, text: str, open_char: str, close_char: str) -> int:
        """Calculate maximum nesting depth of characters."""
        depth = 0
        max_depth = 0
        
        for char in text:
            if char == open_char:
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == close_char:
                depth = max(0, depth - 1)
        
        return max_depth
    
    def _extract_table_references(self, sql: str) -> List[str]:
        """Extract table names from SQL query (simplified)."""
        import re
        
        # Simple pattern to match table references
        patterns = [
            r'FROM\\s+([\\w]+)',
            r'JOIN\\s+([\\w]+)',
            r'UPDATE\\s+([\\w]+)',
            r'INSERT\\s+INTO\\s+([\\w]+)'
        ]
        
        tables = set()
        for pattern in patterns:
            matches = re.findall(pattern, sql, re.IGNORECASE)
            tables.update(matches)
        
        return list(tables)
    
    def _extract_column_references(self, sql: str) -> List[str]:
        """Extract column references (simplified)."""
        import re
        
        # This is a simplified extraction - real-world would need SQL parsing
        # Look for patterns after SELECT, WHERE, GROUP BY, ORDER BY
        patterns = [
            r'SELECT\\s+([^FROM]+)',
            r'WHERE\\s+([^GROUP|ORDER|HAVING|LIMIT]+)',
            r'GROUP\\s+BY\\s+([^ORDER|HAVING|LIMIT]+)',
            r'ORDER\\s+BY\\s+([^LIMIT]+)'
        ]
        
        columns = set()
        for pattern in patterns:
            matches = re.findall(pattern, sql, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Simple tokenization - split by comma and clean up
                cols = [col.strip() for col in match.split(',')]
                columns.update([col for col in cols if col and not col.upper().startswith('FROM')])
        
        return list(columns)[:20]  # Limit to prevent memory issues
    
    def _extract_literals(self, sql: str) -> List[str]:
        """Extract string and numeric literals."""
        import re
        
        # Extract string literals
        string_literals = re.findall(r"'([^']*)'", sql)
        numeric_literals = re.findall(r'\\b\\d+(?:\\.\\d+)?\\b', sql)
        
        return string_literals[:10] + numeric_literals[:10]  # Limit results


class PatternMiner:
    """Mines patterns from query history and performance data."""
    
    def __init__(self):
        self.feature_extractor = QueryFeatureExtractor()
        self.pattern_threshold = 0.7  # Minimum similarity for pattern recognition
        self.min_frequency = 3  # Minimum frequency to consider as pattern
    
    def discover_patterns(self, query_history: List[Dict]) -> List[QueryPattern]:
        """Discover patterns from query execution history."""
        patterns = []
        
        if len(query_history) < self.min_frequency:
            logger.warning(f"Insufficient query history ({len(query_history)} queries) for pattern mining")
            return patterns
        
        # Extract features for all queries
        query_features = []
        for query_data in query_history:
            features = self.feature_extractor.extract_features(query_data.get('sql', ''))
            features['performance'] = query_data.get('execution_time_ms', 0)
            features['success'] = query_data.get('success', True)
            features['timestamp'] = query_data.get('timestamp', datetime.utcnow().isoformat())
            query_features.append((query_data, features))
        
        # Group similar queries
        similarity_groups = self._group_by_similarity(query_features)
        
        # Create patterns from groups
        for group_id, group_queries in enumerate(similarity_groups):
            if len(group_queries) >= self.min_frequency:
                pattern = self._create_pattern_from_group(group_id, group_queries)
                if pattern:
                    patterns.append(pattern)
        
        return patterns
    
    def _group_by_similarity(self, query_features: List[Tuple]) -> List[List[Tuple]]:
        """Group queries by feature similarity."""
        groups = []
        
        for query_data, features in query_features:
            # Find the most similar existing group
            best_group = None
            best_similarity = 0
            
            for group in groups:
                # Calculate similarity with group representative (first query)
                if group:
                    _, group_features = group[0]
                    similarity = self._calculate_similarity(features, group_features)
                    
                    if similarity > best_similarity and similarity > self.pattern_threshold:
                        best_similarity = similarity
                        best_group = group
            
            if best_group is not None:
                best_group.append((query_data, features))
            else:
                groups.append([(query_data, features)])
        
        return groups
    
    def _calculate_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity between two feature sets."""
        # Structural similarity
        structural_features = ['token_count', 'parentheses_depth', 'subquery_count', 'where_conditions']
        structural_sim = 0
        
        for feature in structural_features:
            val1 = features1.get(feature, 0)
            val2 = features2.get(feature, 0)
            if val1 == 0 and val2 == 0:
                structural_sim += 1
            elif val1 != 0 and val2 != 0:
                structural_sim += 1 - abs(val1 - val2) / max(val1, val2)
        
        structural_sim /= len(structural_features)
        
        # Semantic similarity
        semantic_features = ['has_aggregation', 'has_join', 'has_subquery', 'has_grouping', 'has_ordering']
        semantic_sim = sum(
            features1.get(feature, False) == features2.get(feature, False) 
            for feature in semantic_features
        ) / len(semantic_features)
        
        # Keyword similarity (Jaccard similarity)
        keywords1 = set(k for k, v in features1.get('keyword_counts', {}).items() if v > 0)
        keywords2 = set(k for k, v in features2.get('keyword_counts', {}).items() if v > 0)
        
        if keywords1 or keywords2:
            keyword_sim = len(keywords1.intersection(keywords2)) / len(keywords1.union(keywords2))
        else:
            keyword_sim = 1
        
        # Weighted average
        return 0.4 * structural_sim + 0.4 * semantic_sim + 0.2 * keyword_sim
    
    def _create_pattern_from_group(self, group_id: int, group_queries: List[Tuple]) -> Optional[QueryPattern]:
        """Create a pattern from a group of similar queries."""
        if not group_queries:
            return None
        
        # Calculate aggregate statistics
        performance_times = [features['performance'] for _, features in group_queries if features['performance'] > 0]
        success_count = sum(1 for _, features in group_queries if features['success'])
        
        # Get representative features from the most common query structure
        representative_features = self._get_representative_features(group_queries)
        
        # Generate pattern description
        description = self._generate_pattern_description(representative_features)
        
        # Determine pattern type
        pattern_type = self._classify_pattern_type(representative_features)
        
        pattern_id = f"pattern_{group_id}_{hashlib.md5(description.encode()).hexdigest()[:8]}"
        
        return QueryPattern(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            description=description,
            frequency=len(group_queries),
            success_rate=success_count / len(group_queries),
            avg_performance_ms=np.mean(performance_times) if performance_times else 0,
            user_feedback_score=0.0,  # Will be updated with user feedback
            confidence=min(len(group_queries) / 10, 1.0),  # Confidence based on frequency
            examples=[query_data.get('sql', '')[:200] + '...' for query_data, _ in group_queries[:3]],
            created_at=datetime.utcnow().isoformat(),
            last_seen=max(features.get('timestamp', '') for _, features in group_queries)
        )
    
    def _get_representative_features(self, group_queries: List[Tuple]) -> Dict:
        """Get representative features from a group of queries."""
        if not group_queries:
            return {}
        
        # Use the first query's features as base, then aggregate others
        _, base_features = group_queries[0]
        representative = base_features.copy()
        
        # For numeric features, use average
        numeric_features = ['token_count', 'parentheses_depth', 'subquery_count', 'where_conditions']
        for feature in numeric_features:
            values = [features.get(feature, 0) for _, features in group_queries]
            representative[feature] = np.mean(values)
        
        # For boolean features, use majority vote
        boolean_features = ['has_aggregation', 'has_join', 'has_subquery', 'has_grouping', 'has_ordering']
        for feature in boolean_features:
            values = [features.get(feature, False) for _, features in group_queries]
            representative[feature] = sum(values) > len(values) / 2
        
        return representative
    
    def _generate_pattern_description(self, features: Dict) -> str:
        """Generate human-readable pattern description."""
        description_parts = []
        
        # Query type
        if features.get('has_aggregation'):
            description_parts.append("aggregation query")
        elif features.get('has_join'):
            description_parts.append("join query")
        elif features.get('has_subquery'):
            description_parts.append("subquery pattern")
        else:
            description_parts.append("simple query")
        
        # Complexity indicators
        if features.get('token_count', 0) > 50:
            description_parts.append("complex structure")
        
        if features.get('subquery_count', 0) > 2:
            description_parts.append("nested subqueries")
        
        if features.get('has_grouping') and features.get('has_ordering'):
            description_parts.append("with grouping and sorting")
        elif features.get('has_grouping'):
            description_parts.append("with grouping")
        elif features.get('has_ordering'):
            description_parts.append("with sorting")
        
        return " ".join(description_parts).capitalize()
    
    def _classify_pattern_type(self, features: Dict) -> str:
        """Classify the type of pattern."""
        if features.get('has_aggregation') and features.get('has_grouping'):
            return "analytical"
        elif features.get('has_join') and features.get('subquery_count', 0) > 1:
            return "complex_relational"
        elif features.get('has_subquery'):
            return "hierarchical"
        elif features.get('has_join'):
            return "relational"
        else:
            return "simple"


class AdaptiveLearningEngine:
    """Main engine for adaptive learning and recommendation generation."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else Path("./adaptive_learning_data")
        self.storage_path.mkdir(exist_ok=True, parents=True)
        
        self.pattern_miner = PatternMiner()
        self.learned_patterns: List[QueryPattern] = []
        self.user_feedback: Dict[str, List[Dict]] = defaultdict(list)
        
        self._load_learned_data()
    
    def learn_from_queries(self, query_history: List[Dict]) -> List[QueryPattern]:
        """Learn new patterns from query execution history."""
        try:
            logger.info(f"Learning from {len(query_history)} queries")
            
            # Discover new patterns
            new_patterns = self.pattern_miner.discover_patterns(query_history)
            
            # Update existing patterns or add new ones
            self._update_patterns(new_patterns)
            
            # Save learned data
            self._save_learned_data()
            
            logger.info(f"Learned {len(new_patterns)} new patterns, total patterns: {len(self.learned_patterns)}")
            return new_patterns
            
        except Exception as e:
            logger.error(f"Error in adaptive learning: {e}")
            return []
    
    def generate_insights(self, current_sql: str, context: Optional[Dict] = None) -> List[LearningInsight]:
        """Generate insights based on learned patterns for a current query."""
        insights = []
        
        try:
            # Extract features from current query
            current_features = self.pattern_miner.feature_extractor.extract_features(current_sql)
            
            # Find matching patterns
            matching_patterns = []
            for pattern in self.learned_patterns:
                # Simple matching based on pattern description keywords
                similarity_score = self._calculate_pattern_match(current_features, pattern)
                if similarity_score > 0.6:
                    matching_patterns.append((pattern, similarity_score))
            
            # Sort by similarity and confidence
            matching_patterns.sort(key=lambda x: x[1] * x[0].confidence, reverse=True)
            
            # Generate insights from top matches
            for pattern, similarity in matching_patterns[:5]:
                insight = self._create_insight_from_pattern(pattern, similarity, current_features)
                if insight:
                    insights.append(insight)
            
            # Add performance insights
            performance_insights = self._generate_performance_insights(current_features, matching_patterns)
            insights.extend(performance_insights)
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
        
        return insights
    
    def record_feedback(self, query_id: str, feedback_type: str, feedback_data: Dict):
        """Record user feedback to improve learning."""
        feedback_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': feedback_type,
            'data': feedback_data
        }
        
        self.user_feedback[query_id].append(feedback_record)
        
        # Update pattern scores based on feedback
        self._update_pattern_scores_from_feedback(query_id, feedback_record)
        
        # Save updated data
        self._save_learned_data()
    
    def get_learning_statistics(self) -> Dict:
        """Get statistics about the learning system."""
        pattern_types = Counter(p.pattern_type for p in self.learned_patterns)
        
        avg_confidence = np.mean([p.confidence for p in self.learned_patterns]) if self.learned_patterns else 0
        avg_success_rate = np.mean([p.success_rate for p in self.learned_patterns]) if self.learned_patterns else 0
        
        return {
            'total_patterns': len(self.learned_patterns),
            'pattern_types': dict(pattern_types),
            'avg_confidence': round(avg_confidence, 3),
            'avg_success_rate': round(avg_success_rate, 3),
            'total_feedback_records': sum(len(feedback) for feedback in self.user_feedback.values()),
            'high_confidence_patterns': len([p for p in self.learned_patterns if p.confidence > 0.8]),
            'learning_age_days': (datetime.utcnow() - datetime.fromisoformat(
                min([p.created_at for p in self.learned_patterns], default=datetime.utcnow().isoformat())
            )).days if self.learned_patterns else 0
        }
    
    def _update_patterns(self, new_patterns: List[QueryPattern]):
        """Update existing patterns or add new ones."""
        existing_ids = {p.pattern_id for p in self.learned_patterns}
        
        for new_pattern in new_patterns:
            if new_pattern.pattern_id in existing_ids:
                # Update existing pattern
                for i, existing_pattern in enumerate(self.learned_patterns):
                    if existing_pattern.pattern_id == new_pattern.pattern_id:
                        # Update frequency and performance
                        self.learned_patterns[i].frequency += new_pattern.frequency
                        self.learned_patterns[i].last_seen = new_pattern.last_seen
                        
                        # Update performance (weighted average)
                        total_freq = self.learned_patterns[i].frequency
                        old_weight = total_freq - new_pattern.frequency
                        self.learned_patterns[i].avg_performance_ms = (
                            (old_weight * self.learned_patterns[i].avg_performance_ms + 
                             new_pattern.frequency * new_pattern.avg_performance_ms) / total_freq
                        )
                        break
            else:
                # Add new pattern
                self.learned_patterns.append(new_pattern)
    
    def _calculate_pattern_match(self, current_features: Dict, pattern: QueryPattern) -> float:
        """Calculate how well current query matches a learned pattern."""
        # Simple matching based on description keywords and structural features
        # In a real implementation, you'd store and compare the original pattern features
        
        score = 0.0
        
        # Check structural similarity (approximation)
        if pattern.description:
            description_lower = pattern.description.lower()
            if 'aggregation' in description_lower and current_features.get('has_aggregation'):
                score += 0.3
            if 'join' in description_lower and current_features.get('has_join'):
                score += 0.3
            if 'subquery' in description_lower and current_features.get('has_subquery'):
                score += 0.3
            if 'complex' in description_lower and current_features.get('token_count', 0) > 50:
                score += 0.2
        
        return min(score, 1.0)
    
    def _create_insight_from_pattern(self, pattern: QueryPattern, similarity: float, 
                                   current_features: Dict) -> Optional[LearningInsight]:
        """Create an insight based on a matching pattern."""
        if pattern.confidence < 0.5:
            return None
        
        # Generate recommendation based on pattern performance
        if pattern.avg_performance_ms > 1000:
            recommendation = f"Similar {pattern.description} queries typically take {pattern.avg_performance_ms:.0f}ms. Consider optimization."
        elif pattern.success_rate < 0.9:
            recommendation = f"This {pattern.description} pattern has {(1-pattern.success_rate)*100:.1f}% failure rate. Review for potential issues."
        else:
            recommendation = f"This {pattern.description} pattern typically performs well (avg: {pattern.avg_performance_ms:.0f}ms)."
        
        return LearningInsight(
            type='pattern_match',
            confidence=similarity * pattern.confidence,
            description=f"Query matches learned pattern: {pattern.description}",
            recommendation=recommendation,
            supporting_evidence=[f"Pattern seen {pattern.frequency} times", f"Success rate: {pattern.success_rate:.2%}"],
            impact_score=pattern.frequency / 100  # Normalize by frequency
        )
    
    def _generate_performance_insights(self, current_features: Dict, 
                                     matching_patterns: List[Tuple]) -> List[LearningInsight]:
        """Generate performance insights based on learned patterns."""
        insights = []
        
        if not matching_patterns:
            return insights
        
        # Average performance of similar patterns
        avg_perf = np.mean([pattern.avg_performance_ms for pattern, _ in matching_patterns])
        
        if avg_perf > 2000:  # 2 seconds
            insights.append(LearningInsight(
                type='performance_warning',
                confidence=0.8,
                description=f"Similar queries average {avg_perf:.0f}ms execution time",
                recommendation="Consider adding indexes or optimizing query structure based on learned patterns",
                supporting_evidence=[f"Based on {len(matching_patterns)} similar patterns"],
                impact_score=min(avg_perf / 1000, 10)  # Scale by seconds
            ))
        
        return insights
    
    def _update_pattern_scores_from_feedback(self, query_id: str, feedback: Dict):
        """Update pattern scores based on user feedback."""
        if feedback['type'] == 'performance_rating':
            rating = feedback['data'].get('rating', 0)
            # Find patterns that might have influenced this query and update their scores
            # This is a simplified implementation
            for pattern in self.learned_patterns[-10:]:  # Check recent patterns
                pattern.user_feedback_score = (pattern.user_feedback_score + rating) / 2
    
    def _save_learned_data(self):
        """Save learned patterns and feedback to disk."""
        try:
            # Save patterns
            patterns_file = self.storage_path / "learned_patterns.json"
            with open(patterns_file, 'w') as f:
                json.dump([asdict(p) for p in self.learned_patterns], f, indent=2)
            
            # Save feedback
            feedback_file = self.storage_path / "user_feedback.json"
            with open(feedback_file, 'w') as f:
                json.dump(dict(self.user_feedback), f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving learned data: {e}")
    
    def _load_learned_data(self):
        """Load previously learned patterns and feedback."""
        try:
            # Load patterns
            patterns_file = self.storage_path / "learned_patterns.json"
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    pattern_data = json.load(f)
                    self.learned_patterns = [QueryPattern(**p) for p in pattern_data]
            
            # Load feedback
            feedback_file = self.storage_path / "user_feedback.json"
            if feedback_file.exists():
                with open(feedback_file, 'r') as f:
                    self.user_feedback = defaultdict(list, json.load(f))
                    
        except Exception as e:
            logger.error(f"Error loading learned data: {e}")