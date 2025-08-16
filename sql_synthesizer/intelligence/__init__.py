"""
Intelligence Module - Advanced Query Analysis and Optimization

Provides sophisticated NLP-driven insights and adaptive learning capabilities
for SQL query optimization and pattern recognition.
"""

from .adaptive_learning import AdaptiveLearningEngine, LearningInsight, QueryPattern
from .intelligent_cache import CacheInsight, CacheStrategy, IntelligentCacheManager
from .query_insights import (
    QueryComplexity,
    QueryComplexityAnalyzer,
    QueryInsight,
    QueryInsightsEngine,
    QueryPatternAnalyzer,
)

__all__ = [
    "QueryInsight",
    "QueryComplexity",
    "QueryPatternAnalyzer",
    "QueryComplexityAnalyzer",
    "QueryInsightsEngine",
    "AdaptiveLearningEngine",
    "QueryPattern",
    "LearningInsight",
    "IntelligentCacheManager",
    "CacheStrategy",
    "CacheInsight",
]
