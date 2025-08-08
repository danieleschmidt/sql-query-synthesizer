"""
Intelligence Module - Advanced Query Analysis and Optimization

Provides sophisticated NLP-driven insights and adaptive learning capabilities
for SQL query optimization and pattern recognition.
"""

from .query_insights import (
    QueryInsight,
    QueryComplexity,
    QueryPatternAnalyzer,
    QueryComplexityAnalyzer,
    QueryInsightsEngine
)
from .adaptive_learning import AdaptiveLearningEngine, QueryPattern, LearningInsight
from .intelligent_cache import IntelligentCacheManager, CacheStrategy, CacheInsight

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
    "CacheInsight"
]