"""
Intelligent Caching System with Pattern Recognition
Advanced caching that learns from query patterns and user behavior to optimize hit rates.
"""

import hashlib
import logging
import threading
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Different caching strategies based on query patterns."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Learned from patterns
    SEMANTIC = "semantic"  # Based on query similarity
    PREDICTIVE = "predictive"  # Proactive caching


@dataclass
class CacheInsight:
    """Insight about caching performance and recommendations."""

    type: str
    description: str
    recommendation: str
    impact_estimate: str
    confidence: float
    supporting_data: Dict[str, Any]


@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata for intelligent decisions."""

    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl_seconds: Optional[int]
    query_hash: str
    query_pattern: Optional[str]
    estimated_compute_cost: float  # Time to regenerate
    access_pattern: List[float]  # Recent access timestamps
    user_context: Optional[Dict] = None


@dataclass
class QueryContext:
    """Context information for intelligent caching decisions."""

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    time_of_day: int = 0  # Hour 0-23
    day_of_week: int = 0  # 0=Monday
    query_type: Optional[str] = None
    table_dependencies: List[str] = None
    estimated_cost: float = 0.0


class SemanticCacheIndex:
    """Index for semantic similarity-based cache lookups."""

    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.query_vectors: Dict[str, Dict] = {}  # Simple feature vectors
        self.lock = threading.RLock()

    def add_query(self, query_hash: str, sql: str, features: Dict[str, Any]):
        """Add query features to semantic index."""
        with self.lock:
            self.query_vectors[query_hash] = {
                "sql": sql,
                "features": features,
                "normalized_features": self._normalize_features(features),
            }

    def find_similar_queries(
        self, sql: str, features: Dict[str, Any], limit: int = 5
    ) -> List[Tuple[str, float]]:
        """Find semantically similar cached queries."""
        normalized_features = self._normalize_features(features)
        similar_queries = []

        with self.lock:
            for query_hash, data in self.query_vectors.items():
                similarity = self._calculate_similarity(
                    normalized_features, data["normalized_features"]
                )

                if similarity >= self.similarity_threshold:
                    similar_queries.append((query_hash, similarity))

        # Sort by similarity and return top results
        similar_queries.sort(key=lambda x: x[1], reverse=True)
        return similar_queries[:limit]

    def _normalize_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Normalize features for similarity comparison."""
        normalized = {}

        # Numeric features - normalize to 0-1 range
        numeric_features = [
            "token_count",
            "parentheses_depth",
            "subquery_count",
            "where_conditions",
        ]
        for feature in numeric_features:
            value = features.get(feature, 0)
            # Simple normalization - in production you'd use learned min/max
            normalized[feature] = min(value / 100, 1.0)

        # Boolean features - convert to 0/1
        boolean_features = [
            "has_aggregation",
            "has_join",
            "has_subquery",
            "has_grouping",
            "has_ordering",
        ]
        for feature in boolean_features:
            normalized[feature] = 1.0 if features.get(feature, False) else 0.0

        # Keyword counts - normalize by total
        keyword_counts = features.get("keyword_counts", {})
        total_keywords = sum(keyword_counts.values()) if keyword_counts else 1
        for keyword, count in keyword_counts.items():
            normalized[f"kw_{keyword}"] = count / total_keywords

        return normalized

    def _calculate_similarity(
        self, features1: Dict[str, float], features2: Dict[str, float]
    ) -> float:
        """Calculate cosine similarity between normalized feature vectors."""
        # Get common features
        common_features = set(features1.keys()) & set(features2.keys())
        if not common_features:
            return 0.0

        # Calculate cosine similarity
        dot_product = sum(features1[f] * features2[f] for f in common_features)
        norm1 = np.sqrt(sum(features1[f] ** 2 for f in common_features))
        norm2 = np.sqrt(sum(features2[f] ** 2 for f in common_features))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


class AccessPatternPredictor:
    """Predicts future access patterns based on historical data."""

    def __init__(self, lookback_hours: int = 24):
        self.lookback_hours = lookback_hours
        self.access_history: Dict[str, List[float]] = defaultdict(list)
        self.pattern_models: Dict[str, Dict] = {}
        self.lock = threading.RLock()

    def record_access(self, query_hash: str, timestamp: Optional[float] = None):
        """Record query access for pattern learning."""
        if timestamp is None:
            timestamp = time.time()

        with self.lock:
            self.access_history[query_hash].append(timestamp)

            # Keep only recent history
            cutoff = timestamp - (self.lookback_hours * 3600)
            self.access_history[query_hash] = [
                ts for ts in self.access_history[query_hash] if ts >= cutoff
            ]

    def predict_next_access(self, query_hash: str) -> Tuple[float, float]:
        """Predict probability and time of next access."""
        with self.lock:
            history = self.access_history.get(query_hash, [])

            if len(history) < 2:
                return 0.1, 0.0  # Low probability, no time estimate

            # Simple pattern analysis - in production you'd use more sophisticated models
            recent_history = history[-10:]  # Last 10 accesses

            # Calculate average interval between accesses
            intervals = [
                recent_history[i] - recent_history[i - 1]
                for i in range(1, len(recent_history))
            ]

            if not intervals:
                return 0.1, 0.0

            avg_interval = np.mean(intervals)
            interval_std = (
                np.std(intervals) if len(intervals) > 1 else avg_interval * 0.5
            )

            # Time since last access
            time_since_last = time.time() - recent_history[-1]

            # Probability based on typical interval
            if avg_interval > 0:
                probability = max(0.1, min(0.9, time_since_last / avg_interval))
            else:
                probability = 0.1

            # Estimate next access time
            next_access_estimate = avg_interval - time_since_last

            return probability, next_access_estimate

    def get_access_patterns(self, query_hash: str) -> Dict[str, Any]:
        """Get detailed access pattern analysis."""
        with self.lock:
            history = self.access_history.get(query_hash, [])

            if len(history) < 2:
                return {"insufficient_data": True}

            # Analyze patterns
            now = time.time()
            recent_accesses = [ts for ts in history if now - ts <= 3600]  # Last hour
            daily_accesses = [ts for ts in history if now - ts <= 86400]  # Last day

            # Time-of-day pattern
            hours = [(datetime.fromtimestamp(ts).hour) for ts in history]
            hour_distribution = Counter(hours)

            # Day-of-week pattern
            days = [datetime.fromtimestamp(ts).weekday() for ts in history]
            day_distribution = Counter(days)

            return {
                "total_accesses": len(history),
                "recent_accesses_1h": len(recent_accesses),
                "recent_accesses_24h": len(daily_accesses),
                "peak_hours": [h for h, count in hour_distribution.most_common(3)],
                "peak_days": [d for d, count in day_distribution.most_common(3)],
                "access_frequency_per_hour": len(history) / max(self.lookback_hours, 1),
                "last_access": history[-1] if history else 0,
            }


class IntelligentCacheManager:
    """Advanced caching system with learning capabilities."""

    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl

        # Storage
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order = OrderedDict()  # For LRU
        self.access_frequency: Counter = Counter()  # For LFU

        # Intelligence components
        self.semantic_index = SemanticCacheIndex()
        self.pattern_predictor = AccessPatternPredictor()

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "semantic_hits": 0,
            "predictive_hits": 0,
            "total_requests": 0,
        }

        # Thread safety
        self.lock = threading.RLock()

        # Learning parameters
        self.learning_enabled = True
        self.min_confidence_threshold = 0.7
        self.semantic_cache_enabled = True
        self.predictive_cache_enabled = True

    def get(self, key: str, context: Optional[QueryContext] = None) -> Optional[Any]:
        """Intelligent cache retrieval with multiple strategies."""
        with self.lock:
            self.stats["total_requests"] += 1

            # Direct cache hit
            if key in self.cache:
                entry = self.cache[key]
                if self._is_valid_entry(entry):
                    self._update_access_stats(key, entry)
                    self.stats["hits"] += 1
                    return entry.value
                else:
                    # Entry expired or invalid
                    self._remove_entry(key)

            # Semantic similarity search
            if self.semantic_cache_enabled and context:
                semantic_result = self._semantic_lookup(key, context)
                if semantic_result:
                    self.stats["semantic_hits"] += 1
                    self.stats["hits"] += 1
                    return semantic_result

            # Predictive cache check
            if self.predictive_cache_enabled:
                predictive_result = self._predictive_lookup(key, context)
                if predictive_result:
                    self.stats["predictive_hits"] += 1
                    self.stats["hits"] += 1
                    return predictive_result

            self.stats["misses"] += 1
            return None

    def put(
        self,
        key: str,
        value: Any,
        context: Optional[QueryContext] = None,
        ttl: Optional[int] = None,
        estimated_cost: float = 1.0,
    ):
        """Intelligent cache storage with adaptive strategies."""
        with self.lock:
            current_time = time.time()

            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                access_count=1,
                ttl_seconds=ttl or self.default_ttl,
                query_hash=self._generate_query_hash(key),
                query_pattern=context.query_type if context else None,
                estimated_compute_cost=estimated_cost,
                access_pattern=[current_time],
                user_context=asdict(context) if context else None,
            )

            # Eviction if necessary
            self._ensure_space_available()

            # Store entry
            self.cache[key] = entry
            self.access_order[key] = current_time
            self.access_frequency[key] += 1

            # Update semantic index
            if context and hasattr(context, "query_features"):
                self.semantic_index.add_query(
                    entry.query_hash, key, getattr(context, "query_features", {})
                )

            # Record for pattern learning
            if self.learning_enabled:
                self.pattern_predictor.record_access(entry.query_hash, current_time)

    def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching a pattern (e.g., table name)."""
        with self.lock:
            keys_to_remove = []

            for key, entry in self.cache.items():
                if self._matches_invalidation_pattern(key, entry, pattern):
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                self._remove_entry(key)

            logger.info(
                f"Invalidated {len(keys_to_remove)} cache entries matching pattern: {pattern}"
            )

    def get_insights(self) -> List[CacheInsight]:
        """Generate insights about cache performance and optimization opportunities."""
        insights = []

        with self.lock:
            total_requests = self.stats["total_requests"]
            if total_requests == 0:
                return insights

            hit_rate = self.stats["hits"] / total_requests
            semantic_hit_rate = self.stats["semantic_hits"] / total_requests

            # Hit rate analysis
            if hit_rate < 0.3:
                insights.append(
                    CacheInsight(
                        type="performance",
                        description=f"Low cache hit rate ({hit_rate:.2%})",
                        recommendation="Consider increasing cache size or TTL values",
                        impact_estimate="20-50% performance improvement potential",
                        confidence=0.9,
                        supporting_data={
                            "current_hit_rate": hit_rate,
                            "cache_size": len(self.cache),
                        },
                    )
                )

            # Semantic caching effectiveness
            if semantic_hit_rate > 0.05:  # 5% of requests
                insights.append(
                    CacheInsight(
                        type="optimization",
                        description=f"Semantic caching is effective ({semantic_hit_rate:.2%} of requests)",
                        recommendation="Consider expanding semantic similarity threshold",
                        impact_estimate="10-20% additional hit rate improvement",
                        confidence=0.8,
                        supporting_data={"semantic_hit_rate": semantic_hit_rate},
                    )
                )

            # Cache size utilization
            utilization = len(self.cache) / self.max_size
            if utilization > 0.9:
                insights.append(
                    CacheInsight(
                        type="capacity",
                        description=f"High cache utilization ({utilization:.1%})",
                        recommendation="Consider increasing cache size or optimizing eviction policy",
                        impact_estimate="Prevent premature evictions",
                        confidence=0.9,
                        supporting_data={
                            "utilization": utilization,
                            "evictions": self.stats["evictions"],
                        },
                    )
                )

            # Access pattern insights
            pattern_insights = self._analyze_access_patterns()
            insights.extend(pattern_insights)

        return insights

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self.lock:
            total_requests = self.stats["total_requests"]

            return {
                "requests": {
                    "total": total_requests,
                    "hits": self.stats["hits"],
                    "misses": self.stats["misses"],
                    "hit_rate": self.stats["hits"] / max(total_requests, 1),
                },
                "semantic_caching": {
                    "semantic_hits": self.stats["semantic_hits"],
                    "semantic_hit_rate": self.stats["semantic_hits"]
                    / max(total_requests, 1),
                },
                "predictive_caching": {
                    "predictive_hits": self.stats["predictive_hits"],
                    "predictive_hit_rate": self.stats["predictive_hits"]
                    / max(total_requests, 1),
                },
                "capacity": {
                    "current_size": len(self.cache),
                    "max_size": self.max_size,
                    "utilization": len(self.cache) / self.max_size,
                    "evictions": self.stats["evictions"],
                },
                "entries": {
                    "avg_access_count": (
                        np.mean([e.access_count for e in self.cache.values()])
                        if self.cache
                        else 0
                    ),
                    "avg_age_seconds": (
                        np.mean(
                            [time.time() - e.created_at for e in self.cache.values()]
                        )
                        if self.cache
                        else 0
                    ),
                },
            }

    def _is_valid_entry(self, entry: CacheEntry) -> bool:
        """Check if cache entry is still valid."""
        if entry.ttl_seconds is None:
            return True

        return (time.time() - entry.created_at) < entry.ttl_seconds

    def _update_access_stats(self, key: str, entry: CacheEntry):
        """Update access statistics for cache entry."""
        current_time = time.time()
        entry.last_accessed = current_time
        entry.access_count += 1
        entry.access_pattern.append(current_time)

        # Keep only recent access pattern history
        cutoff = current_time - 86400  # 24 hours
        entry.access_pattern = [ts for ts in entry.access_pattern if ts >= cutoff]

        # Update access order and frequency
        self.access_order.move_to_end(key)
        self.access_frequency[key] += 1

        # Record for pattern learning
        if self.learning_enabled:
            self.pattern_predictor.record_access(entry.query_hash, current_time)

    def _semantic_lookup(self, key: str, context: QueryContext) -> Optional[Any]:
        """Attempt semantic similarity-based cache lookup."""
        if not hasattr(context, "query_features"):
            return None

        query_features = getattr(context, "query_features", {})
        similar_queries = self.semantic_index.find_similar_queries(key, query_features)

        for query_hash, similarity in similar_queries:
            # Find cache entries with this query hash
            for entry_key, entry in self.cache.items():
                if (
                    entry.query_hash == query_hash
                    and self._is_valid_entry(entry)
                    and similarity >= self.min_confidence_threshold
                ):

                    # Update access stats for semantic hit
                    self._update_access_stats(entry_key, entry)
                    logger.debug(f"Semantic cache hit: {similarity:.3f} similarity")
                    return entry.value

        return None

    def _predictive_lookup(
        self, key: str, context: Optional[QueryContext]
    ) -> Optional[Any]:
        """Attempt predictive cache lookup based on patterns."""
        # Simple predictive logic - in production this would be more sophisticated
        query_hash = self._generate_query_hash(key)
        probability, _ = self.pattern_predictor.predict_next_access(query_hash)

        if probability > 0.8:  # High probability prediction
            # Check if we have a similar entry that might satisfy this request
            for entry_key, entry in self.cache.items():
                if entry.query_hash == query_hash and self._is_valid_entry(entry):

                    self._update_access_stats(entry_key, entry)
                    logger.debug(f"Predictive cache hit: {probability:.3f} probability")
                    return entry.value

        return None

    def _ensure_space_available(self):
        """Ensure space is available for new entry using intelligent eviction."""
        while len(self.cache) >= self.max_size:
            victim_key = self._select_eviction_victim()
            if victim_key:
                self._remove_entry(victim_key)
                self.stats["evictions"] += 1
            else:
                break  # Safety break

    def _select_eviction_victim(self) -> Optional[str]:
        """Select entry for eviction using adaptive strategy."""
        if not self.cache:
            return None

        # Score-based eviction considering multiple factors
        scores = {}
        current_time = time.time()

        for key, entry in self.cache.items():
            # Base factors
            age_score = (current_time - entry.created_at) / 86400  # Age in days
            access_frequency_score = 1 / (entry.access_count + 1)
            recency_score = (
                current_time - entry.last_accessed
            ) / 3600  # Hours since last access

            # Cost consideration - less likely to evict expensive-to-compute entries
            cost_score = 1 / (entry.estimated_compute_cost + 1)

            # Predictive score - less likely to evict entries predicted to be accessed soon
            query_hash = entry.query_hash
            access_probability, _ = self.pattern_predictor.predict_next_access(
                query_hash
            )
            prediction_score = 1 - access_probability

            # Combined score (higher = more likely to evict)
            combined_score = (
                0.3 * age_score
                + 0.2 * access_frequency_score
                + 0.25 * recency_score
                + 0.1 * cost_score
                + 0.15 * prediction_score
            )

            scores[key] = combined_score

        # Return key with highest eviction score
        return max(scores.keys(), key=lambda k: scores[k])

    def _remove_entry(self, key: str):
        """Remove entry from cache and all associated indices."""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_order:
            del self.access_order[key]
        if key in self.access_frequency:
            del self.access_frequency[key]

    def _generate_query_hash(self, query: str) -> str:
        """Generate hash for query to group similar queries."""
        # Normalize query for better grouping
        normalized = query.strip().lower()
        # Remove extra whitespace
        normalized = " ".join(normalized.split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _matches_invalidation_pattern(
        self, key: str, entry: CacheEntry, pattern: str
    ) -> bool:
        """Check if entry matches invalidation pattern."""
        # Simple pattern matching - in production you'd use more sophisticated logic
        return pattern.lower() in key.lower() or (
            entry.user_context and pattern.lower() in str(entry.user_context).lower()
        )

    def _analyze_access_patterns(self) -> List[CacheInsight]:
        """Analyze access patterns and generate insights."""
        insights = []

        if len(self.cache) < 10:  # Need sufficient data
            return insights

        # Analyze temporal patterns
        current_hour = datetime.now().hour
        recent_accesses = []
        old_accesses = []

        for entry in self.cache.values():
            if entry.access_pattern:
                recent_pattern = [
                    ts for ts in entry.access_pattern if time.time() - ts < 3600
                ]
                if recent_pattern:
                    recent_accesses.extend(recent_pattern)
                else:
                    old_accesses.extend(entry.access_pattern[-1:])

        if len(recent_accesses) > len(old_accesses) * 2:
            insights.append(
                CacheInsight(
                    type="temporal",
                    description="High recent activity detected",
                    recommendation="Consider shorter TTL for better freshness",
                    impact_estimate="Improved data freshness",
                    confidence=0.7,
                    supporting_data={
                        "recent_accesses": len(recent_accesses),
                        "older_accesses": len(old_accesses),
                    },
                )
            )

        return insights
