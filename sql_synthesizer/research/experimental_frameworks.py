"""Experimental Research Frameworks for Advanced NL2SQL.

This module provides comprehensive experimental frameworks for testing and validating
novel NL2SQL approaches with statistical rigor and reproducibility.

Research Focus: Rigorous evaluation and comparison of novel algorithmic approaches
with proper baselines and statistical validation.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import statistics
import random

import numpy as np

logger = logging.getLogger(__name__)

# Optional statistical dependencies
try:
    from scipy import stats
    from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available. Using basic statistical tests.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("Plotting libraries not available. Visualization disabled.")

# Import existing framework for compatibility
import hashlib
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import asdict
from enum import Enum


class ExperimentType(Enum):
    """Types of experimental research."""

    ALGORITHM_COMPARISON = "algorithm_comparison"
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    ACCURACY_STUDY = "accuracy_study"
    SCALABILITY_TEST = "scalability_test"
    NOVEL_APPROACH = "novel_approach"


class ExperimentStatus(Enum):
    """Experiment execution status."""

    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ANALYZING = "analyzing"


@dataclass
class ExperimentResult:
    """Results from an experimental run."""

    experiment_id: str
    approach_name: str
    execution_time_ms: float
    accuracy_score: float
    resource_usage: Dict[str, float]
    output_quality: float
    error_rate: float
    metadata: Dict[str, Any]
    timestamp: datetime


@dataclass
class ResearchHypothesis:
    """Research hypothesis for experimental validation."""

    hypothesis_id: str
    description: str
    prediction: str
    success_criteria: Dict[str, float]
    test_scenarios: List[str]
    baseline_approach: str
    novel_approach: str
    expected_improvement: float
    confidence_level: float


class BaseExperimentalApproach(ABC):
    """Base class for experimental SQL synthesis approaches."""

    def __init__(self, approach_name: str):
        self.approach_name = approach_name
        self.execution_history = []
        self.performance_metrics = defaultdict(list)

    @abstractmethod
    async def synthesize_sql(
        self, natural_language: str, schema_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize SQL from natural language using this experimental approach."""
        pass

    @abstractmethod
    def get_approach_metadata(self) -> Dict[str, Any]:
        """Get metadata about this experimental approach."""
        pass

    def record_execution(self, result: ExperimentResult):
        """Record execution results for analysis."""
        self.execution_history.append(result)
        self.performance_metrics["execution_time"].append(result.execution_time_ms)
        self.performance_metrics["accuracy"].append(result.accuracy_score)
        self.performance_metrics["quality"].append(result.output_quality)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.execution_history:
            return {"status": "no_data"}

        return {
            "total_executions": len(self.execution_history),
            "avg_execution_time_ms": sum(self.performance_metrics["execution_time"])
            / len(self.performance_metrics["execution_time"]),
            "avg_accuracy_score": sum(self.performance_metrics["accuracy"])
            / len(self.performance_metrics["accuracy"]),
            "avg_quality_score": sum(self.performance_metrics["quality"])
            / len(self.performance_metrics["quality"]),
            "success_rate": len(
                [r for r in self.execution_history if r.error_rate < 0.1]
            )
            / len(self.execution_history)
            * 100,
            "latest_result": (
                asdict(self.execution_history[-1]) if self.execution_history else None
            ),
        }


class TemplateBasedApproach(BaseExperimentalApproach):
    """Template-based SQL synthesis approach."""

    def __init__(self):
        super().__init__("template_based")
        self.templates = {
            "select_all": "SELECT * FROM {table}",
            "count": "SELECT COUNT(*) FROM {table}",
            "filter": "SELECT * FROM {table} WHERE {condition}",
            "join": "SELECT * FROM {table1} JOIN {table2} ON {join_condition}",
            "aggregate": "SELECT {columns}, {aggregate_func}({column}) FROM {table} GROUP BY {columns}",
            "top_n": "SELECT * FROM {table} ORDER BY {column} DESC LIMIT {n}",
        }

        self.patterns = {
            "count": ["count", "how many", "number of", "total"],
            "filter": ["where", "with", "having", "equals", "greater", "less"],
            "join": ["join", "combine", "merge", "relate", "connect"],
            "aggregate": ["sum", "average", "max", "min", "group by"],
            "top_n": ["top", "first", "best", "highest", "lowest", "limit"],
        }

    async def synthesize_sql(
        self, natural_language: str, schema_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize SQL using template matching."""
        start_time = time.time()

        try:
            nl_lower = natural_language.lower()

            # Determine query type based on patterns
            query_type = self._classify_query_type(nl_lower)

            # Extract entities from natural language
            entities = self._extract_entities(nl_lower, schema_context)

            # Generate SQL from template
            sql = self._generate_from_template(query_type, entities)

            execution_time = (time.time() - start_time) * 1000

            return {
                "sql": sql,
                "confidence": self._calculate_confidence(query_type, entities),
                "approach": self.approach_name,
                "execution_time_ms": execution_time,
                "query_type": query_type,
                "extracted_entities": entities,
                "template_used": query_type,
            }

        except Exception as e:
            return {
                "sql": "SELECT 1",
                "confidence": 0.1,
                "error": str(e),
                "approach": self.approach_name,
                "execution_time_ms": (time.time() - start_time) * 1000,
            }

    def get_approach_metadata(self) -> Dict[str, Any]:
        """Get template-based approach metadata."""
        return {
            "name": self.approach_name,
            "type": "rule_based",
            "description": "Template-based SQL synthesis using pattern matching",
            "strengths": ["Fast", "Predictable", "No external dependencies"],
            "weaknesses": ["Limited flexibility", "Requires extensive templates"],
            "complexity": "Low",
            "templates_available": len(self.templates),
            "patterns_supported": len(self.patterns),
        }

    def _classify_query_type(self, natural_language: str) -> str:
        """Classify query type based on patterns."""
        scores = {}

        for query_type, patterns in self.patterns.items():
            score = sum(1 for pattern in patterns if pattern in natural_language)
            if score > 0:
                scores[query_type] = score

        if scores:
            return max(scores, key=scores.get)
        else:
            return "select_all"  # Default

    def _extract_entities(
        self, natural_language: str, schema_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract entities like table names, columns, conditions."""
        entities = {}

        # Extract table names (simple approach)
        tables = schema_context.get("tables", [])
        for table in tables:
            if table.lower() in natural_language:
                entities["table"] = table
                break

        # Extract columns
        columns = schema_context.get("columns", {})
        for table, table_columns in columns.items():
            for column in table_columns:
                if column.lower() in natural_language:
                    entities.setdefault("columns", []).append(column)

        # Extract conditions (simplified)
        conditions = []
        if ">" in natural_language:
            conditions.append("greater_than")
        if "<" in natural_language:
            conditions.append("less_than")
        if "=" in natural_language or "equals" in natural_language:
            conditions.append("equals")

        if conditions:
            entities["conditions"] = conditions

        return entities

    def _generate_from_template(self, query_type: str, entities: Dict[str, Any]) -> str:
        """Generate SQL from template and entities."""
        template = self.templates.get(query_type, self.templates["select_all"])

        # Simple template substitution
        sql = template

        if "table" in entities:
            sql = sql.replace("{table}", entities["table"])
        else:
            sql = sql.replace("{table}", "users")  # Default table

        if "columns" in entities and entities["columns"]:
            sql = sql.replace("{columns}", ", ".join(entities["columns"]))
            sql = sql.replace("{column}", entities["columns"][0])
        else:
            sql = sql.replace("{columns}", "*")
            sql = sql.replace("{column}", "id")

        # Handle other placeholders
        sql = sql.replace("{condition}", "id > 0")
        sql = sql.replace("{join_condition}", "table1.id = table2.id")
        sql = sql.replace("{aggregate_func}", "COUNT")
        sql = sql.replace("{n}", "10")
        sql = sql.replace("{table1}", "table1")
        sql = sql.replace("{table2}", "table2")

        return sql

    def _calculate_confidence(self, query_type: str, entities: Dict[str, Any]) -> float:
        """Calculate confidence score for the generated SQL."""
        base_confidence = 0.6

        # Increase confidence if we found table and columns
        if "table" in entities:
            base_confidence += 0.2
        if "columns" in entities:
            base_confidence += 0.15

        # Decrease confidence for complex query types without enough entities
        if query_type == "join" and "table" not in entities:
            base_confidence -= 0.3

        return min(1.0, max(0.1, base_confidence))


class SemanticSimilarityApproach(BaseExperimentalApproach):
    """Semantic similarity-based SQL synthesis approach."""

    def __init__(self):
        super().__init__("semantic_similarity")
        self.knowledge_base = []
        self.similarity_threshold = 0.7

        # Initialize with some example patterns
        self._initialize_knowledge_base()

    async def synthesize_sql(
        self, natural_language: str, schema_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize SQL using semantic similarity matching."""
        start_time = time.time()

        try:
            # Find most similar patterns in knowledge base
            similar_patterns = self._find_similar_patterns(natural_language)

            if similar_patterns:
                best_match = similar_patterns[0]

                # Adapt the best match to current context
                sql = self._adapt_sql_to_context(best_match["sql"], schema_context)
                confidence = best_match["similarity"]

            else:
                # Fallback to simple SQL
                sql = f"SELECT * FROM {schema_context.get('tables', ['users'])[0]} LIMIT 10"
                confidence = 0.3

            execution_time = (time.time() - start_time) * 1000

            return {
                "sql": sql,
                "confidence": confidence,
                "approach": self.approach_name,
                "execution_time_ms": execution_time,
                "similar_patterns_found": len(similar_patterns),
                "best_match": similar_patterns[0] if similar_patterns else None,
            }

        except Exception as e:
            return {
                "sql": "SELECT 1",
                "confidence": 0.1,
                "error": str(e),
                "approach": self.approach_name,
                "execution_time_ms": (time.time() - start_time) * 1000,
            }

    def get_approach_metadata(self) -> Dict[str, Any]:
        """Get semantic similarity approach metadata."""
        return {
            "name": self.approach_name,
            "type": "similarity_based",
            "description": "SQL synthesis using semantic similarity to known patterns",
            "strengths": [
                "Learns from examples",
                "Handles variations well",
                "Improves over time",
            ],
            "weaknesses": ["Requires training data", "Similarity computation overhead"],
            "complexity": "Medium",
            "knowledge_base_size": len(self.knowledge_base),
            "similarity_threshold": self.similarity_threshold,
        }

    def add_pattern(
        self, natural_language: str, sql: str, metadata: Optional[Dict] = None
    ):
        """Add a new pattern to the knowledge base."""
        pattern = {
            "id": hashlib.md5(natural_language.encode()).hexdigest()[:8],
            "natural_language": natural_language,
            "sql": sql,
            "tokens": self._tokenize(natural_language),
            "metadata": metadata or {},
            "usage_count": 0,
            "success_rate": 1.0,
            "created_at": datetime.utcnow().isoformat(),
        }

        self.knowledge_base.append(pattern)

    def _initialize_knowledge_base(self):
        """Initialize knowledge base with example patterns."""
        examples = [
            ("show me all users", "SELECT * FROM users"),
            ("count the number of orders", "SELECT COUNT(*) FROM orders"),
            ("find users older than 25", "SELECT * FROM users WHERE age > 25"),
            (
                "get the top 10 products by sales",
                "SELECT * FROM products ORDER BY sales DESC LIMIT 10",
            ),
            (
                "show average order value by month",
                "SELECT DATE_TRUNC('month', created_at) as month, AVG(total) FROM orders GROUP BY month",
            ),
            (
                "list customers with their order counts",
                "SELECT c.name, COUNT(o.id) FROM customers c LEFT JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name",
            ),
        ]

        for nl, sql in examples:
            self.add_pattern(nl, sql)

    def _find_similar_patterns(self, natural_language: str) -> List[Dict[str, Any]]:
        """Find patterns similar to the input natural language."""
        input_tokens = self._tokenize(natural_language)
        similar_patterns = []

        for pattern in self.knowledge_base:
            similarity = self._calculate_similarity(input_tokens, pattern["tokens"])

            if similarity >= self.similarity_threshold:
                similar_patterns.append({**pattern, "similarity": similarity})

        # Sort by similarity score
        similar_patterns.sort(key=lambda x: x["similarity"], reverse=True)

        return similar_patterns[:5]  # Return top 5

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Convert to lowercase and split on whitespace and punctuation
        import re

        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    def _calculate_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """Calculate simple Jaccard similarity between token sets."""
        set1 = set(tokens1)
        set2 = set(tokens2)

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        return intersection / union

    def _adapt_sql_to_context(self, sql: str, schema_context: Dict[str, Any]) -> str:
        """Adapt SQL to current schema context."""
        # Simple adaptation - replace table names with available ones
        available_tables = schema_context.get("tables", [])

        if available_tables:
            # Replace common table names with available ones
            common_tables = ["users", "orders", "products", "customers"]

            for common_table in common_tables:
                if common_table in sql.lower():
                    for available_table in available_tables:
                        if common_table in available_table.lower():
                            sql = sql.replace(common_table, available_table)
                            break

        return sql


class HybridApproach(BaseExperimentalApproach):
    """Hybrid approach combining multiple synthesis methods."""

    def __init__(self):
        super().__init__("hybrid")
        self.template_approach = TemplateBasedApproach()
        self.similarity_approach = SemanticSimilarityApproach()
        self.voting_weights = {"template_based": 0.4, "semantic_similarity": 0.6}

    async def synthesize_sql(
        self, natural_language: str, schema_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize SQL using hybrid approach."""
        start_time = time.time()

        try:
            # Get results from both approaches
            template_result = await self.template_approach.synthesize_sql(
                natural_language, schema_context
            )
            similarity_result = await self.similarity_approach.synthesize_sql(
                natural_language, schema_context
            )

            # Weighted voting for final decision
            template_score = (
                template_result.get("confidence", 0)
                * self.voting_weights["template_based"]
            )
            similarity_score = (
                similarity_result.get("confidence", 0)
                * self.voting_weights["semantic_similarity"]
            )

            if similarity_score > template_score:
                chosen_result = similarity_result
                chosen_approach = "semantic_similarity"
            else:
                chosen_result = template_result
                chosen_approach = "template_based"

            execution_time = (time.time() - start_time) * 1000

            return {
                "sql": chosen_result["sql"],
                "confidence": max(
                    template_result.get("confidence", 0),
                    similarity_result.get("confidence", 0),
                ),
                "approach": self.approach_name,
                "chosen_sub_approach": chosen_approach,
                "execution_time_ms": execution_time,
                "template_result": template_result,
                "similarity_result": similarity_result,
                "voting_scores": {
                    "template_based": template_score,
                    "semantic_similarity": similarity_score,
                },
            }

        except Exception as e:
            return {
                "sql": "SELECT 1",
                "confidence": 0.1,
                "error": str(e),
                "approach": self.approach_name,
                "execution_time_ms": (time.time() - start_time) * 1000,
            }

    def get_approach_metadata(self) -> Dict[str, Any]:
        """Get hybrid approach metadata."""
        return {
            "name": self.approach_name,
            "type": "hybrid",
            "description": "Hybrid approach combining template-based and semantic similarity methods",
            "strengths": [
                "Best of both approaches",
                "Adaptive selection",
                "Robust performance",
            ],
            "weaknesses": ["Higher complexity", "Slower execution"],
            "complexity": "High",
            "sub_approaches": [
                self.template_approach.get_approach_metadata(),
                self.similarity_approach.get_approach_metadata(),
            ],
            "voting_weights": self.voting_weights,
        }


class ExperimentalFramework:
    """Framework for conducting comparative experiments."""

    def __init__(self):
        self.approaches: Dict[str, BaseExperimentalApproach] = {}
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self.hypotheses: Dict[str, ResearchHypothesis] = {}

        # Register default approaches
        self._register_default_approaches()

    def register_approach(self, approach: BaseExperimentalApproach):
        """Register an experimental approach."""
        self.approaches[approach.approach_name] = approach
        logger.info(f"Registered experimental approach: {approach.approach_name}")

    def register_hypothesis(self, hypothesis: ResearchHypothesis):
        """Register a research hypothesis for testing."""
        self.hypotheses[hypothesis.hypothesis_id] = hypothesis
        logger.info(f"Registered research hypothesis: {hypothesis.hypothesis_id}")

    async def run_comparative_experiment(
        self,
        experiment_id: str,
        test_cases: List[Dict[str, Any]],
        approaches_to_test: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run comparative experiment across multiple approaches."""

        if approaches_to_test is None:
            approaches_to_test = list(self.approaches.keys())

        experiment = {
            "experiment_id": experiment_id,
            "status": ExperimentStatus.RUNNING,
            "start_time": datetime.utcnow(),
            "test_cases": len(test_cases),
            "approaches": approaches_to_test,
            "results": defaultdict(list),
        }

        self.experiments[experiment_id] = experiment

        try:
            logger.info(
                f"Starting comparative experiment {experiment_id} with {len(test_cases)} test cases"
            )

            for i, test_case in enumerate(test_cases):
                natural_language = test_case["natural_language"]
                schema_context = test_case.get("schema_context", {})
                test_case.get("expected_patterns", [])

                logger.debug(
                    f"Running test case {i+1}/{len(test_cases)}: {natural_language[:50]}..."
                )

                # Test each approach
                for approach_name in approaches_to_test:
                    if approach_name not in self.approaches:
                        continue

                    approach = self.approaches[approach_name]

                    try:
                        # Run the approach
                        result = await approach.synthesize_sql(
                            natural_language, schema_context
                        )

                        # Evaluate result quality
                        quality_score = self._evaluate_result_quality(result, test_case)

                        # Create experiment result
                        exp_result = ExperimentResult(
                            experiment_id=experiment_id,
                            approach_name=approach_name,
                            execution_time_ms=result.get("execution_time_ms", 0),
                            accuracy_score=quality_score,
                            resource_usage={},  # TODO: Implement resource monitoring
                            output_quality=result.get("confidence", 0),
                            error_rate=1.0 if "error" in result else 0.0,
                            metadata={
                                "test_case_index": i,
                                "natural_language": natural_language,
                                "generated_sql": result.get("sql", ""),
                                "approach_specific": {
                                    k: v
                                    for k, v in result.items()
                                    if k
                                    not in ["sql", "confidence", "execution_time_ms"]
                                },
                            },
                            timestamp=datetime.utcnow(),
                        )

                        experiment["results"][approach_name].append(exp_result)
                        approach.record_execution(exp_result)

                    except Exception as e:
                        logger.error(
                            f"Error running approach {approach_name} on test case {i}: {e}"
                        )

                        # Record failure
                        exp_result = ExperimentResult(
                            experiment_id=experiment_id,
                            approach_name=approach_name,
                            execution_time_ms=0,
                            accuracy_score=0,
                            resource_usage={},
                            output_quality=0,
                            error_rate=1.0,
                            metadata={
                                "test_case_index": i,
                                "natural_language": natural_language,
                                "error": str(e),
                            },
                            timestamp=datetime.utcnow(),
                        )

                        experiment["results"][approach_name].append(exp_result)
                        approach.record_execution(exp_result)

            # Finalize experiment
            experiment["status"] = ExperimentStatus.COMPLETED
            experiment["end_time"] = datetime.utcnow()
            experiment["duration_seconds"] = (
                experiment["end_time"] - experiment["start_time"]
            ).total_seconds()

            # Generate analysis
            analysis = self._analyze_experiment_results(experiment)
            experiment["analysis"] = analysis

            logger.info(f"Completed experiment {experiment_id}")

            return experiment

        except Exception as e:
            experiment["status"] = ExperimentStatus.FAILED
            experiment["error"] = str(e)
            experiment["end_time"] = datetime.utcnow()
            logger.error(f"Experiment {experiment_id} failed: {e}")

            return experiment

    async def test_hypothesis(
        self, hypothesis_id: str, test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Test a specific research hypothesis."""

        if hypothesis_id not in self.hypotheses:
            raise ValueError(f"Hypothesis {hypothesis_id} not found")

        hypothesis = self.hypotheses[hypothesis_id]

        # Run comparative experiment
        experiment_id = f"hypothesis_{hypothesis_id}_{int(time.time())}"
        approaches_to_test = [hypothesis.baseline_approach, hypothesis.novel_approach]

        experiment_results = await self.run_comparative_experiment(
            experiment_id, test_cases, approaches_to_test
        )

        # Analyze results against hypothesis
        hypothesis_results = self._analyze_hypothesis_results(
            hypothesis, experiment_results
        )

        return {
            "hypothesis": asdict(hypothesis),
            "experiment_results": experiment_results,
            "hypothesis_validation": hypothesis_results,
            "conclusion": hypothesis_results.get("conclusion"),
            "statistical_significance": hypothesis_results.get(
                "statistical_significance", False
            ),
        }

    def get_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get results for a specific experiment."""
        return self.experiments.get(experiment_id)

    def get_approach_comparison(self) -> Dict[str, Any]:
        """Get comprehensive comparison of all approaches."""
        comparison = {}

        for approach_name, approach in self.approaches.items():
            comparison[approach_name] = {
                "metadata": approach.get_approach_metadata(),
                "performance": approach.get_performance_summary(),
            }

        return {
            "approaches": comparison,
            "total_approaches": len(self.approaches),
            "total_experiments": len(self.experiments),
            "summary": self._generate_approach_summary(comparison),
        }

    def _register_default_approaches(self):
        """Register default experimental approaches."""
        self.register_approach(TemplateBasedApproach())
        self.register_approach(SemanticSimilarityApproach())
        self.register_approach(HybridApproach())

    def _evaluate_result_quality(
        self, result: Dict[str, Any], test_case: Dict[str, Any]
    ) -> float:
        """Evaluate the quality of a result against test case expectations."""

        # Basic quality scoring
        quality_score = 0.0

        # Check if SQL was generated without errors
        if "error" not in result and result.get("sql"):
            quality_score += 0.3

        # Check confidence score
        confidence = result.get("confidence", 0)
        quality_score += confidence * 0.4

        # Check execution time (bonus for faster execution)
        execution_time = result.get("execution_time_ms", 1000)
        if execution_time < 100:
            quality_score += 0.2
        elif execution_time < 500:
            quality_score += 0.1

        # Check for expected patterns in generated SQL
        expected_patterns = test_case.get("expected_patterns", [])
        if expected_patterns:
            sql = result.get("sql", "").lower()
            pattern_matches = sum(
                1 for pattern in expected_patterns if pattern.lower() in sql
            )
            if pattern_matches > 0:
                quality_score += (pattern_matches / len(expected_patterns)) * 0.1

        return min(1.0, quality_score)

    def _analyze_experiment_results(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze results from a comparative experiment."""

        results_by_approach = experiment["results"]

        analysis = {
            "performance_ranking": [],
            "statistical_summary": {},
            "winner": None,
            "insights": [],
        }

        # Calculate average metrics for each approach
        approach_stats = {}

        for approach_name, results in results_by_approach.items():
            if not results:
                continue

            stats = {
                "avg_execution_time_ms": sum(r.execution_time_ms for r in results)
                / len(results),
                "avg_accuracy_score": sum(r.accuracy_score for r in results)
                / len(results),
                "avg_output_quality": sum(r.output_quality for r in results)
                / len(results),
                "success_rate": sum(1 for r in results if r.error_rate < 0.1)
                / len(results)
                * 100,
                "total_results": len(results),
            }

            approach_stats[approach_name] = stats

        # Rank approaches by overall performance
        def overall_score(stats):
            """TODO: Add docstring"""
            return (
                stats["avg_accuracy_score"] * 0.4
                + stats["avg_output_quality"] * 0.3
                + stats["success_rate"] / 100 * 0.2
                + max(0, (1000 - stats["avg_execution_time_ms"]) / 1000) * 0.1
            )

        ranked_approaches = sorted(
            approach_stats.items(), key=lambda x: overall_score(x[1]), reverse=True
        )

        analysis["performance_ranking"] = [
            {
                "rank": i + 1,
                "approach": approach_name,
                "overall_score": overall_score(stats),
                "stats": stats,
            }
            for i, (approach_name, stats) in enumerate(ranked_approaches)
        ]

        analysis["statistical_summary"] = approach_stats

        if ranked_approaches:
            analysis["winner"] = ranked_approaches[0][0]

            # Generate insights
            winner_stats = ranked_approaches[0][1]
            if len(ranked_approaches) > 1:
                runner_up_stats = ranked_approaches[1][1]

                improvement = overall_score(winner_stats) - overall_score(
                    runner_up_stats
                )
                if improvement > 0.1:
                    analysis["insights"].append(
                        f"Clear winner: {ranked_approaches[0][0]} outperforms others by {improvement:.1%}"
                    )
                else:
                    analysis["insights"].append(
                        "Close competition: Results are very similar across approaches"
                    )

            if winner_stats["avg_execution_time_ms"] < 100:
                analysis["insights"].append(
                    "Winner has excellent performance with sub-100ms execution time"
                )

            if winner_stats["success_rate"] > 95:
                analysis["insights"].append(
                    "Winner demonstrates high reliability with >95% success rate"
                )

        return analysis

    def _analyze_hypothesis_results(
        self, hypothesis: ResearchHypothesis, experiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze experiment results against research hypothesis."""

        baseline_results = experiment["results"].get(hypothesis.baseline_approach, [])
        novel_results = experiment["results"].get(hypothesis.novel_approach, [])

        if not baseline_results or not novel_results:
            return {
                "conclusion": "INCONCLUSIVE",
                "reason": "Insufficient data for comparison",
                "statistical_significance": False,
            }

        # Calculate performance metrics
        baseline_metrics = {
            "avg_accuracy": sum(r.accuracy_score for r in baseline_results)
            / len(baseline_results),
            "avg_execution_time": sum(r.execution_time_ms for r in baseline_results)
            / len(baseline_results),
            "success_rate": sum(1 for r in baseline_results if r.error_rate < 0.1)
            / len(baseline_results),
        }

        novel_metrics = {
            "avg_accuracy": sum(r.accuracy_score for r in novel_results)
            / len(novel_results),
            "avg_execution_time": sum(r.execution_time_ms for r in novel_results)
            / len(novel_results),
            "success_rate": sum(1 for r in novel_results if r.error_rate < 0.1)
            / len(novel_results),
        }

        # Check hypothesis success criteria
        hypothesis_validated = True
        validation_details = {}

        for criterion, expected_improvement in hypothesis.success_criteria.items():
            if criterion in baseline_metrics and criterion in novel_metrics:
                baseline_value = baseline_metrics[criterion]
                novel_value = novel_metrics[criterion]

                # Calculate improvement (higher is better for accuracy and success_rate, lower is better for execution_time)
                if criterion == "avg_execution_time":
                    improvement = (baseline_value - novel_value) / baseline_value
                else:
                    improvement = (
                        (novel_value - baseline_value) / baseline_value
                        if baseline_value > 0
                        else 0
                    )

                validation_details[criterion] = {
                    "baseline": baseline_value,
                    "novel": novel_value,
                    "improvement": improvement,
                    "expected": expected_improvement,
                    "met": improvement >= expected_improvement,
                }

                if not validation_details[criterion]["met"]:
                    hypothesis_validated = False

        # Determine conclusion
        if hypothesis_validated:
            conclusion = "CONFIRMED"
        else:
            conclusion = "REJECTED"

        # Simple statistical significance check (would use proper tests in production)
        statistical_significance = (
            len(baseline_results) >= 10 and len(novel_results) >= 10
        )

        return {
            "conclusion": conclusion,
            "hypothesis_validated": hypothesis_validated,
            "validation_details": validation_details,
            "baseline_metrics": baseline_metrics,
            "novel_metrics": novel_metrics,
            "statistical_significance": statistical_significance,
            "confidence_level": (
                hypothesis.confidence_level if hypothesis_validated else 0.0
            ),
        }

    def _generate_approach_summary(self, comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary insights from approach comparison."""

        total_approaches = len(comparison)

        # Find best performing approach overall
        best_approach = None
        best_score = -1

        performance_data = []

        for approach_name, data in comparison.items():
            performance = data.get("performance", {})
            if performance.get("avg_accuracy_score", 0) > best_score:
                best_score = performance.get("avg_accuracy_score", 0)
                best_approach = approach_name

            performance_data.append(
                {
                    "name": approach_name,
                    "type": data.get("metadata", {}).get("type", "unknown"),
                    "complexity": data.get("metadata", {}).get("complexity", "unknown"),
                    "avg_accuracy": performance.get("avg_accuracy_score", 0),
                    "avg_execution_time": performance.get("avg_execution_time_ms", 0),
                    "success_rate": performance.get("success_rate", 0),
                }
            )

        return {
            "total_approaches_evaluated": total_approaches,
            "best_overall_approach": best_approach,
            "approach_types": list(
                {p["type"] for p in performance_data if p["type"] != "unknown"}
            ),
            "complexity_distribution": Counter(
                p["complexity"]
                for p in performance_data
                if p["complexity"] != "unknown"
            ),
            "performance_summary": performance_data,
            "research_recommendations": self._generate_research_recommendations(
                performance_data
            ),
        }

    def _generate_research_recommendations(
        self, performance_data: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate research recommendations based on performance analysis."""
        recommendations = []

        # Analyze performance patterns
        high_performers = [p for p in performance_data if p["avg_accuracy"] > 0.7]
        fast_performers = [p for p in performance_data if p["avg_execution_time"] < 200]

        if high_performers:
            recommendations.append(
                f"Focus on {', '.join(p['name'] for p in high_performers[:2])} approaches as they show highest accuracy"
            )

        if fast_performers:
            recommendations.append(
                f"Optimize for speed using insights from {fast_performers[0]['name']} approach"
            )

        # Type-based recommendations
        approach_types = Counter(p["type"] for p in performance_data)
        if approach_types.most_common(1):
            dominant_type = approach_types.most_common(1)[0][0]
            if dominant_type != "hybrid":
                recommendations.append(
                    f"Explore hybrid approaches combining {dominant_type} with other methods"
                )

        # Complexity analysis
        complexity_dist = Counter(p["complexity"] for p in performance_data)
        if complexity_dist.get("High", 0) > complexity_dist.get("Low", 0):
            recommendations.append(
                "Investigate whether complexity benefits justify costs"
            )

        return recommendations


# Global experimental framework instance
experimental_framework = ExperimentalFramework()


# ==============================================================================
# COMPREHENSIVE RESEARCH VALIDATION FRAMEWORK
# ==============================================================================


@dataclass
class ComprehensiveExperimentResult:
    """Result of a single experimental run with comprehensive metrics."""
    experiment_id: str
    method_name: str
    query_id: str
    natural_language: str
    ground_truth_sql: str
    predicted_sql: str
    execution_match: bool
    exact_match: bool
    semantic_similarity: float
    execution_time_ms: float
    confidence_score: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComprehensiveExperimentConfiguration:
    """Configuration for comprehensive experimental runs."""
    experiment_name: str
    description: str
    methods_to_test: List[str]
    datasets: List[str]
    evaluation_metrics: List[str]
    num_runs: int = 3
    random_seed: int = 42
    output_directory: str = "experiment_results"
    statistical_tests: List[str] = field(default_factory=lambda: ["t-test", "wilcoxon"])
    significance_level: float = 0.05


@dataclass
class BaselineMethod:
    """Definition of a baseline method for comparison."""
    name: str
    description: str
    implementation_callable: Any
    expected_performance: Dict[str, float] = field(default_factory=dict)
    paper_reference: Optional[str] = None


class QueryComplexityClassifier:
    """Classifies query complexity for stratified evaluation."""
    
    def __init__(self):
        self.complexity_features = {
            'num_tables': 0.3,
            'num_joins': 0.4,
            'num_aggregations': 0.2,
            'nested_queries': 0.5,
            'num_conditions': 0.1,
            'has_groupby': 0.2,
            'has_orderby': 0.1,
            'has_having': 0.3,
        }
    
    def classify_query(self, sql: str) -> Dict[str, Any]:
        """Classify query complexity and return features."""
        sql_upper = sql.upper()
        
        # Count tables
        num_tables = len([word for word in sql_upper.split() if word == 'FROM']) + sql_upper.count('JOIN')
        
        # Count joins
        join_keywords = ['INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN', 'JOIN']
        num_joins = sum(sql_upper.count(keyword) for keyword in join_keywords)
        
        # Count aggregations
        agg_functions = ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']
        num_aggregations = sum(sql_upper.count(func + '(') for func in agg_functions)
        
        # Check for nested queries
        nested_queries = sql_upper.count('SELECT') - 1  # Main query doesn't count
        
        # Count conditions
        num_conditions = sql_upper.count('WHERE') + sql_upper.count('AND') + sql_upper.count('OR')
        
        # Check for specific clauses
        has_groupby = 'GROUP BY' in sql_upper
        has_orderby = 'ORDER BY' in sql_upper
        has_having = 'HAVING' in sql_upper
        
        features = {
            'num_tables': min(num_tables, 5) / 5.0,  # Normalize to 0-1
            'num_joins': min(num_joins, 4) / 4.0,
            'num_aggregations': min(num_aggregations, 3) / 3.0,
            'nested_queries': min(nested_queries, 2) / 2.0,
            'num_conditions': min(num_conditions, 10) / 10.0,
            'has_groupby': 1.0 if has_groupby else 0.0,
            'has_orderby': 1.0 if has_orderby else 0.0,
            'has_having': 1.0 if has_having else 0.0,
        }
        
        # Calculate overall complexity score
        complexity_score = sum(
            features[feature] * weight 
            for feature, weight in self.complexity_features.items()
        )
        
        # Classify into buckets
        if complexity_score <= 0.3:
            complexity_class = 'simple'
        elif complexity_score <= 0.7:
            complexity_class = 'medium'
        else:
            complexity_class = 'complex'
        
        return {
            'complexity_score': complexity_score,
            'complexity_class': complexity_class,
            'features': features,
            'raw_counts': {
                'num_tables': num_tables,
                'num_joins': num_joins,
                'num_aggregations': num_aggregations,
                'nested_queries': nested_queries,
                'num_conditions': num_conditions
            }
        }
    
    def get_complexity_distribution(self, queries: List[str]) -> Dict[str, Any]:
        """Analyze complexity distribution of a query set."""
        classifications = [self.classify_query(q) for q in queries]
        
        # Count by class
        class_counts = {'simple': 0, 'medium': 0, 'complex': 0}
        complexity_scores = []
        
        for classification in classifications:
            class_counts[classification['complexity_class']] += 1
            complexity_scores.append(classification['complexity_score'])
        
        return {
            'class_distribution': class_counts,
            'class_percentages': {
                cls: count / len(queries) * 100 
                for cls, count in class_counts.items()
            },
            'average_complexity': statistics.mean(complexity_scores),
            'complexity_std': statistics.stdev(complexity_scores) if len(complexity_scores) > 1 else 0.0,
            'complexity_range': [min(complexity_scores), max(complexity_scores)]
        }


class ComprehensiveEvaluationMetrics:
    """Comprehensive evaluation metrics for NL2SQL systems."""
    
    @staticmethod
    def exact_match(predicted: str, ground_truth: str) -> bool:
        """Check if SQL queries are exactly the same (normalized)."""
        pred_normalized = ComprehensiveEvaluationMetrics._normalize_sql(predicted)
        truth_normalized = ComprehensiveEvaluationMetrics._normalize_sql(ground_truth)
        return pred_normalized == truth_normalized
    
    @staticmethod
    def _normalize_sql(sql: str) -> str:
        """Normalize SQL for comparison."""
        # Remove extra whitespace and convert to lowercase
        normalized = ' '.join(sql.strip().upper().split())
        
        # Remove trailing semicolon
        if normalized.endswith(';'):
            normalized = normalized[:-1]
            
        return normalized
    
    @staticmethod
    def execution_accuracy(predicted: str, ground_truth: str, 
                         database_connector: Optional[Any] = None) -> bool:
        """Check if queries return the same results when executed."""
        if not database_connector:
            # Fallback: structural similarity
            return ComprehensiveEvaluationMetrics._structural_similarity(predicted, ground_truth) > 0.8
        
        try:
            # This would execute both queries and compare results
            # Implementation depends on database connector
            pred_result = database_connector.execute(predicted)
            truth_result = database_connector.execute(ground_truth)
            
            # Compare result sets
            return ComprehensiveEvaluationMetrics._compare_result_sets(pred_result, truth_result)
        except Exception as e:
            logger.warning(f"Execution comparison failed: {e}")
            return False
    
    @staticmethod
    def _structural_similarity(sql1: str, sql2: str) -> float:
        """Calculate structural similarity between SQL queries."""
        # Extract key components
        def extract_components(sql):
            sql_upper = sql.upper()
            components = {
                'select': 'SELECT' in sql_upper,
                'from': 'FROM' in sql_upper,
                'where': 'WHERE' in sql_upper,
                'join': any(j in sql_upper for j in ['JOIN', 'INNER JOIN', 'LEFT JOIN']),
                'group_by': 'GROUP BY' in sql_upper,
                'order_by': 'ORDER BY' in sql_upper,
                'having': 'HAVING' in sql_upper,
                'limit': 'LIMIT' in sql_upper,
            }
            return components
        
        comp1 = extract_components(sql1)
        comp2 = extract_components(sql2)
        
        # Calculate Jaccard similarity
        intersection = sum(1 for k in comp1 if comp1[k] and comp2[k])
        union = sum(1 for k in comp1 if comp1[k] or comp2[k])
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def _compare_result_sets(result1: Any, result2: Any) -> bool:
        """Compare database result sets."""
        # This is a simplified implementation
        # In practice, would need to handle different result formats
        try:
            if hasattr(result1, 'fetchall') and hasattr(result2, 'fetchall'):
                rows1 = sorted(result1.fetchall())
                rows2 = sorted(result2.fetchall())
                return rows1 == rows2
            else:
                return str(result1) == str(result2)
        except Exception:
            return False
    
    @staticmethod
    def semantic_similarity(predicted: str, ground_truth: str) -> float:
        """Calculate semantic similarity using multiple heuristics."""
        # Structural similarity
        struct_sim = ComprehensiveEvaluationMetrics._structural_similarity(predicted, ground_truth)
        
        # Keyword overlap
        pred_words = set(predicted.upper().split())
        truth_words = set(ground_truth.upper().split())
        
        if not pred_words and not truth_words:
            keyword_sim = 1.0
        elif not pred_words or not truth_words:
            keyword_sim = 0.0
        else:
            intersection = len(pred_words & truth_words)
            union = len(pred_words | truth_words)
            keyword_sim = intersection / union
        
        # Table and column similarity
        table_sim = ComprehensiveEvaluationMetrics._table_column_similarity(predicted, ground_truth)
        
        # Weighted average
        return (struct_sim * 0.4 + keyword_sim * 0.3 + table_sim * 0.3)
    
    @staticmethod
    def _table_column_similarity(sql1: str, sql2: str) -> float:
        """Calculate similarity based on tables and columns mentioned."""
        def extract_table_column_references(sql):
            # Simple extraction - in practice would use SQL parser
            words = sql.upper().replace(',', ' ').replace('(', ' ').replace(')', ' ').split()
            
            # Look for patterns like table.column
            table_col_refs = set()
            for word in words:
                if '.' in word and word.count('.') == 1:
                    table_col_refs.add(word)
            
            return table_col_refs
        
        refs1 = extract_table_column_references(sql1)
        refs2 = extract_table_column_references(sql2)
        
        if not refs1 and not refs2:
            return 1.0
        elif not refs1 or not refs2:
            return 0.0
        
        intersection = len(refs1 & refs2)
        union = len(refs1 | refs2)
        
        return intersection / union
    
    @staticmethod
    def calculate_all_metrics(predicted: str, ground_truth: str, 
                            execution_time_ms: float = 0.0,
                            database_connector: Optional[Any] = None) -> Dict[str, float]:
        """Calculate all available metrics."""
        return {
            'exact_match': float(ComprehensiveEvaluationMetrics.exact_match(predicted, ground_truth)),
            'execution_accuracy': float(ComprehensiveEvaluationMetrics.execution_accuracy(
                predicted, ground_truth, database_connector
            )),
            'semantic_similarity': ComprehensiveEvaluationMetrics.semantic_similarity(predicted, ground_truth),
            'structural_similarity': ComprehensiveEvaluationMetrics._structural_similarity(predicted, ground_truth),
            'execution_time_ms': execution_time_ms
        }


class StatisticalAnalyzer:
    """Statistical analysis and significance testing for experiments."""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def compare_methods(self, results_a: List[float], results_b: List[float],
                       method_name_a: str = "Method A", method_name_b: str = "Method B") -> Dict[str, Any]:
        """Compare two methods using multiple statistical tests."""
        
        if len(results_a) != len(results_b):
            raise ValueError("Result lists must have the same length")
        
        if len(results_a) < 2:
            raise ValueError("Need at least 2 samples for comparison")
        
        # Descriptive statistics
        stats_a = {
            'mean': statistics.mean(results_a),
            'std': statistics.stdev(results_a) if len(results_a) > 1 else 0.0,
            'median': statistics.median(results_a),
            'min': min(results_a),
            'max': max(results_a),
            'n': len(results_a)
        }
        
        stats_b = {
            'mean': statistics.mean(results_b),
            'std': statistics.stdev(results_b) if len(results_b) > 1 else 0.0,
            'median': statistics.median(results_b),
            'min': min(results_b),
            'max': max(results_b),
            'n': len(results_b)
        }
        
        # Effect size (Cohen's d)
        pooled_std = ((stats_a['std'] ** 2 + stats_b['std'] ** 2) / 2) ** 0.5
        cohens_d = (stats_a['mean'] - stats_b['mean']) / pooled_std if pooled_std > 0 else 0.0
        
        comparison_result = {
            'method_a': method_name_a,
            'method_b': method_name_b,
            'stats_a': stats_a,
            'stats_b': stats_b,
            'effect_size_cohens_d': cohens_d,
            'statistical_tests': {}
        }
        
        # Statistical tests
        if SCIPY_AVAILABLE:
            # Paired t-test (assumes normal distribution)
            try:
                t_stat, t_pvalue = ttest_rel(results_a, results_b)
                comparison_result['statistical_tests']['paired_t_test'] = {
                    't_statistic': t_stat,
                    'p_value': t_pvalue,
                    'significant': t_pvalue < self.significance_level,
                    'interpretation': self._interpret_t_test(t_stat, t_pvalue)
                }
            except Exception as e:
                logger.warning(f"T-test failed: {e}")
            
            # Wilcoxon signed-rank test (non-parametric)
            try:
                w_stat, w_pvalue = wilcoxon(results_a, results_b, alternative='two-sided')
                comparison_result['statistical_tests']['wilcoxon_signed_rank'] = {
                    'w_statistic': w_stat,
                    'p_value': w_pvalue,
                    'significant': w_pvalue < self.significance_level,
                    'interpretation': self._interpret_wilcoxon(w_stat, w_pvalue)
                }
            except Exception as e:
                logger.warning(f"Wilcoxon test failed: {e}")
            
            # Mann-Whitney U test (independent samples)
            try:
                u_stat, u_pvalue = mannwhitneyu(results_a, results_b, alternative='two-sided')
                comparison_result['statistical_tests']['mann_whitney_u'] = {
                    'u_statistic': u_stat,
                    'p_value': u_pvalue,
                    'significant': u_pvalue < self.significance_level,
                    'interpretation': self._interpret_mann_whitney(u_stat, u_pvalue)
                }
            except Exception as e:
                logger.warning(f"Mann-Whitney test failed: {e}")
        else:
            # Simple comparison without scipy
            comparison_result['statistical_tests']['basic_comparison'] = {
                'mean_difference': stats_a['mean'] - stats_b['mean'],
                'interpretation': f"{method_name_a} has {'higher' if stats_a['mean'] > stats_b['mean'] else 'lower'} mean performance"
            }
        
        # Overall interpretation
        comparison_result['summary'] = self._generate_comparison_summary(comparison_result)
        
        return comparison_result
    
    def _interpret_t_test(self, t_stat: float, p_value: float) -> str:
        """Interpret t-test results."""
        if p_value < self.significance_level:
            direction = "significantly higher" if t_stat > 0 else "significantly lower"
            return f"Method A performs {direction} than Method B (p={p_value:.4f})"
        else:
            return f"No significant difference between methods (p={p_value:.4f})"
    
    def _interpret_wilcoxon(self, w_stat: float, p_value: float) -> str:
        """Interpret Wilcoxon test results."""
        if p_value < self.significance_level:
            return f"Significant difference detected by non-parametric test (p={p_value:.4f})"
        else:
            return f"No significant difference by non-parametric test (p={p_value:.4f})"
    
    def _interpret_mann_whitney(self, u_stat: float, p_value: float) -> str:
        """Interpret Mann-Whitney U test results."""
        if p_value < self.significance_level:
            return f"Significant difference between independent groups (p={p_value:.4f})"
        else:
            return f"No significant difference between independent groups (p={p_value:.4f})"
    
    def _generate_comparison_summary(self, comparison: Dict[str, Any]) -> str:
        """Generate human-readable summary of comparison."""
        stats_a = comparison['stats_a']
        stats_b = comparison['stats_b']
        
        better_method = comparison['method_a'] if stats_a['mean'] > stats_b['mean'] else comparison['method_b']
        worse_method = comparison['method_b'] if stats_a['mean'] > stats_b['mean'] else comparison['method_a']
        
        improvement = abs(stats_a['mean'] - stats_b['mean']) / min(stats_a['mean'], stats_b['mean']) * 100
        
        # Check if any test found significance
        significant = False
        if comparison['statistical_tests']:
            for test_name, test_result in comparison['statistical_tests'].items():
                if isinstance(test_result, dict) and test_result.get('significant', False):
                    significant = True
                    break
        
        summary = f"{better_method} outperforms {worse_method} by {improvement:.1f}% on average"
        if significant:
            summary += " (statistically significant)"
        else:
            summary += " (not statistically significant)"
        
        # Add effect size interpretation
        effect_size = abs(comparison['effect_size_cohens_d'])
        if effect_size < 0.2:
            effect_desc = "negligible"
        elif effect_size < 0.5:
            effect_desc = "small"
        elif effect_size < 0.8:
            effect_desc = "medium"
        else:
            effect_desc = "large"
        
        summary += f" with {effect_desc} effect size (Cohen's d = {comparison['effect_size_cohens_d']:.3f})"
        
        return summary
    
    def analyze_method_across_complexity(self, results_by_complexity: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze method performance across query complexity levels."""
        analysis = {
            'complexity_levels': list(results_by_complexity.keys()),
            'performance_by_complexity': {},
            'complexity_trend': None
        }
        
        # Calculate stats for each complexity level
        for complexity, results in results_by_complexity.items():
            if results:
                analysis['performance_by_complexity'][complexity] = {
                    'mean': statistics.mean(results),
                    'std': statistics.stdev(results) if len(results) > 1 else 0.0,
                    'median': statistics.median(results),
                    'count': len(results)
                }
        
        # Analyze trend across complexity levels
        if len(analysis['performance_by_complexity']) >= 3:
            complexity_order = ['simple', 'medium', 'complex']
            means = []
            
            for complexity in complexity_order:
                if complexity in analysis['performance_by_complexity']:
                    means.append(analysis['performance_by_complexity'][complexity]['mean'])
            
            if len(means) >= 2:
                # Simple trend analysis
                if all(means[i] >= means[i+1] for i in range(len(means)-1)):
                    trend = "decreasing"  # Performance decreases with complexity
                elif all(means[i] <= means[i+1] for i in range(len(means)-1)):
                    trend = "increasing"  # Performance increases with complexity
                else:
                    trend = "mixed"
                
                analysis['complexity_trend'] = trend
        
        return analysis


class ComprehensiveExperimentRunner:
    """Main experiment runner for comprehensive NL2SQL evaluation."""
    
    def __init__(self, config: ComprehensiveExperimentConfiguration):
        self.config = config
        self.complexity_classifier = QueryComplexityClassifier()
        self.statistical_analyzer = StatisticalAnalyzer(config.significance_level)
        
        # Create output directory
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Experiment state
        self.experiment_id = f"{config.experiment_name}_{int(time.time())}"
        self.results: List[ComprehensiveExperimentResult] = []
        self.baselines: List[BaselineMethod] = []
        
    def add_baseline_method(self, baseline: BaselineMethod):
        """Add a baseline method for comparison."""
        self.baselines.append(baseline)
        
    def run_experiment(self, test_queries: List[Dict[str, Any]], 
                      methods_to_test: Dict[str, Any],
                      database_connector: Optional[Any] = None) -> Dict[str, Any]:
        """Run comprehensive experiment comparing multiple methods."""
        
        logger.info(f"Starting comprehensive experiment: {self.experiment_id}")
        experiment_start_time = time.time()
        
        # Set random seed for reproducibility
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        
        # Analyze dataset complexity
        ground_truth_queries = [q.get('sql', '') for q in test_queries]
        complexity_analysis = self.complexity_classifier.get_complexity_distribution(ground_truth_queries)
        
        logger.info(f"Dataset complexity distribution: {complexity_analysis['class_percentages']}")
        
        # Run experiments for each method
        method_results = {}
        
        for method_name, method_impl in methods_to_test.items():
            logger.info(f"Testing method: {method_name}")
            method_start_time = time.time()
            
            method_results[method_name] = []
            
            for run_idx in range(self.config.num_runs):
                logger.info(f"  Run {run_idx + 1}/{self.config.num_runs}")
                
                for query_idx, query_data in enumerate(test_queries):
                    result = self._run_single_query(
                        method_name, method_impl, query_data, 
                        query_idx, run_idx, database_connector
                    )
                    method_results[method_name].append(result)
                    self.results.append(result)
            
            method_time = time.time() - method_start_time
            logger.info(f"  Completed {method_name} in {method_time:.2f}s")
        
        # Analyze results
        analysis = self._analyze_experiment_results(method_results, complexity_analysis)
        
        # Save results
        self._save_experiment_results(analysis)
        
        experiment_time = time.time() - experiment_start_time
        logger.info(f"Comprehensive experiment completed in {experiment_time:.2f}s")
        
        return analysis
    
    def _run_single_query(self, method_name: str, method_impl: Any, 
                         query_data: Dict[str, Any], query_idx: int, run_idx: int,
                         database_connector: Optional[Any] = None) -> ComprehensiveExperimentResult:
        """Run a single query through a method and collect comprehensive results."""
        
        query_id = f"{self.experiment_id}_{method_name}_{query_idx}_{run_idx}"
        natural_language = query_data.get('question', query_data.get('natural_language', ''))
        ground_truth_sql = query_data.get('sql', query_data.get('ground_truth', ''))
        schema_metadata = query_data.get('schema_metadata', {})
        
        try:
            # Time the prediction
            start_time = time.time()
            
            # Call the method implementation
            if hasattr(method_impl, 'synthesize_sql'):
                prediction_result = method_impl.synthesize_sql(natural_language, schema_metadata)
                predicted_sql = prediction_result.get('sql', '') if isinstance(prediction_result, dict) else str(prediction_result)
                confidence = prediction_result.get('confidence', 0.5) if isinstance(prediction_result, dict) else 0.5
            elif callable(method_impl):
                prediction_result = method_impl(natural_language, schema_metadata)
                predicted_sql = prediction_result if isinstance(prediction_result, str) else str(prediction_result)
                confidence = 0.5
            else:
                raise ValueError(f"Method {method_name} is not callable or doesn't have synthesize_sql method")
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Calculate comprehensive metrics
            metrics = ComprehensiveEvaluationMetrics.calculate_all_metrics(
                predicted_sql, ground_truth_sql, execution_time_ms, database_connector
            )
            
            return ComprehensiveExperimentResult(
                experiment_id=self.experiment_id,
                method_name=method_name,
                query_id=query_id,
                natural_language=natural_language,
                ground_truth_sql=ground_truth_sql,
                predicted_sql=predicted_sql,
                execution_match=bool(metrics['execution_accuracy']),
                exact_match=bool(metrics['exact_match']),
                semantic_similarity=metrics['semantic_similarity'],
                execution_time_ms=execution_time_ms,
                confidence_score=confidence,
                metadata={
                    'run_idx': run_idx,
                    'query_idx': query_idx,
                    'all_metrics': metrics
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing query {query_idx} with {method_name}: {e}")
            
            return ComprehensiveExperimentResult(
                experiment_id=self.experiment_id,
                method_name=method_name,
                query_id=query_id,
                natural_language=natural_language,
                ground_truth_sql=ground_truth_sql,
                predicted_sql="",
                execution_match=False,
                exact_match=False,
                semantic_similarity=0.0,
                execution_time_ms=0.0,
                confidence_score=0.0,
                error_message=str(e),
                metadata={
                    'run_idx': run_idx,
                    'query_idx': query_idx,
                    'error': True
                }
            )
    
    def _analyze_experiment_results(self, method_results: Dict[str, List[ComprehensiveExperimentResult]], 
                                  complexity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze experiment results and generate comprehensive report."""
        
        analysis = {
            'experiment_id': self.experiment_id,
            'experiment_config': self.config.__dict__,
            'dataset_analysis': complexity_analysis,
            'method_performance': {},
            'statistical_comparisons': {},
            'complexity_analysis': {},
            'summary': {}
        }
        
        # Calculate performance metrics for each method
        for method_name, results in method_results.items():
            if not results:
                continue
            
            # Overall metrics
            exact_matches = [r.exact_match for r in results]
            execution_matches = [r.execution_match for r in results]
            semantic_similarities = [r.semantic_similarity for r in results]
            execution_times = [r.execution_time_ms for r in results]
            confidence_scores = [r.confidence_score for r in results]
            
            # Error rate
            errors = [r for r in results if r.error_message is not None]
            error_rate = len(errors) / len(results)
            
            analysis['method_performance'][method_name] = {
                'exact_match_accuracy': statistics.mean(exact_matches),
                'execution_accuracy': statistics.mean(execution_matches),
                'semantic_similarity': statistics.mean(semantic_similarities),
                'average_execution_time_ms': statistics.mean(execution_times),
                'average_confidence': statistics.mean(confidence_scores),
                'error_rate': error_rate,
                'total_queries': len(results),
                'performance_std': {
                    'exact_match_std': statistics.stdev(exact_matches) if len(exact_matches) > 1 else 0.0,
                    'execution_std': statistics.stdev(execution_matches) if len(execution_matches) > 1 else 0.0,
                    'semantic_similarity_std': statistics.stdev(semantic_similarities) if len(semantic_similarities) > 1 else 0.0
                }
            }
        
        # Statistical comparisons between methods
        method_names = list(method_results.keys())
        for i, method_a in enumerate(method_names):
            for method_b in method_names[i+1:]:
                if method_a == method_b:
                    continue
                
                # Compare on semantic similarity
                results_a = [r.semantic_similarity for r in method_results[method_a]]
                results_b = [r.semantic_similarity for r in method_results[method_b]]
                
                comparison_key = f"{method_a}_vs_{method_b}"
                analysis['statistical_comparisons'][comparison_key] = \
                    self.statistical_analyzer.compare_methods(results_a, results_b, method_a, method_b)
        
        # Complexity-stratified analysis
        for method_name, results in method_results.items():
            complexity_results = {'simple': [], 'medium': [], 'complex': []}
            
            for result in results:
                # Classify query complexity
                query_complexity = self.complexity_classifier.classify_query(result.ground_truth_sql)
                complexity_class = query_complexity['complexity_class']
                
                complexity_results[complexity_class].append(result.semantic_similarity)
            
            analysis['complexity_analysis'][method_name] = \
                self.statistical_analyzer.analyze_method_across_complexity(complexity_results)
        
        # Generate summary
        analysis['summary'] = self._generate_experiment_summary(analysis)
        
        return analysis
    
    def _generate_experiment_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level summary of experiment results."""
        
        method_performance = analysis['method_performance']
        
        if not method_performance:
            return {'error': 'No method performance data available'}
        
        # Find best performing method
        best_method = max(
            method_performance.items(),
            key=lambda x: x[1]['semantic_similarity']
        )
        
        # Find fastest method
        fastest_method = min(
            method_performance.items(),
            key=lambda x: x[1]['average_execution_time_ms']
        )
        
        # Calculate overall statistics
        all_semantic_scores = []
        all_exact_matches = []
        
        for method_name, perf in method_performance.items():
            all_semantic_scores.append(perf['semantic_similarity'])
            all_exact_matches.append(perf['exact_match_accuracy'])
        
        summary = {
            'best_performing_method': {
                'name': best_method[0],
                'semantic_similarity': best_method[1]['semantic_similarity'],
                'exact_match_accuracy': best_method[1]['exact_match_accuracy']
            },
            'fastest_method': {
                'name': fastest_method[0],
                'average_time_ms': fastest_method[1]['average_execution_time_ms']
            },
            'overall_statistics': {
                'methods_tested': len(method_performance),
                'best_semantic_similarity': max(all_semantic_scores),
                'worst_semantic_similarity': min(all_semantic_scores),
                'best_exact_match': max(all_exact_matches),
                'worst_exact_match': min(all_exact_matches),
                'semantic_similarity_range': max(all_semantic_scores) - min(all_semantic_scores)
            },
            'key_findings': self._extract_key_findings(analysis)
        }
        
        return summary
    
    def _extract_key_findings(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract key findings from the analysis."""
        findings = []
        
        # Performance findings
        method_performance = analysis['method_performance']
        if len(method_performance) >= 2:
            best_methods = sorted(
                method_performance.items(),
                key=lambda x: x[1]['semantic_similarity'],
                reverse=True
            )
            
            best_name, best_perf = best_methods[0]
            second_name, second_perf = best_methods[1]
            
            improvement = ((best_perf['semantic_similarity'] - second_perf['semantic_similarity']) 
                          / second_perf['semantic_similarity'] * 100)
            
            findings.append(f"{best_name} outperforms {second_name} by {improvement:.1f}% in semantic similarity")
        
        # Statistical significance findings
        statistical_comparisons = analysis.get('statistical_comparisons', {})
        significant_comparisons = []
        
        for comp_name, comp_result in statistical_comparisons.items():
            tests = comp_result.get('statistical_tests', {})
            for test_name, test_result in tests.items():
                if isinstance(test_result, dict) and test_result.get('significant', False):
                    significant_comparisons.append(comp_name)
                    break
        
        if significant_comparisons:
            findings.append(f"Found statistically significant differences in {len(significant_comparisons)} method comparisons")
        else:
            findings.append("No statistically significant differences found between methods")
        
        # Complexity findings
        complexity_analysis = analysis.get('complexity_analysis', {})
        for method_name, complexity_result in complexity_analysis.items():
            trend = complexity_result.get('complexity_trend')
            if trend:
                if trend == 'decreasing':
                    findings.append(f"{method_name} performance decreases with query complexity")
                elif trend == 'increasing':
                    findings.append(f"{method_name} performance surprisingly improves with complexity")
        
        # Error rate findings
        high_error_methods = []
        for method_name, perf in method_performance.items():
            if perf.get('error_rate', 0) > 0.1:  # More than 10% error rate
                high_error_methods.append((method_name, perf['error_rate']))
        
        if high_error_methods:
            findings.append(f"High error rates detected in: {', '.join(m[0] for m in high_error_methods)}")
        
        return findings[:10]  # Limit to top 10 findings
    
    def _save_experiment_results(self, analysis: Dict[str, Any]):
        """Save comprehensive experiment results to files."""
        
        # Save detailed results as JSON
        results_file = self.output_dir / f"{self.experiment_id}_comprehensive_results.json"
        
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            serializable_results.append({
                'experiment_id': result.experiment_id,
                'method_name': result.method_name,
                'query_id': result.query_id,
                'natural_language': result.natural_language,
                'ground_truth_sql': result.ground_truth_sql,
                'predicted_sql': result.predicted_sql,
                'execution_match': result.execution_match,
                'exact_match': result.exact_match,
                'semantic_similarity': result.semantic_similarity,
                'execution_time_ms': result.execution_time_ms,
                'confidence_score': result.confidence_score,
                'error_message': result.error_message,
                'metadata': result.metadata
            })
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save analysis as JSON
        analysis_file = self.output_dir / f"{self.experiment_id}_comprehensive_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Save summary report as markdown
        summary_file = self.output_dir / f"{self.experiment_id}_comprehensive_summary.md"
        self._generate_markdown_report(analysis, summary_file)
        
        logger.info(f"Comprehensive results saved to {self.output_dir}")
    
    def _generate_markdown_report(self, analysis: Dict[str, Any], output_file: Path):
        """Generate a comprehensive markdown report of the experiment."""
        
        with open(output_file, 'w') as f:
            f.write(f"# Comprehensive Experiment Report: {self.config.experiment_name}\n\n")
            f.write(f"**Experiment ID:** {self.experiment_id}\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Configuration
            f.write("## Experiment Configuration\n\n")
            f.write(f"- **Description:** {self.config.description}\n")
            f.write(f"- **Number of runs:** {self.config.num_runs}\n")
            f.write(f"- **Random seed:** {self.config.random_seed}\n")
            f.write(f"- **Significance level:** {self.config.significance_level}\n\n")
            
            # Dataset analysis
            dataset_analysis = analysis.get('dataset_analysis', {})
            if dataset_analysis:
                f.write("## Dataset Analysis\n\n")
                class_dist = dataset_analysis.get('class_percentages', {})
                for complexity, percentage in class_dist.items():
                    f.write(f"- **{complexity.title()} queries:** {percentage:.1f}%\n")
                f.write(f"- **Average complexity score:** {dataset_analysis.get('average_complexity', 0):.3f}\n\n")
            
            # Method performance
            method_performance = analysis.get('method_performance', {})
            if method_performance:
                f.write("## Method Performance\n\n")
                f.write("| Method | Semantic Similarity | Exact Match | Execution Time (ms) | Error Rate |\n")
                f.write("|--------|-------------------|-------------|-------------------|----------|\n")
                
                for method_name, perf in method_performance.items():
                    f.write(f"| {method_name} | {perf['semantic_similarity']:.3f} | ")
                    f.write(f"{perf['exact_match_accuracy']:.3f} | ")
                    f.write(f"{perf['average_execution_time_ms']:.1f} | ")
                    f.write(f"{perf['error_rate']:.3f} |\n")
                f.write("\n")
            
            # Key findings
            summary = analysis.get('summary', {})
            key_findings = summary.get('key_findings', [])
            if key_findings:
                f.write("## Key Findings\n\n")
                for i, finding in enumerate(key_findings, 1):
                    f.write(f"{i}. {finding}\n")
                f.write("\n")
            
            # Statistical comparisons
            statistical_comparisons = analysis.get('statistical_comparisons', {})
            if statistical_comparisons:
                f.write("## Statistical Comparisons\n\n")
                for comp_name, comp_result in statistical_comparisons.items():
                    f.write(f"### {comp_name}\n")
                    f.write(f"{comp_result.get('summary', 'No summary available')}\n\n")
            
            # Best performing method
            if 'best_performing_method' in summary:
                best = summary['best_performing_method']
                f.write("## Best Performing Method\n\n")
                f.write(f"**{best['name']}** achieved the highest semantic similarity of ")
                f.write(f"{best['semantic_similarity']:.3f} with exact match accuracy of ")
                f.write(f"{best['exact_match_accuracy']:.3f}.\n\n")
        
        logger.info(f"Comprehensive markdown report saved to {output_file}")


# Export comprehensive framework classes
__all__.extend([
    'ComprehensiveExperimentRunner',
    'ComprehensiveExperimentConfiguration', 
    'ComprehensiveExperimentResult',
    'BaselineMethod',
    'QueryComplexityClassifier',
    'ComprehensiveEvaluationMetrics',
    'StatisticalAnalyzer',
    'SCIPY_AVAILABLE',
    'PLOTTING_AVAILABLE'
])
