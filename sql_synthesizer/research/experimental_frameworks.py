"""
Experimental Research Frameworks
Advanced algorithmic research and experimental SQL synthesis approaches.
"""

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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
                expected_patterns = test_case.get("expected_patterns", [])

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
                set(p["type"] for p in performance_data if p["type"] != "unknown")
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
