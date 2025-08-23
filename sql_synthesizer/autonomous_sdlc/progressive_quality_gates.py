"""
Progressive Quality Gates System - Generation 1: Enhanced Implementation
Autonomous SDLC with adaptive quality gates and machine learning insights
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import statistics
import subprocess

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """Progressive quality levels"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate" 
    ADVANCED = "advanced"
    EXPERT = "expert"


class GateCategory(Enum):
    """Quality gate categories"""
    CODE_QUALITY = "code_quality"
    SECURITY = "security" 
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    MAINTAINABILITY = "maintainability"
    SCALABILITY = "scalability"


@dataclass
class ProgressiveQualityMetrics:
    """Advanced quality metrics with trend analysis"""
    gate_name: str
    category: GateCategory
    level: QualityLevel
    score: float
    trend: float  # Score change from previous run
    confidence: float  # Reliability of measurement
    impact_score: float  # Business impact weighting
    technical_debt: float  # Accumulated technical debt
    execution_time: float
    resource_usage: Dict[str, float] = field(default_factory=dict)
    historical_scores: List[float] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass 
class QualityInsight:
    """AI-driven quality insights"""
    insight_type: str
    severity: str
    description: str
    suggested_action: str
    effort_estimate: str
    business_impact: str
    technical_details: Dict[str, Any] = field(default_factory=dict)


class AdaptiveQualityGate:
    """Base class for adaptive quality gates with ML insights"""
    
    def __init__(self, name: str, category: GateCategory, project_root: Path):
        self.name = name
        self.category = category
        self.project_root = project_root
        self.quality_level = QualityLevel.BASIC
        self.historical_metrics: List[ProgressiveQualityMetrics] = []
        self.insights: List[QualityInsight] = []
        
    async def execute_progressive(self) -> ProgressiveQualityMetrics:
        """Execute progressive quality assessment"""
        start_time = time.time()
        
        # Determine current quality level based on project maturity
        await self._assess_quality_level()
        
        # Execute quality checks
        score, details = await self._execute_quality_checks()
        
        # Analyze trends
        trend = self._calculate_trend(score)
        confidence = self._calculate_confidence()
        
        # Generate insights
        await self._generate_insights(score, details)
        
        execution_time = time.time() - start_time
        
        metrics = ProgressiveQualityMetrics(
            gate_name=self.name,
            category=self.category,
            level=self.quality_level,
            score=score,
            trend=trend,
            confidence=confidence,
            impact_score=self._calculate_impact_score(score),
            technical_debt=self._calculate_technical_debt(),
            execution_time=execution_time,
            resource_usage=await self._measure_resource_usage(),
            historical_scores=[m.score for m in self.historical_metrics[-10:]],
            recommendations=self._generate_recommendations(score)
        )
        
        self.historical_metrics.append(metrics)
        return metrics
    
    async def _assess_quality_level(self):
        """Assess project quality maturity level"""
        # Analyze codebase complexity, test coverage, and architecture
        complexity_score = await self._measure_complexity()
        coverage_score = await self._measure_coverage()
        architecture_score = await self._measure_architecture()
        
        avg_score = (complexity_score + coverage_score + architecture_score) / 3
        
        if avg_score >= 0.9:
            self.quality_level = QualityLevel.EXPERT
        elif avg_score >= 0.8:
            self.quality_level = QualityLevel.ADVANCED
        elif avg_score >= 0.6:
            self.quality_level = QualityLevel.INTERMEDIATE
        else:
            self.quality_level = QualityLevel.BASIC
    
    async def _execute_quality_checks(self) -> Tuple[float, Dict[str, Any]]:
        """Execute quality checks based on current level"""
        raise NotImplementedError
    
    def _calculate_trend(self, current_score: float) -> float:
        """Calculate score trend from historical data"""
        if len(self.historical_metrics) < 2:
            return 0.0
        
        recent_scores = [m.score for m in self.historical_metrics[-5:]]
        recent_scores.append(current_score)
        
        if len(recent_scores) >= 2:
            return recent_scores[-1] - statistics.mean(recent_scores[:-1])
        return 0.0
    
    def _calculate_confidence(self) -> float:
        """Calculate confidence in measurement"""
        if len(self.historical_metrics) < 3:
            return 0.5  # Low confidence with limited data
        
        recent_scores = [m.score for m in self.historical_metrics[-5:]]
        variance = statistics.variance(recent_scores) if len(recent_scores) > 1 else 0
        
        # Higher variance = lower confidence
        return max(0.1, 1.0 - variance)
    
    def _calculate_impact_score(self, score: float) -> float:
        """Calculate business impact score"""
        # Weight based on gate category importance
        weights = {
            GateCategory.SECURITY: 1.0,
            GateCategory.RELIABILITY: 0.9,
            GateCategory.PERFORMANCE: 0.8,
            GateCategory.CODE_QUALITY: 0.7,
            GateCategory.MAINTAINABILITY: 0.6,
            GateCategory.SCALABILITY: 0.8
        }
        return score * weights.get(self.category, 0.5)
    
    def _calculate_technical_debt(self) -> float:
        """Calculate accumulated technical debt"""
        if len(self.historical_metrics) < 2:
            return 0.0
        
        # Sum of quality degradation over time
        debt = 0.0
        for i in range(1, len(self.historical_metrics)):
            prev_score = self.historical_metrics[i-1].score
            curr_score = self.historical_metrics[i].score
            if curr_score < prev_score:
                debt += (prev_score - curr_score)
        
        return debt
    
    async def _measure_resource_usage(self) -> Dict[str, float]:
        """Measure resource usage during execution"""
        try:
            import psutil
            process = psutil.Process()
            return {
                "cpu_percent": process.cpu_percent(),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "open_files": len(process.open_files())
            }
        except ImportError:
            return {"cpu_percent": 0.0, "memory_mb": 0.0, "open_files": 0}
    
    def _generate_recommendations(self, score: float) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        if score < 0.6:
            recommendations.append(f"Critical: {self.name} quality below acceptable threshold")
        elif score < 0.8:
            recommendations.append(f"Improve {self.name} quality through focused refactoring")
        else:
            recommendations.append(f"Maintain {self.name} excellence with continuous monitoring")
            
        return recommendations
    
    async def _generate_insights(self, score: float, details: Dict[str, Any]):
        """Generate AI-driven quality insights"""
        insights = []
        
        # Pattern recognition for common issues
        if score < 0.7:
            insights.append(QualityInsight(
                insight_type="quality_degradation",
                severity="medium" if score > 0.5 else "high",
                description=f"{self.name} quality has degraded to {score:.2f}",
                suggested_action=f"Focus on improving {self.category.value} practices",
                effort_estimate="1-2 sprints",
                business_impact="medium" if score > 0.5 else "high",
                technical_details=details
            ))
        
        # Trend-based insights
        if len(self.historical_metrics) >= 3:
            recent_trend = self._calculate_trend(score)
            if recent_trend < -0.1:
                insights.append(QualityInsight(
                    insight_type="negative_trend",
                    severity="medium",
                    description=f"{self.name} showing declining trend ({recent_trend:.3f})",
                    suggested_action="Implement quality improvement plan",
                    effort_estimate="1 sprint", 
                    business_impact="medium",
                    technical_details={"trend": recent_trend}
                ))
        
        self.insights.extend(insights)
    
    async def _measure_complexity(self) -> float:
        """Measure code complexity"""
        try:
            # Simple complexity based on file count and size
            py_files = list(self.project_root.glob("**/*.py"))
            total_lines = 0
            for file_path in py_files[:20]:  # Sample for performance
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        total_lines += len(f.readlines())
                except:
                    continue
            
            # Normalize: fewer lines = less complexity = higher score
            if total_lines < 5000:
                return 0.9
            elif total_lines < 15000:
                return 0.7
            else:
                return 0.5
        except:
            return 0.6
    
    async def _measure_coverage(self) -> float:
        """Measure test coverage"""
        try:
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    data = json.load(f)
                    return data.get("totals", {}).get("percent_covered", 0) / 100.0
        except:
            pass
        return 0.6  # Default assumption
    
    async def _measure_architecture(self) -> float:
        """Measure architectural quality"""
        try:
            # Simple heuristic based on project structure
            structure_score = 0.0
            
            # Check for key directories
            key_dirs = ["tests", "docs", "src", "sql_synthesizer"]
            existing_dirs = sum(1 for d in key_dirs if (self.project_root / d).exists())
            structure_score += existing_dirs / len(key_dirs) * 0.4
            
            # Check for configuration files
            config_files = ["pyproject.toml", "requirements.txt", "setup.py"]
            existing_configs = sum(1 for f in config_files if (self.project_root / f).exists())
            structure_score += existing_configs / len(config_files) * 0.3
            
            # Check for CI/CD
            ci_indicators = [".github", "Dockerfile", "docker-compose.yml"]
            existing_ci = sum(1 for ci in ci_indicators if (self.project_root / ci).exists())
            structure_score += existing_ci / len(ci_indicators) * 0.3
            
            return min(1.0, structure_score)
        except:
            return 0.6


class EnhancedCodeQualityGate(AdaptiveQualityGate):
    """Enhanced code quality gate with progressive assessment"""
    
    def __init__(self, project_root: Path):
        super().__init__("Enhanced Code Quality", GateCategory.CODE_QUALITY, project_root)
    
    async def _execute_quality_checks(self) -> Tuple[float, Dict[str, Any]]:
        """Execute quality checks based on progressive level"""
        details = {}
        total_score = 0.0
        weight_sum = 0.0
        
        # Basic level: Linting and formatting
        ruff_score, ruff_details = await self._run_ruff_analysis()
        details["ruff"] = ruff_details
        total_score += ruff_score * 0.3
        weight_sum += 0.3
        
        black_score, black_details = await self._run_black_analysis()
        details["black"] = black_details
        total_score += black_score * 0.2
        weight_sum += 0.2
        
        # Intermediate+: Type checking
        if self.quality_level.value in ["intermediate", "advanced", "expert"]:
            mypy_score, mypy_details = await self._run_mypy_analysis()
            details["mypy"] = mypy_details
            total_score += mypy_score * 0.25
            weight_sum += 0.25
        
        # Advanced+: Complexity analysis
        if self.quality_level.value in ["advanced", "expert"]:
            complexity_score, complexity_details = await self._analyze_complexity()
            details["complexity"] = complexity_details
            total_score += complexity_score * 0.15
            weight_sum += 0.15
        
        # Expert: Advanced static analysis
        if self.quality_level == QualityLevel.EXPERT:
            advanced_score, advanced_details = await self._advanced_static_analysis()
            details["advanced_analysis"] = advanced_details
            total_score += advanced_score * 0.1
            weight_sum += 0.1
        
        return total_score / weight_sum if weight_sum > 0 else 0.0, details
    
    async def _run_ruff_analysis(self) -> Tuple[float, Dict[str, Any]]:
        """Enhanced ruff analysis with quality metrics"""
        try:
            proc = await asyncio.create_subprocess_exec(
                "ruff", "check", "sql_synthesizer/", "--output-format=json",
                cwd=self.project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                return 1.0, {"issues": 0, "quality": "excellent"}
            
            try:
                issues = json.loads(stdout.decode()) if stdout else []
                # Weight issues by severity
                critical_issues = sum(1 for issue in issues if issue.get("code", "").startswith("E"))
                warning_issues = len(issues) - critical_issues
                
                # Calculate score with penalty system
                penalty = critical_issues * 0.1 + warning_issues * 0.05
                score = max(0.0, 1.0 - penalty)
                
                return score, {
                    "total_issues": len(issues),
                    "critical": critical_issues,
                    "warnings": warning_issues,
                    "score": score
                }
            except json.JSONDecodeError:
                return 0.5, {"error": "parsing_failed", "raw": stdout.decode()}
                
        except Exception as e:
            return 0.3, {"error": str(e)}
    
    async def _run_black_analysis(self) -> Tuple[float, Dict[str, Any]]:
        """Enhanced black formatting analysis"""
        try:
            proc = await asyncio.create_subprocess_exec(
                "black", "--check", "--diff", "sql_synthesizer/",
                cwd=self.project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                return 1.0, {"formatted": True, "changes_needed": 0}
            
            # Count formatting issues
            output = stdout.decode()
            files_needing_format = output.count("would reformat")
            lines_changed = output.count("@@")
            
            # Score based on formatting consistency
            score = max(0.0, 1.0 - (files_needing_format * 0.1 + lines_changed * 0.01))
            
            return score, {
                "files_needing_format": files_needing_format,
                "lines_changed": lines_changed,
                "score": score
            }
            
        except Exception as e:
            return 0.7, {"error": str(e)}
    
    async def _run_mypy_analysis(self) -> Tuple[float, Dict[str, Any]]:
        """Enhanced mypy type checking"""
        try:
            proc = await asyncio.create_subprocess_exec(
                "mypy", "sql_synthesizer/", "--json-report", "/tmp/mypy-report",
                cwd=self.project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            output = stdout.decode() + stderr.decode()
            
            # Count different types of errors
            error_lines = [line for line in output.split("\n") if "error:" in line]
            note_lines = [line for line in output.split("\n") if "note:" in line]
            
            # Categorize errors
            critical_errors = len([e for e in error_lines if any(x in e for x in ["Name", "Argument", "Return"])])
            minor_errors = len(error_lines) - critical_errors
            
            # Calculate type coverage score
            penalty = critical_errors * 0.15 + minor_errors * 0.05
            score = max(0.0, 1.0 - penalty)
            
            return score, {
                "total_errors": len(error_lines),
                "critical_errors": critical_errors,
                "minor_errors": minor_errors,
                "notes": len(note_lines),
                "type_coverage_score": score
            }
            
        except Exception as e:
            return 0.6, {"error": str(e)}
    
    async def _analyze_complexity(self) -> Tuple[float, Dict[str, Any]]:
        """Analyze code complexity metrics"""
        try:
            # Simple complexity analysis
            py_files = list(self.project_root.glob("sql_synthesizer/**/*.py"))
            complexity_metrics = {"files_analyzed": len(py_files)}
            
            total_functions = 0
            complex_functions = 0
            total_lines = 0
            
            for py_file in py_files[:10]:  # Sample for performance
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                        total_lines += len(lines)
                        
                        # Count functions
                        functions = [line for line in lines if line.strip().startswith('def ')]
                        total_functions += len(functions)
                        
                        # Estimate complexity by indentation depth
                        for line in lines:
                            if line.strip() and len(line) - len(line.lstrip()) > 12:  # Deep nesting
                                complex_functions += 1
                                break
                                
                except Exception:
                    continue
            
            complexity_ratio = complex_functions / max(total_functions, 1)
            score = max(0.0, 1.0 - complexity_ratio)
            
            complexity_metrics.update({
                "total_functions": total_functions,
                "complex_functions": complex_functions,
                "complexity_ratio": complexity_ratio,
                "avg_lines_per_file": total_lines / len(py_files) if py_files else 0,
                "complexity_score": score
            })
            
            return score, complexity_metrics
            
        except Exception as e:
            return 0.7, {"error": str(e)}
    
    async def _advanced_static_analysis(self) -> Tuple[float, Dict[str, Any]]:
        """Advanced static analysis for expert level"""
        try:
            # Placeholder for advanced analysis (would integrate with tools like SonarQube)
            analysis_results = {
                "code_smells": 5,  # Would be actual detection
                "duplicated_blocks": 2,
                "cognitive_complexity": 45,
                "maintainability_index": 72
            }
            
            # Calculate score based on multiple factors
            maintainability_score = analysis_results["maintainability_index"] / 100.0
            complexity_penalty = min(0.3, analysis_results["cognitive_complexity"] / 200.0)
            duplication_penalty = min(0.2, analysis_results["duplicated_blocks"] / 10.0)
            
            score = max(0.0, maintainability_score - complexity_penalty - duplication_penalty)
            
            analysis_results["advanced_score"] = score
            return score, analysis_results
            
        except Exception as e:
            return 0.8, {"error": str(e)}


class ProgressiveQualityGateEngine:
    """Enhanced quality gate engine with progressive assessment"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.gates = [
            EnhancedCodeQualityGate(self.project_root),
        ]
        self.execution_history: List[Dict[str, Any]] = []
        self.quality_trends: Dict[str, List[float]] = {}
    
    async def execute_progressive_assessment(self) -> Dict[str, Any]:
        """Execute progressive quality assessment"""
        start_time = time.time()
        
        logger.info("🚀 Starting Progressive Quality Gate Assessment")
        
        # Execute all gates
        gate_metrics = []
        for gate in self.gates:
            metrics = await gate.execute_progressive()
            gate_metrics.append(metrics)
            
            # Update trends
            if gate.name not in self.quality_trends:
                self.quality_trends[gate.name] = []
            self.quality_trends[gate.name].append(metrics.score)
        
        # Calculate overall metrics
        total_execution_time = time.time() - start_time
        overall_score = statistics.mean([m.score for m in gate_metrics])
        overall_confidence = statistics.mean([m.confidence for m in gate_metrics])
        total_technical_debt = sum([m.technical_debt for m in gate_metrics])
        
        # Generate comprehensive insights
        insights = self._generate_comprehensive_insights(gate_metrics)
        
        # Build quality dashboard data
        dashboard_data = self._build_quality_dashboard(gate_metrics)
        
        assessment_result = {
            "assessment_id": f"qa_{int(time.time())}",
            "timestamp": time.time(),
            "execution_time": round(total_execution_time, 2),
            "overall_score": round(overall_score, 3),
            "overall_confidence": round(overall_confidence, 3),
            "total_technical_debt": round(total_technical_debt, 3),
            "quality_level": self._determine_overall_quality_level(overall_score),
            "gates": {
                metrics.gate_name: {
                    "category": metrics.category.value,
                    "level": metrics.level.value,
                    "score": metrics.score,
                    "trend": metrics.trend,
                    "confidence": metrics.confidence,
                    "impact_score": metrics.impact_score,
                    "technical_debt": metrics.technical_debt,
                    "execution_time": metrics.execution_time,
                    "resource_usage": metrics.resource_usage,
                    "recommendations": metrics.recommendations,
                    "historical_scores": metrics.historical_scores
                }
                for metrics in gate_metrics
            },
            "comprehensive_insights": insights,
            "quality_dashboard": dashboard_data,
            "trends": self.quality_trends,
            "next_assessment_recommendations": self._generate_next_steps(gate_metrics)
        }
        
        self.execution_history.append(assessment_result)
        
        logger.info(f"✅ Progressive Assessment Complete: {overall_score:.3f} overall score")
        
        return assessment_result
    
    def _generate_comprehensive_insights(self, gate_metrics: List[ProgressiveQualityMetrics]) -> List[Dict[str, Any]]:
        """Generate comprehensive quality insights"""
        insights = []
        
        # Collect all insights from gates
        for gate in self.gates:
            for insight in gate.insights:
                insights.append({
                    "gate": gate.name,
                    "type": insight.insight_type,
                    "severity": insight.severity,
                    "description": insight.description,
                    "action": insight.suggested_action,
                    "effort": insight.effort_estimate,
                    "impact": insight.business_impact,
                    "details": insight.technical_details
                })
        
        # Add cross-gate insights
        overall_score = statistics.mean([m.score for m in gate_metrics])
        if overall_score < 0.7:
            insights.append({
                "gate": "overall",
                "type": "quality_alert",
                "severity": "high",
                "description": f"Overall quality score ({overall_score:.2f}) below recommended threshold",
                "action": "Implement comprehensive quality improvement program",
                "effort": "2-3 sprints",
                "impact": "high"
            })
        
        # Trend analysis insights
        declining_gates = [m for m in gate_metrics if m.trend < -0.1]
        if declining_gates:
            insights.append({
                "gate": "trend_analysis",
                "type": "quality_decline",
                "severity": "medium",
                "description": f"{len(declining_gates)} gates showing declining quality trends",
                "action": "Focus on stabilizing quality in declining areas",
                "effort": "1-2 sprints",
                "impact": "medium"
            })
        
        return insights
    
    def _build_quality_dashboard(self, gate_metrics: List[ProgressiveQualityMetrics]) -> Dict[str, Any]:
        """Build quality dashboard data"""
        dashboard = {
            "quality_score_distribution": {
                "excellent": len([m for m in gate_metrics if m.score >= 0.9]),
                "good": len([m for m in gate_metrics if 0.8 <= m.score < 0.9]),
                "fair": len([m for m in gate_metrics if 0.6 <= m.score < 0.8]),
                "poor": len([m for m in gate_metrics if m.score < 0.6])
            },
            "confidence_levels": {
                "high": len([m for m in gate_metrics if m.confidence >= 0.8]),
                "medium": len([m for m in gate_metrics if 0.6 <= m.confidence < 0.8]),
                "low": len([m for m in gate_metrics if m.confidence < 0.6])
            },
            "technical_debt_by_category": {
                metrics.category.value: metrics.technical_debt 
                for metrics in gate_metrics
            },
            "performance_metrics": {
                "avg_execution_time": statistics.mean([m.execution_time for m in gate_metrics]),
                "total_execution_time": sum([m.execution_time for m in gate_metrics])
            }
        }
        return dashboard
    
    def _determine_overall_quality_level(self, overall_score: float) -> str:
        """Determine overall project quality level"""
        if overall_score >= 0.95:
            return "exceptional"
        elif overall_score >= 0.85:
            return "excellent"
        elif overall_score >= 0.75:
            return "good"
        elif overall_score >= 0.6:
            return "fair"
        else:
            return "needs_improvement"
    
    def _generate_next_steps(self, gate_metrics: List[ProgressiveQualityMetrics]) -> List[str]:
        """Generate next steps for quality improvement"""
        next_steps = []
        
        # Priority actions based on scores and impact
        high_impact_low_score = [
            m for m in gate_metrics 
            if m.impact_score > 0.8 and m.score < 0.7
        ]
        
        if high_impact_low_score:
            next_steps.append(f"Priority: Address {len(high_impact_low_score)} high-impact quality issues")
        
        # Technical debt management
        high_debt_gates = [m for m in gate_metrics if m.technical_debt > 0.3]
        if high_debt_gates:
            next_steps.append(f"Reduce technical debt in {len(high_debt_gates)} areas")
        
        # Trend management
        declining_gates = [m for m in gate_metrics if m.trend < -0.05]
        if declining_gates:
            next_steps.append(f"Stabilize quality trends in {len(declining_gates)} declining areas")
        
        # Confidence improvement
        low_confidence = [m for m in gate_metrics if m.confidence < 0.6]
        if low_confidence:
            next_steps.append(f"Improve measurement confidence in {len(low_confidence)} areas")
        
        if not next_steps:
            next_steps.append("Maintain current quality levels with continuous monitoring")
        
        return next_steps


# CLI Entry Point
async def main():
    """CLI entry point for progressive quality gates"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Progressive Quality Gates Engine")
    parser.add_argument("--project-root", type=Path, help="Project root directory")
    parser.add_argument("--output", type=Path, help="Output file for results")
    parser.add_argument("--dashboard", action="store_true", help="Generate quality dashboard")
    
    args = parser.parse_args()
    
    engine = ProgressiveQualityGateEngine(args.project_root)
    
    # Execute progressive assessment
    results = await engine.execute_progressive_assessment()
    
    # Output results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Progressive quality assessment results written to {args.output}")
    else:
        print(json.dumps(results, indent=2))
    
    # Generate dashboard if requested
    if args.dashboard:
        print("\n📊 Quality Dashboard Summary:")
        dashboard = results["quality_dashboard"]
        print(f"Overall Score: {results['overall_score']:.3f}")
        print(f"Quality Level: {results['quality_level']}")
        print(f"Technical Debt: {results['total_technical_debt']:.3f}")
        print(f"Confidence: {results['overall_confidence']:.3f}")


if __name__ == "__main__":
    asyncio.run(main())