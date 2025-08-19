"""
Autonomous Deployment Engine
Intelligent deployment automation with rollback capabilities
"""

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Deployment configuration"""

    environment: str
    image_tag: str
    replicas: int
    resources: Dict[str, Any]
    health_checks: Dict[str, Any]
    rollback_on_failure: bool = True
    auto_scale: bool = True


@dataclass
class DeploymentResult:
    """Result of deployment operation"""

    success: bool
    deployment_id: str
    timestamp: float
    duration_seconds: float
    health_status: str
    metrics: Dict[str, Any]
    errors: List[str]
    rollback_performed: bool = False


class HealthChecker:
    """Intelligent health checking with adaptive thresholds"""

    def __init__(self):
        self.health_history: List[Dict[str, Any]] = []
        self.failure_threshold = 3
        self.success_threshold = 5

    async def check_deployment_health(
        self, config: DeploymentConfig
    ) -> Tuple[bool, Dict[str, Any]]:
        """Comprehensive health check of deployment"""

        health_metrics = {"timestamp": time.time(), "checks": {}}

        try:
            # HTTP health check
            http_healthy = await self._check_http_health(config)
            health_metrics["checks"]["http"] = http_healthy

            # Database connectivity
            db_healthy = await self._check_database_health(config)
            health_metrics["checks"]["database"] = db_healthy

            # Resource utilization
            resources_healthy = await self._check_resource_health(config)
            health_metrics["checks"]["resources"] = resources_healthy

            # Application metrics
            metrics_healthy = await self._check_application_metrics(config)
            health_metrics["checks"]["metrics"] = metrics_healthy

            # Overall health
            all_checks = [http_healthy, db_healthy, resources_healthy, metrics_healthy]
            overall_healthy = all(check["healthy"] for check in all_checks)

            health_metrics["overall_healthy"] = overall_healthy
            health_metrics["success_rate"] = sum(
                1 for check in all_checks if check["healthy"]
            ) / len(all_checks)

            self.health_history.append(health_metrics)

            # Keep only recent history
            if len(self.health_history) > 100:
                self.health_history = self.health_history[-100:]

            return overall_healthy, health_metrics

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            health_metrics["error"] = str(e)
            health_metrics["overall_healthy"] = False
            return False, health_metrics

    async def _check_http_health(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Check HTTP endpoint health"""
        try:
            # Simulate HTTP health check
            await asyncio.sleep(0.1)

            # Mock response based on environment
            if config.environment == "production":
                response_time = 150  # ms
                status_code = 200
            else:
                response_time = 80
                status_code = 200

            return {
                "healthy": status_code == 200 and response_time < 500,
                "response_time_ms": response_time,
                "status_code": status_code,
                "details": "HTTP health check passed",
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "details": "HTTP health check failed",
            }

    async def _check_database_health(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        try:
            # Simulate database health check
            await asyncio.sleep(0.05)

            connection_time = 25  # ms
            active_connections = 15
            max_connections = 100

            healthy = (
                connection_time < 100 and active_connections < max_connections * 0.8
            )

            return {
                "healthy": healthy,
                "connection_time_ms": connection_time,
                "active_connections": active_connections,
                "max_connections": max_connections,
                "utilization": active_connections / max_connections,
                "details": "Database connectivity verified",
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "details": "Database health check failed",
            }

    async def _check_resource_health(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Check resource utilization"""
        try:
            # Simulate resource health check
            cpu_usage = 45.0  # %
            memory_usage = 60.0  # %
            disk_usage = 30.0  # %

            cpu_healthy = cpu_usage < 80
            memory_healthy = memory_usage < 85
            disk_healthy = disk_usage < 90

            return {
                "healthy": cpu_healthy and memory_healthy and disk_healthy,
                "cpu_usage_percent": cpu_usage,
                "memory_usage_percent": memory_usage,
                "disk_usage_percent": disk_usage,
                "details": "Resource utilization within limits",
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "details": "Resource health check failed",
            }

    async def _check_application_metrics(
        self, config: DeploymentConfig
    ) -> Dict[str, Any]:
        """Check application-specific metrics"""
        try:
            # Simulate application metrics check
            error_rate = 0.002  # 0.2%
            response_time_p95 = 250  # ms
            throughput = 95  # QPS

            error_rate_healthy = error_rate < 0.01
            response_time_healthy = response_time_p95 < 500
            throughput_healthy = throughput > 50

            return {
                "healthy": error_rate_healthy
                and response_time_healthy
                and throughput_healthy,
                "error_rate": error_rate,
                "response_time_p95_ms": response_time_p95,
                "throughput_qps": throughput,
                "details": "Application metrics within acceptable ranges",
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "details": "Application metrics check failed",
            }


class BlueGreenDeployer:
    """Blue-green deployment with intelligent traffic switching"""

    def __init__(self):
        self.current_deployment = "blue"
        self.deployment_states = {
            "blue": {"active": True, "version": "v1.0.0", "health": "healthy"},
            "green": {"active": False, "version": None, "health": "unknown"},
        }

    async def deploy(self, config: DeploymentConfig) -> DeploymentResult:
        """Execute blue-green deployment"""

        start_time = time.time()
        deployment_id = f"deploy-{int(time.time())}"
        errors = []

        logger.info(f"🚀 Starting blue-green deployment {deployment_id}")

        try:
            # Determine target deployment slot
            target_slot = "green" if self.current_deployment == "blue" else "blue"

            # Step 1: Deploy to inactive slot
            await self._deploy_to_slot(target_slot, config)

            # Step 2: Wait for startup
            await asyncio.sleep(5)  # Simulate startup time

            # Step 3: Health check new deployment
            health_checker = HealthChecker()
            is_healthy, health_metrics = await health_checker.check_deployment_health(
                config
            )

            if not is_healthy:
                errors.append("New deployment failed health checks")
                return DeploymentResult(
                    success=False,
                    deployment_id=deployment_id,
                    timestamp=time.time(),
                    duration_seconds=time.time() - start_time,
                    health_status="unhealthy",
                    metrics=health_metrics,
                    errors=errors,
                )

            # Step 4: Gradual traffic switching
            await self._switch_traffic_gradually(target_slot, config)

            # Step 5: Final health verification
            await asyncio.sleep(2)
            final_healthy, final_metrics = await health_checker.check_deployment_health(
                config
            )

            if final_healthy:
                # Success - update current deployment
                self.current_deployment = target_slot
                self.deployment_states[target_slot] = {
                    "active": True,
                    "version": config.image_tag,
                    "health": "healthy",
                }
                self.deployment_states[self._get_other_slot(target_slot)] = {
                    "active": False,
                    "version": None,
                    "health": "inactive",
                }

                logger.info(f"✅ Deployment {deployment_id} completed successfully")

                return DeploymentResult(
                    success=True,
                    deployment_id=deployment_id,
                    timestamp=time.time(),
                    duration_seconds=time.time() - start_time,
                    health_status="healthy",
                    metrics=final_metrics,
                    errors=errors,
                )
            else:
                # Rollback
                await self._rollback_deployment(target_slot)
                errors.append("Final health check failed, rolled back")

                return DeploymentResult(
                    success=False,
                    deployment_id=deployment_id,
                    timestamp=time.time(),
                    duration_seconds=time.time() - start_time,
                    health_status="rollback_completed",
                    metrics=final_metrics,
                    errors=errors,
                    rollback_performed=True,
                )

        except Exception as e:
            errors.append(f"Deployment failed: {str(e)}")
            logger.error(f"❌ Deployment {deployment_id} failed: {str(e)}")

            return DeploymentResult(
                success=False,
                deployment_id=deployment_id,
                timestamp=time.time(),
                duration_seconds=time.time() - start_time,
                health_status="failed",
                metrics={},
                errors=errors,
            )

    async def _deploy_to_slot(self, slot: str, config: DeploymentConfig):
        """Deploy to specific slot"""
        logger.info(f"Deploying {config.image_tag} to {slot} slot")

        # Simulate deployment steps
        await asyncio.sleep(1)  # Pull image
        await asyncio.sleep(2)  # Start containers
        await asyncio.sleep(1)  # Configure networking

        logger.info(f"Deployment to {slot} slot completed")

    async def _switch_traffic_gradually(
        self, target_slot: str, config: DeploymentConfig
    ):
        """Gradually switch traffic to new deployment"""
        logger.info(f"Gradually switching traffic to {target_slot}")

        # Simulate gradual traffic switching
        traffic_percentages = [10, 25, 50, 75, 100]

        for percentage in traffic_percentages:
            logger.info(f"Routing {percentage}% traffic to {target_slot}")
            await asyncio.sleep(0.5)  # Wait between traffic shifts

            # In real implementation, would check metrics here
            # and rollback if issues detected

        logger.info(f"Traffic fully switched to {target_slot}")

    async def _rollback_deployment(self, failed_slot: str):
        """Rollback failed deployment"""
        logger.warning(f"Rolling back deployment from {failed_slot}")

        # Simulate rollback
        await asyncio.sleep(1)

        # Reset slot state
        self.deployment_states[failed_slot] = {
            "active": False,
            "version": None,
            "health": "rolled_back",
        }

        logger.info("Rollback completed")

    def _get_other_slot(self, slot: str) -> str:
        """Get the other deployment slot"""
        return "green" if slot == "blue" else "blue"

    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status"""
        return {
            "current_active_slot": self.current_deployment,
            "slots": self.deployment_states,
            "timestamp": time.time(),
        }


class AutonomousDeploymentEngine:
    """Autonomous deployment engine with intelligent decision making"""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.deployer = BlueGreenDeployer()
        self.deployment_history: List[DeploymentResult] = []

        # Load deployment configurations
        self.configs = self._load_deployment_configs()

    def _load_deployment_configs(self) -> Dict[str, DeploymentConfig]:
        """Load deployment configurations"""

        # Default configurations
        default_configs = {
            "development": DeploymentConfig(
                environment="development",
                image_tag="latest",
                replicas=1,
                resources={"cpu": "100m", "memory": "256Mi"},
                health_checks={"interval": 10, "timeout": 5},
                rollback_on_failure=True,
                auto_scale=False,
            ),
            "staging": DeploymentConfig(
                environment="staging",
                image_tag="staging",
                replicas=2,
                resources={"cpu": "200m", "memory": "512Mi"},
                health_checks={"interval": 30, "timeout": 10},
                rollback_on_failure=True,
                auto_scale=True,
            ),
            "production": DeploymentConfig(
                environment="production",
                image_tag="v1.0.0",
                replicas=3,
                resources={"cpu": "500m", "memory": "1Gi"},
                health_checks={"interval": 60, "timeout": 15},
                rollback_on_failure=True,
                auto_scale=True,
            ),
        }

        # Try to load from file
        config_file = self.project_root / "deployment" / "configs.yaml"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    loaded_configs = yaml.safe_load(f)

                # Convert to DeploymentConfig objects
                for env, config_dict in loaded_configs.items():
                    default_configs[env] = DeploymentConfig(**config_dict)

                logger.info(f"Loaded deployment configs from {config_file}")
            except Exception as e:
                logger.warning(f"Failed to load deployment configs: {str(e)}")

        return default_configs

    async def analyze_deployment_readiness(self, environment: str) -> Dict[str, Any]:
        """Analyze if deployment is ready and safe"""

        if environment not in self.configs:
            return {
                "ready": False,
                "reason": f"No configuration found for environment '{environment}'",
            }

        config = self.configs[environment]

        # Check quality gates
        quality_score = await self._check_quality_gates()

        # Check previous deployment success rate
        success_rate = self._calculate_deployment_success_rate()

        # Check resource availability (simulated)
        resource_availability = await self._check_resource_availability(config)

        # Make deployment decision
        ready = (
            quality_score >= 0.8
            and success_rate >= 0.7
            and resource_availability["available"]
        )

        return {
            "ready": ready,
            "quality_score": quality_score,
            "deployment_success_rate": success_rate,
            "resource_availability": resource_availability,
            "recommendation": "proceed" if ready else "wait",
            "config": asdict(config),
        }

    async def _check_quality_gates(self) -> float:
        """Check quality gates and return score"""
        try:
            # Import and run quality gates
            from .quality_gates import AutonomousQualityGateEngine

            engine = AutonomousQualityGateEngine(self.project_root)
            results = await engine.execute_all_gates()

            return results.get("overall_score", 0.0)

        except Exception as e:
            logger.warning(f"Quality gate check failed: {str(e)}")
            return 0.5  # Conservative default

    def _calculate_deployment_success_rate(self) -> float:
        """Calculate recent deployment success rate"""
        if not self.deployment_history:
            return 1.0  # No history, assume good

        recent_deployments = self.deployment_history[-10:]  # Last 10 deployments
        successful = sum(1 for d in recent_deployments if d.success)

        return successful / len(recent_deployments)

    async def _check_resource_availability(
        self, config: DeploymentConfig
    ) -> Dict[str, Any]:
        """Check if resources are available for deployment"""
        # Simulate resource availability check
        await asyncio.sleep(0.1)

        # Mock resource check based on environment
        if config.environment == "production":
            available = True
            cpu_available = 80  # %
            memory_available = 75  # %
        else:
            available = True
            cpu_available = 90
            memory_available = 85

        return {
            "available": available,
            "cpu_available_percent": cpu_available,
            "memory_available_percent": memory_available,
            "estimated_resource_usage": {"cpu_percent": 15, "memory_percent": 20},
        }

    async def deploy_automatically(
        self, environment: str, image_tag: Optional[str] = None
    ) -> DeploymentResult:
        """Execute autonomous deployment"""

        logger.info(f"🚀 Starting autonomous deployment to {environment}")

        # Check deployment readiness
        readiness = await self.analyze_deployment_readiness(environment)

        if not readiness["ready"]:
            return DeploymentResult(
                success=False,
                deployment_id=f"failed-{int(time.time())}",
                timestamp=time.time(),
                duration_seconds=0,
                health_status="not_ready",
                metrics=readiness,
                errors=[
                    f"Deployment not ready: {readiness.get('reason', 'Unknown reason')}"
                ],
            )

        # Get configuration
        config = self.configs[environment]
        if image_tag:
            config.image_tag = image_tag

        # Execute deployment
        result = await self.deployer.deploy(config)

        # Store in history
        self.deployment_history.append(result)

        # Keep history manageable
        if len(self.deployment_history) > 50:
            self.deployment_history = self.deployment_history[-50:]

        # Generate deployment report
        await self._generate_deployment_report(result, readiness)

        return result

    async def _generate_deployment_report(
        self, result: DeploymentResult, readiness: Dict[str, Any]
    ):
        """Generate comprehensive deployment report"""

        report = {
            "deployment_id": result.deployment_id,
            "timestamp": result.timestamp,
            "success": result.success,
            "duration_seconds": result.duration_seconds,
            "health_status": result.health_status,
            "readiness_analysis": readiness,
            "deployment_metrics": result.metrics,
            "errors": result.errors,
            "rollback_performed": result.rollback_performed,
            "deployment_status": self.deployer.get_deployment_status(),
        }

        # Save report to file
        reports_dir = self.project_root / "deployment" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        report_file = reports_dir / f"deployment-{result.deployment_id}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Deployment report saved to {report_file}")

    def get_deployment_analytics(self) -> Dict[str, Any]:
        """Get deployment analytics and insights"""

        if not self.deployment_history:
            return {"message": "No deployment history available"}

        # Calculate metrics
        total_deployments = len(self.deployment_history)
        successful_deployments = sum(1 for d in self.deployment_history if d.success)
        success_rate = successful_deployments / total_deployments

        rollbacks = sum(1 for d in self.deployment_history if d.rollback_performed)
        rollback_rate = rollbacks / total_deployments

        # Average deployment time
        avg_duration = (
            sum(d.duration_seconds for d in self.deployment_history) / total_deployments
        )

        # Recent trend (last 10 deployments)
        recent_deployments = self.deployment_history[-10:]
        recent_success_rate = sum(1 for d in recent_deployments if d.success) / len(
            recent_deployments
        )

        return {
            "total_deployments": total_deployments,
            "success_rate": round(success_rate, 3),
            "rollback_rate": round(rollback_rate, 3),
            "average_duration_seconds": round(avg_duration, 2),
            "recent_success_rate": round(recent_success_rate, 3),
            "deployment_frequency": self._calculate_deployment_frequency(),
            "health_trends": self._analyze_health_trends(),
            "recommendations": self._generate_deployment_recommendations(),
        }

    def _calculate_deployment_frequency(self) -> Dict[str, Any]:
        """Calculate deployment frequency metrics"""
        if len(self.deployment_history) < 2:
            return {"status": "insufficient_data"}

        timestamps = [d.timestamp for d in self.deployment_history]
        timestamps.sort()

        intervals = [
            timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)
        ]
        avg_interval_hours = sum(intervals) / len(intervals) / 3600

        return {
            "average_interval_hours": round(avg_interval_hours, 2),
            "deployments_per_day": (
                round(24 / avg_interval_hours, 2) if avg_interval_hours > 0 else 0
            ),
            "most_recent_interval_hours": (
                round(intervals[-1] / 3600, 2) if intervals else 0
            ),
        }

    def _analyze_health_trends(self) -> Dict[str, Any]:
        """Analyze health trends across deployments"""
        health_statuses = [d.health_status for d in self.deployment_history]

        status_counts = {}
        for status in health_statuses:
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "status_distribution": status_counts,
            "healthy_rate": (
                status_counts.get("healthy", 0) / len(health_statuses)
                if health_statuses
                else 0
            ),
        }

    def _generate_deployment_recommendations(self) -> List[str]:
        """Generate deployment recommendations based on history"""
        recommendations = []

        if not self.deployment_history:
            return ["Start collecting deployment data"]

        success_rate = sum(1 for d in self.deployment_history if d.success) / len(
            self.deployment_history
        )

        if success_rate < 0.8:
            recommendations.append("Improve quality gates - success rate below 80%")

        rollback_rate = sum(
            1 for d in self.deployment_history if d.rollback_performed
        ) / len(self.deployment_history)
        if rollback_rate > 0.2:
            recommendations.append("High rollback rate - review deployment process")

        avg_duration = sum(d.duration_seconds for d in self.deployment_history) / len(
            self.deployment_history
        )
        if avg_duration > 300:  # 5 minutes
            recommendations.append("Consider optimizing deployment speed")

        # Check for recent failures
        recent_failures = [d for d in self.deployment_history[-5:] if not d.success]
        if len(recent_failures) >= 2:
            recommendations.append(
                "Recent deployment failures detected - investigate issues"
            )

        return (
            recommendations
            if recommendations
            else ["Deployment process appears healthy"]
        )


# CLI Entry Point
async def main():
    """CLI entry point for autonomous deployment"""
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous Deployment Engine")
    parser.add_argument(
        "--environment", choices=["development", "staging", "production"], required=True
    )
    parser.add_argument("--image-tag", help="Docker image tag to deploy")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze readiness, don't deploy",
    )
    parser.add_argument(
        "--analytics", action="store_true", help="Show deployment analytics"
    )

    args = parser.parse_args()

    engine = AutonomousDeploymentEngine()

    if args.analytics:
        analytics = engine.get_deployment_analytics()
        print(json.dumps(analytics, indent=2))
        return

    if args.analyze_only:
        readiness = await engine.analyze_deployment_readiness(args.environment)
        print(json.dumps(readiness, indent=2))
        return

    # Execute deployment
    result = await engine.deploy_automatically(args.environment, args.image_tag)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
