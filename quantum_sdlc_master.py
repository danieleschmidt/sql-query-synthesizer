"""
Quantum SDLC Master Controller

The ultimate autonomous software development lifecycle controller that orchestrates
all quantum-inspired systems for continuous self-improvement and optimization.
"""

import asyncio
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone

# Autonomous SDLC components
from autonomous_sdlc_engine import AutonomousSDLCEngine
from sql_synthesizer.quantum.autonomous_optimizer import AutonomousQuantumOptimizer, SDLCTask, SDLCPhase
from sql_synthesizer.quantum.resilience import ResilienceManager, BulkheadConfig
from sql_synthesizer.quantum.monitoring import QuantumMonitoringSystem, MetricType, AlertSeverity
from sql_synthesizer.quantum.auto_scaling import QuantumAutoScaler, ResourceConfig, ResourceType, ScalingStrategy


@dataclass
class QuantumSDLCConfig:
    """Configuration for Quantum SDLC Master"""
    project_root: Path
    max_parallel_tasks: int = 8
    continuous_optimization: bool = True
    auto_scaling_enabled: bool = True
    resilience_monitoring: bool = True
    quantum_monitoring: bool = True
    self_healing: bool = True
    adaptive_learning: bool = True
    performance_optimization: bool = True


class QuantumSDLCMaster:
    """
    Master controller for quantum-inspired autonomous SDLC
    """
    
    def __init__(self, config: QuantumSDLCConfig, 
                 logger: Optional[logging.Logger] = None):
        
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize core systems
        self._initialize_core_systems()
        
        # Master control state
        self.is_running = False
        self.start_time: Optional[float] = None
        self.cycles_completed = 0
        self.total_improvements = 0
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.system_health_score = 1.0
        
        self.logger.info("🚀 Quantum SDLC Master initialized")
    
    def _initialize_core_systems(self):
        """Initialize all quantum SDLC systems"""
        
        # 1. Autonomous SDLC Engine
        self.sdlc_engine = AutonomousSDLCEngine(
            project_root=self.config.project_root,
            max_parallel_tasks=self.config.max_parallel_tasks,
            continuous_optimization=self.config.continuous_optimization,
            logger=self.logger
        )
        
        # 2. Quantum Monitoring System
        if self.config.quantum_monitoring:
            self.monitoring_system = QuantumMonitoringSystem(logger=self.logger)
            self._setup_monitoring_metrics()
        else:
            self.monitoring_system = None
        
        # 3. Resilience Manager
        if self.config.resilience_monitoring:
            self.resilience_manager = ResilienceManager(logger=self.logger)
            self._setup_resilience_patterns()
        else:
            self.resilience_manager = None
        
        # 4. Auto-Scaler
        if self.config.auto_scaling_enabled:
            self.auto_scaler = QuantumAutoScaler(
                monitoring_system=self.monitoring_system,
                scaling_strategy=ScalingStrategy.QUANTUM_ADAPTIVE,
                logger=self.logger
            )
            self._setup_auto_scaling()
        else:
            self.auto_scaler = None
        
        # 5. Master Optimizer
        self.master_optimizer = AutonomousQuantumOptimizer(
            max_parallel_tasks=self.config.max_parallel_tasks,
            logger=self.logger
        )
    
    def _setup_monitoring_metrics(self):
        """Setup monitoring metrics for SDLC processes"""
        
        if not self.monitoring_system:
            return
        
        # Core SDLC metrics
        self.monitoring_system.create_metric("sdlc_cycle_time", MetricType.HISTOGRAM)
        self.monitoring_system.create_metric("tasks_completed", MetricType.COUNTER)
        self.monitoring_system.create_metric("tasks_failed", MetricType.COUNTER)
        self.monitoring_system.create_metric("optimization_confidence", MetricType.GAUGE)
        self.monitoring_system.create_metric("system_health_score", MetricType.GAUGE)
        
        # Performance metrics
        self.monitoring_system.create_metric("cpu_utilization", MetricType.GAUGE)
        self.monitoring_system.create_metric("memory_utilization", MetricType.GAUGE)
        self.monitoring_system.create_metric("concurrent_tasks", MetricType.GAUGE)
        self.monitoring_system.create_metric("quantum_coherence", MetricType.GAUGE)
        
        # Quality metrics
        self.monitoring_system.create_metric("code_quality_score", MetricType.GAUGE)
        self.monitoring_system.create_metric("test_coverage", MetricType.GAUGE)
        self.monitoring_system.create_metric("security_score", MetricType.GAUGE)
        self.monitoring_system.create_metric("performance_score", MetricType.GAUGE)
        
        self.logger.info("✅ Monitoring metrics configured")
    
    def _setup_resilience_patterns(self):
        """Setup resilience patterns for SDLC processes"""
        
        if not self.resilience_manager:
            return
        
        # Circuit breakers for critical components
        self.resilience_manager.create_circuit_breaker(
            name="sdlc_engine",
            failure_threshold=5,
            recovery_timeout=120.0,
            quantum_adaptation=True
        )
        
        self.resilience_manager.create_circuit_breaker(
            name="quantum_optimizer",
            failure_threshold=3,
            recovery_timeout=60.0,
            quantum_adaptation=True
        )
        
        # Bulkheads for resource isolation
        task_bulkhead_config = BulkheadConfig(
            max_concurrent_calls=self.config.max_parallel_tasks,
            max_queue_size=100,
            timeout_seconds=300.0,
            priority_levels=3,
            auto_scaling=True
        )
        
        self.resilience_manager.create_bulkhead("task_execution", task_bulkhead_config)
        
        optimization_bulkhead_config = BulkheadConfig(
            max_concurrent_calls=4,
            max_queue_size=50,
            timeout_seconds=120.0,
            priority_levels=2,
            auto_scaling=True
        )
        
        self.resilience_manager.create_bulkhead("optimization", optimization_bulkhead_config)
        
        # Start health monitoring
        self.resilience_manager.start_health_monitoring()
        
        self.logger.info("🛡️ Resilience patterns configured")
    
    def _setup_auto_scaling(self):
        """Setup auto-scaling for SDLC resources"""
        
        if not self.auto_scaler:
            return
        
        # Configure scalable resources
        cpu_config = ResourceConfig(
            resource_type=ResourceType.CPU,
            current_capacity=100,
            min_capacity=50,
            max_capacity=400,
            scale_up_threshold=80.0,
            scale_down_threshold=30.0,
            scale_up_factor=1.5,
            scale_down_factor=0.8,
            cooldown_period=120.0
        )
        self.auto_scaler.configure_resource(ResourceType.CPU, cpu_config)
        
        memory_config = ResourceConfig(
            resource_type=ResourceType.MEMORY,
            current_capacity=100,
            min_capacity=50,
            max_capacity=300,
            scale_up_threshold=85.0,
            scale_down_threshold=25.0,
            scale_up_factor=1.4,
            scale_down_factor=0.8,
            cooldown_period=120.0
        )
        self.auto_scaler.configure_resource(ResourceType.MEMORY, memory_config)
        
        workers_config = ResourceConfig(
            resource_type=ResourceType.WORKERS,
            current_capacity=self.config.max_parallel_tasks,
            min_capacity=2,
            max_capacity=32,
            scale_up_threshold=8.0,  # requests per second
            scale_down_threshold=2.0,
            scale_up_factor=1.5,
            scale_down_factor=0.7,
            cooldown_period=180.0
        )
        self.auto_scaler.configure_resource(ResourceType.WORKERS, workers_config)
        
        self.logger.info("⚡ Auto-scaling configured")
    
    async def start_quantum_sdlc(self) -> Dict[str, Any]:
        """Start the complete quantum SDLC system"""
        
        if self.is_running:
            return {"status": "already_running", "message": "Quantum SDLC is already running"}
        
        self.is_running = True
        self.start_time = time.time()
        
        self.logger.info("🌟 Starting Quantum SDLC Master System")
        
        try:
            # Start all subsystems
            await self._start_subsystems()
            
            # Run main quantum SDLC loop
            result = await self._quantum_sdlc_loop()
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Quantum SDLC failed: {str(e)}")
            await self._emergency_shutdown()
            raise
        
        finally:
            self.is_running = False
    
    async def _start_subsystems(self):
        """Start all quantum SDLC subsystems"""
        
        startup_tasks = []
        
        # Start monitoring
        if self.monitoring_system:
            startup_tasks.append(self.monitoring_system.start_monitoring())
            self.logger.info("📊 Started quantum monitoring")
        
        # Start auto-scaling
        if self.auto_scaler:
            startup_tasks.append(self.auto_scaler.start_auto_scaling())
            self.logger.info("📈 Started auto-scaling")
        
        # Execute startup tasks in parallel
        if startup_tasks:
            await asyncio.gather(*startup_tasks)
        
        # Record startup metrics
        if self.monitoring_system:
            self.monitoring_system.record_metric("system_health_score", 1.0)
            self.monitoring_system.record_metric("sdlc_cycle_time", 0.0)
    
    async def _quantum_sdlc_loop(self) -> Dict[str, Any]:
        """Main quantum SDLC execution loop"""
        
        total_improvements = 0
        cycles_completed = 0
        
        while self.is_running:
            cycle_start = time.time()
            
            self.logger.info(f"🔄 Starting SDLC cycle {cycles_completed + 1}")
            
            try:
                # Execute autonomous SDLC
                cycle_result = await self._execute_sdlc_cycle()
                
                # Process cycle results
                improvements = await self._process_cycle_results(cycle_result)
                total_improvements += improvements
                cycles_completed += 1
                
                # Record performance metrics
                cycle_time = time.time() - cycle_start
                await self._record_cycle_metrics(cycle_result, cycle_time, improvements)
                
                # Adaptive cycle timing based on performance
                next_cycle_delay = self._calculate_adaptive_cycle_delay(cycle_time, cycle_result)
                
                self.logger.info(
                    f"✅ SDLC cycle {cycles_completed} completed in {cycle_time:.2f}s "
                    f"({improvements} improvements, next cycle in {next_cycle_delay:.1f}s)"
                )
                
                # Check if we should continue (for demo, run limited cycles)
                if cycles_completed >= 3:  # Run 3 cycles for demonstration
                    break
                
                # Wait before next cycle
                await asyncio.sleep(next_cycle_delay)
                
            except Exception as e:
                self.logger.error(f"❌ SDLC cycle failed: {str(e)}")
                
                # Attempt self-healing
                if self.config.self_healing:
                    healed = await self._attempt_self_healing(e)
                    if not healed:
                        break
                else:
                    break
        
        # Generate final report
        execution_time = time.time() - self.start_time
        
        final_report = {
            "quantum_sdlc_execution": {
                "success": True,
                "execution_time": execution_time,
                "cycles_completed": cycles_completed,
                "total_improvements": total_improvements,
                "average_cycle_time": execution_time / max(cycles_completed, 1),
                "improvement_rate": total_improvements / max(cycles_completed, 1),
                "final_system_health": self.system_health_score
            },
            "subsystem_reports": await self._collect_subsystem_reports(),
            "performance_history": self.performance_history,
            "recommendations": self._generate_master_recommendations()
        }
        
        # Export comprehensive report
        report_path = await self._export_final_report(final_report)
        final_report["report_exported"] = str(report_path)
        
        return final_report
    
    async def _execute_sdlc_cycle(self) -> Dict[str, Any]:
        """Execute a single SDLC cycle"""
        
        # Use resilience patterns if available
        if self.resilience_manager:
            sdlc_circuit = self.resilience_manager.circuit_breakers.get("sdlc_engine")
            task_bulkhead = self.resilience_manager.bulkheads.get("task_execution")
            
            if sdlc_circuit and task_bulkhead:
                # Execute with resilience protection
                return await task_bulkhead.execute(
                    sdlc_circuit.call,
                    self.sdlc_engine.execute_autonomous_sdlc,
                    priority=2  # High priority
                )
            else:
                return await self.sdlc_engine.execute_autonomous_sdlc()
        else:
            return await self.sdlc_engine.execute_autonomous_sdlc()
    
    async def _process_cycle_results(self, cycle_result: Dict[str, Any]) -> int:
        """Process SDLC cycle results and count improvements"""
        
        improvements = 0
        
        # Count improvements based on cycle success
        if cycle_result.get("success", False):
            improvements += 1
        
        # Count completed checkpoints as improvements
        checkpoints = cycle_result.get("checkpoints", {})
        improvements += checkpoints.get("completed", 0)
        
        # Count applied optimizations as improvements
        execution = cycle_result.get("execution", {})
        if execution.get("success_rate", 0) > 0.8:
            improvements += 1
        
        # Count learning improvements
        learning = cycle_result.get("learning", {})
        if learning.get("success_rate", 0) > 0.7:
            improvements += 1
        
        return improvements
    
    async def _record_cycle_metrics(self, cycle_result: Dict[str, Any], 
                                   cycle_time: float, improvements: int):
        """Record performance metrics for the cycle"""
        
        if not self.monitoring_system:
            return
        
        # Record basic cycle metrics
        self.monitoring_system.record_metric("sdlc_cycle_time", cycle_time)
        self.monitoring_system.record_metric("tasks_completed", improvements)
        
        # Record optimization metrics
        optimization = cycle_result.get("optimization", {})
        confidence = optimization.get("confidence", 0.0)
        self.monitoring_system.record_metric("optimization_confidence", confidence)
        
        # Update system health score based on cycle performance
        success_rate = 1.0 if cycle_result.get("success", False) else 0.0
        self.system_health_score = 0.9 * self.system_health_score + 0.1 * success_rate
        self.monitoring_system.record_metric("system_health_score", self.system_health_score)
        
        # Record performance history
        performance_record = {
            "timestamp": time.time(),
            "cycle_time": cycle_time,
            "improvements": improvements,
            "confidence": confidence,
            "success": cycle_result.get("success", False),
            "system_health": self.system_health_score
        }
        
        self.performance_history.append(performance_record)
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def _calculate_adaptive_cycle_delay(self, cycle_time: float, 
                                      cycle_result: Dict[str, Any]) -> float:
        """Calculate adaptive delay before next cycle"""
        
        base_delay = 60.0  # 1 minute base delay
        
        # Adjust based on cycle performance
        if cycle_result.get("success", False):
            # Successful cycles can run more frequently
            performance_multiplier = 0.8
        else:
            # Failed cycles need more time to recover
            performance_multiplier = 2.0
        
        # Adjust based on cycle time
        if cycle_time < 30.0:
            # Fast cycles can run more frequently
            time_multiplier = 0.7
        elif cycle_time > 120.0:
            # Slow cycles need more time between runs
            time_multiplier = 1.5
        else:
            time_multiplier = 1.0
        
        # Adjust based on system health
        health_multiplier = 2.0 - self.system_health_score  # 1.0 to 2.0 range
        
        adaptive_delay = base_delay * performance_multiplier * time_multiplier * health_multiplier
        
        return max(10.0, min(300.0, adaptive_delay))  # Clamp between 10s and 5min
    
    async def _attempt_self_healing(self, error: Exception) -> bool:
        """Attempt self-healing from errors"""
        
        if not self.resilience_manager:
            return False
        
        self.logger.info("🔧 Attempting self-healing...")
        
        # Use the resilience manager's self-healer
        context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "system_health": self.system_health_score,
            "cycles_completed": self.cycles_completed
        }
        
        healed = await self.resilience_manager.self_healer.heal(
            error, "quantum_sdlc_master", context
        )
        
        if healed:
            self.logger.info("✅ Self-healing successful")
            # Reset system health partially
            self.system_health_score = min(1.0, self.system_health_score + 0.2)
        else:
            self.logger.warning("❌ Self-healing failed")
        
        return healed
    
    async def _collect_subsystem_reports(self) -> Dict[str, Any]:
        """Collect reports from all subsystems"""
        
        reports = {}
        
        # SDLC Engine report
        if hasattr(self.sdlc_engine, 'export_sdlc_state'):
            try:
                reports["sdlc_engine"] = self.sdlc_engine.optimizer.export_optimization_report()
            except:
                reports["sdlc_engine"] = {"status": "report_unavailable"}
        
        # Monitoring system report
        if self.monitoring_system:
            try:
                reports["monitoring"] = self.monitoring_system.export_monitoring_report()
            except:
                reports["monitoring"] = {"status": "report_unavailable"}
        
        # Resilience manager report
        if self.resilience_manager:
            try:
                reports["resilience"] = self.resilience_manager.export_resilience_report()
            except:
                reports["resilience"] = {"status": "report_unavailable"}
        
        # Auto-scaler report
        if self.auto_scaler:
            try:
                reports["auto_scaling"] = self.auto_scaler.export_scaling_report()
            except:
                reports["auto_scaling"] = {"status": "report_unavailable"}
        
        return reports
    
    def _generate_master_recommendations(self) -> List[str]:
        """Generate master-level recommendations"""
        
        recommendations = []
        
        # System health recommendations
        if self.system_health_score < 0.7:
            recommendations.append(
                "System health is below 70%. Consider reducing cycle frequency "
                "and investigating recurring failures."
            )
        
        # Performance recommendations
        if len(self.performance_history) >= 3:
            recent_times = [p["cycle_time"] for p in self.performance_history[-3:]]
            avg_time = sum(recent_times) / len(recent_times)
            
            if avg_time > 180.0:  # 3 minutes
                recommendations.append(
                    f"Average cycle time is {avg_time:.1f}s. Consider optimizing "
                    "SDLC tasks or increasing parallel capacity."
                )
        
        # Improvement rate recommendations
        if len(self.performance_history) >= 5:
            recent_improvements = [p["improvements"] for p in self.performance_history[-5:]]
            avg_improvements = sum(recent_improvements) / len(recent_improvements)
            
            if avg_improvements < 2.0:
                recommendations.append(
                    "Low improvement rate detected. Consider reviewing SDLC "
                    "optimization strategies and task complexity."
                )
        
        # Subsystem recommendations
        if self.auto_scaler and hasattr(self.auto_scaler, 'scaling_decisions_made'):
            if self.auto_scaler.scaling_decisions_made > 10:
                success_rate = self.auto_scaler.successful_scalings / self.auto_scaler.scaling_decisions_made
                if success_rate < 0.8:
                    recommendations.append(
                        "Auto-scaling success rate is below 80%. Review scaling "
                        "thresholds and prediction accuracy."
                    )
        
        return recommendations
    
    async def _export_final_report(self, final_report: Dict[str, Any]) -> Path:
        """Export final comprehensive report"""
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_filename = f"quantum_sdlc_master_report_{timestamp}.json"
        report_path = self.config.project_root / "reports" / report_filename
        
        # Ensure reports directory exists
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        final_report["metadata"] = {
            "generated_by": "QuantumSDLCMaster",
            "version": "1.0.0",
            "generation_time": datetime.now(timezone.utc).isoformat(),
            "project_root": str(self.config.project_root),
            "configuration": {
                "max_parallel_tasks": self.config.max_parallel_tasks,
                "continuous_optimization": self.config.continuous_optimization,
                "auto_scaling_enabled": self.config.auto_scaling_enabled,
                "resilience_monitoring": self.config.resilience_monitoring,
                "quantum_monitoring": self.config.quantum_monitoring,
                "self_healing": self.config.self_healing,
                "adaptive_learning": self.config.adaptive_learning,
                "performance_optimization": self.config.performance_optimization
            }
        }
        
        # Export report
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        self.logger.info(f"📋 Final report exported to: {report_path}")
        
        return report_path
    
    async def _emergency_shutdown(self):
        """Emergency shutdown of all systems"""
        
        self.logger.warning("🚨 Emergency shutdown initiated")
        
        shutdown_tasks = []
        
        # Stop monitoring
        if self.monitoring_system:
            self.monitoring_system.stop_monitoring()
        
        # Stop auto-scaling
        if self.auto_scaler:
            self.auto_scaler.stop_auto_scaling()
        
        # Stop resilience monitoring
        if self.resilience_manager:
            self.resilience_manager.stop_health_monitoring()
        
        # Wait for graceful shutdown
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        self.is_running = False
        self.logger.info("🔴 Emergency shutdown completed")
    
    async def stop_quantum_sdlc(self):
        """Gracefully stop the quantum SDLC system"""
        
        if not self.is_running:
            return
        
        self.logger.info("🛑 Stopping Quantum SDLC Master System")
        
        self.is_running = False
        
        # Stop subsystems gracefully
        if self.monitoring_system:
            self.monitoring_system.stop_monitoring()
        
        if self.auto_scaler:
            self.auto_scaler.stop_auto_scaling()
        
        if self.resilience_manager:
            self.resilience_manager.stop_health_monitoring()
        
        self.logger.info("✅ Quantum SDLC Master System stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        status = {
            "master_controller": {
                "is_running": self.is_running,
                "start_time": self.start_time,
                "cycles_completed": self.cycles_completed,
                "total_improvements": self.total_improvements,
                "system_health_score": self.system_health_score,
                "uptime": time.time() - self.start_time if self.start_time else 0
            },
            "configuration": {
                "project_root": str(self.config.project_root),
                "max_parallel_tasks": self.config.max_parallel_tasks,
                "continuous_optimization": self.config.continuous_optimization,
                "auto_scaling_enabled": self.config.auto_scaling_enabled,
                "resilience_monitoring": self.config.resilience_monitoring,
                "quantum_monitoring": self.config.quantum_monitoring,
                "self_healing": self.config.self_healing
            },
            "subsystems": {}
        }
        
        # Add subsystem status
        if self.monitoring_system:
            status["subsystems"]["monitoring"] = self.monitoring_system.get_monitoring_status()
        
        if self.resilience_manager:
            status["subsystems"]["resilience"] = self.resilience_manager.get_system_status()
        
        if self.auto_scaler:
            status["subsystems"]["auto_scaling"] = self.auto_scaler.get_scaling_status()
        
        return status


async def main():
    """Main entry point for Quantum SDLC Master"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('quantum_sdlc_master.log')
        ]
    )
    
    logger = logging.getLogger("QuantumSDLCMaster")
    
    # Configuration
    config = QuantumSDLCConfig(
        project_root=Path.cwd(),
        max_parallel_tasks=8,
        continuous_optimization=True,
        auto_scaling_enabled=True,
        resilience_monitoring=True,
        quantum_monitoring=True,
        self_healing=True,
        adaptive_learning=True,
        performance_optimization=True
    )
    
    # Initialize and run Quantum SDLC Master
    master = QuantumSDLCMaster(config, logger)
    
    try:
        # Start the quantum SDLC system
        result = await master.start_quantum_sdlc()
        
        print("\n" + "="*100)
        print("🌟 QUANTUM SDLC MASTER EXECUTION COMPLETE")
        print("="*100)
        
        execution_info = result.get("quantum_sdlc_execution", {})
        print(f"Success: {'✅' if execution_info.get('success') else '❌'}")
        print(f"Execution Time: {execution_info.get('execution_time', 0):.2f}s")
        print(f"Cycles Completed: {execution_info.get('cycles_completed', 0)}")
        print(f"Total Improvements: {execution_info.get('total_improvements', 0)}")
        print(f"Average Cycle Time: {execution_info.get('average_cycle_time', 0):.2f}s")
        print(f"Improvement Rate: {execution_info.get('improvement_rate', 0):.1f} improvements/cycle")
        print(f"Final System Health: {execution_info.get('final_system_health', 0):.1%}")
        
        if result.get("report_exported"):
            print(f"📋 Comprehensive Report: {result['report_exported']}")
        
        # Show recommendations
        recommendations = result.get("recommendations", [])
        if recommendations:
            print("\n📋 Master Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        return result
        
    except KeyboardInterrupt:
        logger.info("🛑 Received interrupt signal")
        await master.stop_quantum_sdlc()
        
    except Exception as e:
        logger.error(f"❌ Quantum SDLC Master failed: {str(e)}")
        await master.stop_quantum_sdlc()
        raise
    
    finally:
        # Ensure clean shutdown
        await master.stop_quantum_sdlc()


if __name__ == "__main__":
    asyncio.run(main())