"""
Quantum-Inspired Monitoring and Observability Framework

Advanced monitoring system with quantum-inspired anomaly detection,
predictive analytics, and autonomous alerting mechanisms.
"""

import asyncio
import logging
import time
import json
import statistics
from typing import Dict, List, Any, Optional, Callable, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from datetime import datetime, timezone
import threading
import math

from .exceptions import QuantumMonitoringError
from .core import QuantumState


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AnomalyType(Enum):
    """Types of anomalies detected"""
    SPIKE = "spike"
    DROP = "drop"
    TREND = "trend"
    OUTLIER = "outlier"
    PATTERN_BREAK = "pattern_break"


@dataclass
class MetricPoint:
    """Single metric measurement"""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Anomaly:
    """Detected anomaly"""
    metric_name: str
    anomaly_type: AnomalyType
    severity: AlertSeverity
    timestamp: float
    value: float
    expected_range: tuple
    confidence: float
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """System alert"""
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    timestamp: float
    resolved: bool = False
    resolution_time: Optional[float] = None
    source_metric: Optional[str] = None
    anomaly: Optional[Anomaly] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuantumMetricCollector:
    """
    Quantum-inspired metric collector with superposition-based sampling
    """
    
    def __init__(self, name: str, metric_type: MetricType, 
                 max_points: int = 1000,
                 quantum_sampling: bool = True,
                 logger: Optional[logging.Logger] = None):
        
        self.name = name
        self.metric_type = metric_type
        self.max_points = max_points
        self.quantum_sampling = quantum_sampling
        self.logger = logger or logging.getLogger(__name__)
        
        self.points: deque = deque(maxlen=max_points)
        self.quantum_state = QuantumState.SUPERPOSITION
        
        # Statistics
        self._sum = 0.0
        self._count = 0
        self._min = float('inf')
        self._max = float('-inf')
        
        self._lock = threading.RLock()
    
    def record(self, value: float, labels: Dict[str, str] = None,
              metadata: Dict[str, Any] = None):
        """Record a metric point with quantum sampling"""
        
        with self._lock:
            timestamp = time.time()
            
            # Quantum sampling: use probabilistic recording
            if self.quantum_sampling and len(self.points) > self.max_points * 0.8:
                # Use quantum-inspired probability based on value significance
                sampling_prob = self._calculate_quantum_sampling_probability(value)
                if abs(hash(str(value)) % 100) / 100.0 > sampling_prob:
                    return  # Skip this sample
            
            point = MetricPoint(
                timestamp=timestamp,
                value=value,
                labels=labels or {},
                metadata=metadata or {}
            )
            
            self.points.append(point)
            
            # Update statistics
            self._sum += value
            self._count += 1
            self._min = min(self._min, value)
            self._max = max(self._max, value)
    
    def _calculate_quantum_sampling_probability(self, value: float) -> float:
        """Calculate quantum sampling probability based on value significance"""
        
        if not self.points:
            return 1.0
        
        recent_points = list(self.points)[-100:] if len(self.points) > 100 else list(self.points)
        recent_values = [p.value for p in recent_points]
        
        if len(recent_values) < 2:
            return 1.0
        
        mean_val = statistics.mean(recent_values)
        std_val = statistics.stdev(recent_values) if len(recent_values) > 1 else 1.0
        
        # Higher probability for outliers and significant changes
        z_score = abs(value - mean_val) / (std_val + 1e-8)
        
        # Quantum probability: higher for more unusual values
        quantum_prob = 0.3 + 0.7 * (1 - math.exp(-z_score / 2))
        
        return min(quantum_prob, 1.0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get metric statistics"""
        
        with self._lock:
            if self._count == 0:
                return {
                    "count": 0,
                    "sum": 0.0,
                    "mean": 0.0,
                    "min": 0.0,
                    "max": 0.0
                }
            
            return {
                "count": self._count,
                "sum": self._sum,
                "mean": self._sum / self._count,
                "min": self._min,
                "max": self._max
            }
    
    def get_recent_points(self, limit: int = 100) -> List[MetricPoint]:
        """Get recent metric points"""
        
        with self._lock:
            recent = list(self.points)[-limit:] if limit else list(self.points)
            return recent.copy()


class QuantumAnomalyDetector:
    """
    Quantum-inspired anomaly detection using superposition of detection algorithms
    """
    
    def __init__(self, sensitivity: float = 0.8,
                 window_size: int = 50,
                 quantum_ensemble: bool = True,
                 logger: Optional[logging.Logger] = None):
        
        self.sensitivity = sensitivity
        self.window_size = window_size
        self.quantum_ensemble = quantum_ensemble
        self.logger = logger or logging.getLogger(__name__)
        
        # Detection algorithms in superposition
        self.detection_algorithms = [
            self._z_score_detection,
            self._iqr_detection,
            self._moving_average_detection,
            self._quantum_pattern_detection
        ]
        
        # Algorithm weights (quantum superposition coefficients)
        self.algorithm_weights = [0.3, 0.25, 0.25, 0.2]
        
        # Historical context
        self.pattern_history = defaultdict(list)
        self.anomaly_history = []
        
    async def detect_anomalies(self, metric_name: str, 
                             points: List[MetricPoint]) -> List[Anomaly]:
        """Detect anomalies using quantum ensemble of algorithms"""
        
        if len(points) < self.window_size:
            return []
        
        values = [p.value for p in points]
        timestamps = [p.timestamp for p in points]
        
        anomalies = []
        
        if self.quantum_ensemble:
            # Run detection algorithms in superposition (parallel)
            detection_tasks = []
            for i, algorithm in enumerate(self.detection_algorithms):
                weight = self.algorithm_weights[i]
                task = asyncio.create_task(
                    self._run_detection_algorithm(
                        algorithm, metric_name, values, timestamps, weight
                    )
                )
                detection_tasks.append(task)
            
            # Gather results from all algorithms
            algorithm_results = await asyncio.gather(*detection_tasks, return_exceptions=True)
            
            # Quantum interference: combine results
            anomalies = self._combine_detection_results(
                algorithm_results, metric_name, values, timestamps
            )
        else:
            # Sequential detection (classical approach)
            for i, algorithm in enumerate(self.detection_algorithms):
                weight = self.algorithm_weights[i]
                algo_anomalies = await self._run_detection_algorithm(
                    algorithm, metric_name, values, timestamps, weight
                )
                anomalies.extend(algo_anomalies)
        
        # Filter and rank anomalies
        anomalies = self._filter_and_rank_anomalies(anomalies)
        
        # Update history
        self.anomaly_history.extend(anomalies)
        if len(self.anomaly_history) > 1000:
            self.anomaly_history = self.anomaly_history[-1000:]
        
        return anomalies
    
    async def _run_detection_algorithm(self, algorithm: Callable,
                                     metric_name: str,
                                     values: List[float],
                                     timestamps: List[float],
                                     weight: float) -> List[Anomaly]:
        """Run a single detection algorithm"""
        
        try:
            # Run algorithm in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, algorithm, metric_name, values, timestamps, weight
            )
            return result
            
        except Exception as e:
            self.logger.warning(
                f"Detection algorithm failed for {metric_name}: {str(e)}"
            )
            return []
    
    def _z_score_detection(self, metric_name: str, values: List[float],
                          timestamps: List[float], weight: float) -> List[Anomaly]:
        """Z-score based anomaly detection"""
        
        if len(values) < 10:
            return []
        
        anomalies = []
        
        # Calculate rolling statistics
        for i in range(len(values) - self.window_size + 1):
            window = values[i:i + self.window_size]
            current_value = values[i + self.window_size - 1]
            current_timestamp = timestamps[i + self.window_size - 1]
            
            mean_val = statistics.mean(window[:-1])  # Exclude current value
            std_val = statistics.stdev(window[:-1]) if len(window) > 2 else 1.0
            
            if std_val > 0:
                z_score = abs(current_value - mean_val) / std_val
                threshold = 2.0 / self.sensitivity  # Adaptive threshold
                
                if z_score > threshold:
                    anomaly_type = AnomalyType.SPIKE if current_value > mean_val else AnomalyType.DROP
                    severity = self._calculate_severity(z_score, threshold)
                    
                    anomaly = Anomaly(
                        metric_name=metric_name,
                        anomaly_type=anomaly_type,
                        severity=severity,
                        timestamp=current_timestamp,
                        value=current_value,
                        expected_range=(mean_val - 2*std_val, mean_val + 2*std_val),
                        confidence=weight * min(z_score / threshold, 1.0),
                        context={"z_score": z_score, "algorithm": "z_score"}
                    )
                    
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _iqr_detection(self, metric_name: str, values: List[float],
                      timestamps: List[float], weight: float) -> List[Anomaly]:
        """Interquartile range based anomaly detection"""
        
        if len(values) < 20:
            return []
        
        anomalies = []
        
        for i in range(len(values) - self.window_size + 1):
            window = values[i:i + self.window_size]
            current_value = values[i + self.window_size - 1]
            current_timestamp = timestamps[i + self.window_size - 1]
            
            # Calculate IQR
            sorted_window = sorted(window[:-1])
            q1 = sorted_window[len(sorted_window) // 4]
            q3 = sorted_window[3 * len(sorted_window) // 4]
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            if current_value < lower_bound or current_value > upper_bound:
                anomaly_type = AnomalyType.SPIKE if current_value > upper_bound else AnomalyType.DROP
                
                # Calculate severity based on distance from bounds
                if current_value > upper_bound:
                    distance = (current_value - upper_bound) / (iqr + 1e-8)
                else:
                    distance = (lower_bound - current_value) / (iqr + 1e-8)
                
                severity = self._calculate_severity(distance, 1.0)
                
                anomaly = Anomaly(
                    metric_name=metric_name,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    timestamp=current_timestamp,
                    value=current_value,
                    expected_range=(lower_bound, upper_bound),
                    confidence=weight * min(distance, 1.0),
                    context={"iqr": iqr, "algorithm": "iqr"}
                )
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def _moving_average_detection(self, metric_name: str, values: List[float],
                                timestamps: List[float], weight: float) -> List[Anomaly]:
        """Moving average based trend anomaly detection"""
        
        if len(values) < 20:
            return []
        
        anomalies = []
        window_size = min(self.window_size, 20)
        
        for i in range(window_size, len(values)):
            recent_window = values[i-window_size:i]
            older_window = values[max(0, i-2*window_size):i-window_size]
            
            if not older_window:
                continue
            
            recent_avg = statistics.mean(recent_window)
            older_avg = statistics.mean(older_window)
            
            # Detect significant trend changes
            if older_avg != 0:
                trend_change = abs(recent_avg - older_avg) / abs(older_avg)
                threshold = 0.3 / self.sensitivity
                
                if trend_change > threshold:
                    anomaly_type = AnomalyType.TREND
                    severity = self._calculate_severity(trend_change, threshold)
                    
                    anomaly = Anomaly(
                        metric_name=metric_name,
                        anomaly_type=anomaly_type,
                        severity=severity,
                        timestamp=timestamps[i],
                        value=values[i],
                        expected_range=(older_avg * 0.8, older_avg * 1.2),
                        confidence=weight * min(trend_change / threshold, 1.0),
                        context={
                            "trend_change": trend_change,
                            "recent_avg": recent_avg,
                            "older_avg": older_avg,
                            "algorithm": "moving_average"
                        }
                    )
                    
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _quantum_pattern_detection(self, metric_name: str, values: List[float],
                                 timestamps: List[float], weight: float) -> List[Anomaly]:
        """Quantum-inspired pattern break detection"""
        
        if len(values) < 30:
            return []
        
        anomalies = []
        
        # Look for pattern breaks in quantum superposition
        patterns = self._extract_patterns(values)
        
        for pattern_name, pattern_data in patterns.items():
            if pattern_data['confidence'] > 0.7:
                # Check if recent data breaks the pattern
                recent_values = values[-10:]
                pattern_fit = self._calculate_pattern_fit(recent_values, pattern_data)
                
                if pattern_fit < 0.5:  # Pattern break threshold
                    anomaly = Anomaly(
                        metric_name=metric_name,
                        anomaly_type=AnomalyType.PATTERN_BREAK,
                        severity=AlertSeverity.MEDIUM,
                        timestamp=timestamps[-1],
                        value=values[-1],
                        expected_range=pattern_data.get('expected_range', (0, 0)),
                        confidence=weight * (1.0 - pattern_fit),
                        context={
                            "pattern_name": pattern_name,
                            "pattern_fit": pattern_fit,
                            "algorithm": "quantum_pattern"
                        }
                    )
                    
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _extract_patterns(self, values: List[float]) -> Dict[str, Dict[str, Any]]:
        """Extract patterns using quantum-inspired analysis"""
        
        patterns = {}
        
        # Periodic pattern detection
        if len(values) >= 50:
            autocorr = self._autocorrelation(values)
            max_corr_idx = max(range(1, len(autocorr)), key=lambda i: autocorr[i])
            
            if autocorr[max_corr_idx] > 0.6:
                patterns['periodic'] = {
                    'period': max_corr_idx,
                    'correlation': autocorr[max_corr_idx],
                    'confidence': autocorr[max_corr_idx],
                    'expected_range': (min(values) * 0.9, max(values) * 1.1)
                }
        
        # Linear trend pattern
        if len(values) >= 20:
            x = list(range(len(values)))
            slope, intercept = self._linear_regression(x, values)
            correlation = self._correlation_coefficient(x, values)
            
            if abs(correlation) > 0.7:
                patterns['linear_trend'] = {
                    'slope': slope,
                    'intercept': intercept,
                    'correlation': correlation,
                    'confidence': abs(correlation),
                    'expected_range': (min(values) * 0.9, max(values) * 1.1)
                }
        
        return patterns
    
    def _autocorrelation(self, values: List[float]) -> List[float]:
        """Calculate autocorrelation for pattern detection"""
        
        n = len(values)
        mean_val = statistics.mean(values)
        var_val = statistics.variance(values)
        
        autocorr = []
        for lag in range(min(n // 4, 20)):  # Limit computation
            if lag == 0:
                autocorr.append(1.0)
            else:
                correlation = sum(
                    (values[i] - mean_val) * (values[i - lag] - mean_val)
                    for i in range(lag, n)
                ) / ((n - lag) * var_val)
                
                autocorr.append(correlation)
        
        return autocorr
    
    def _linear_regression(self, x: List[float], y: List[float]) -> tuple:
        """Simple linear regression"""
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi ** 2 for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        intercept = (sum_y - slope * sum_x) / n
        
        return slope, intercept
    
    def _correlation_coefficient(self, x: List[float], y: List[float]) -> float:
        """Calculate correlation coefficient"""
        
        n = len(x)
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        
        sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
        sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)
        
        denominator = math.sqrt(sum_sq_x * sum_sq_y)
        
        return numerator / denominator if denominator > 0 else 0
    
    def _calculate_pattern_fit(self, values: List[float], 
                             pattern_data: Dict[str, Any]) -> float:
        """Calculate how well values fit a pattern"""
        
        # Simplified pattern fitting
        if 'period' in pattern_data:
            # For periodic patterns, check if recent values follow the period
            return 0.8  # Simplified
        
        elif 'slope' in pattern_data:
            # For linear trends, check if recent values follow the trend
            return 0.7  # Simplified
        
        return 0.5
    
    def _combine_detection_results(self, algorithm_results: List[List[Anomaly]],
                                 metric_name: str, values: List[float],
                                 timestamps: List[float]) -> List[Anomaly]:
        """Combine anomaly detection results using quantum interference"""
        
        all_anomalies = []
        
        for result in algorithm_results:
            if isinstance(result, list):
                all_anomalies.extend(result)
        
        # Quantum interference: merge similar anomalies
        merged_anomalies = []
        time_tolerance = 10.0  # seconds
        
        for anomaly in all_anomalies:
            # Check if we should merge with existing anomaly
            merged = False
            for existing in merged_anomalies:
                if (abs(anomaly.timestamp - existing.timestamp) < time_tolerance and
                    anomaly.anomaly_type == existing.anomaly_type):
                    # Quantum superposition: combine confidences
                    existing.confidence = math.sqrt(
                        existing.confidence ** 2 + anomaly.confidence ** 2
                    )
                    existing.confidence = min(existing.confidence, 1.0)
                    
                    # Update severity if higher
                    if anomaly.severity.value == "critical" or \
                       (anomaly.severity.value == "high" and existing.severity.value != "critical"):
                        existing.severity = anomaly.severity
                    
                    merged = True
                    break
            
            if not merged:
                merged_anomalies.append(anomaly)
        
        return merged_anomalies
    
    def _filter_and_rank_anomalies(self, anomalies: List[Anomaly]) -> List[Anomaly]:
        """Filter and rank anomalies by severity and confidence"""
        
        # Filter by minimum confidence
        min_confidence = 0.3
        filtered = [a for a in anomalies if a.confidence >= min_confidence]
        
        # Sort by severity and confidence
        severity_order = {
            AlertSeverity.CRITICAL: 4,
            AlertSeverity.HIGH: 3,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.LOW: 1
        }
        
        filtered.sort(
            key=lambda a: (severity_order[a.severity], a.confidence),
            reverse=True
        )
        
        return filtered
    
    def _calculate_severity(self, score: float, threshold: float) -> AlertSeverity:
        """Calculate anomaly severity based on detection score"""
        
        ratio = score / threshold
        
        if ratio > 3.0:
            return AlertSeverity.CRITICAL
        elif ratio > 2.0:
            return AlertSeverity.HIGH
        elif ratio > 1.5:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW


class QuantumMonitoringSystem:
    """
    Comprehensive quantum monitoring system
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        self.metrics: Dict[str, QuantumMetricCollector] = {}
        self.anomaly_detector = QuantumAnomalyDetector(logger=self.logger)
        
        self.alerts: List[Alert] = []
        self.alert_handlers: List[Callable] = []
        
        # Monitoring configuration
        self.monitoring_interval = 30.0
        self.monitoring_task: Optional[asyncio.Task] = None
        self.auto_anomaly_detection = True
        
        self._lock = asyncio.Lock()
    
    def create_metric(self, name: str, metric_type: MetricType, **kwargs) -> QuantumMetricCollector:
        """Create a new metric collector"""
        
        metric = QuantumMetricCollector(
            name=name, 
            metric_type=metric_type,
            logger=self.logger,
            **kwargs
        )
        
        self.metrics[name] = metric
        self.logger.info(f"Created metric: {name} ({metric_type.value})")
        
        return metric
    
    def record_metric(self, name: str, value: float, 
                     labels: Dict[str, str] = None,
                     metadata: Dict[str, Any] = None):
        """Record a metric value"""
        
        if name not in self.metrics:
            # Auto-create metric as gauge
            self.create_metric(name, MetricType.GAUGE)
        
        self.metrics[name].record(value, labels, metadata)
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler"""
        self.alert_handlers.append(handler)
        self.logger.info(f"Added alert handler: {handler.__name__}")
    
    async def start_monitoring(self):
        """Start continuous monitoring"""
        
        if self.monitoring_task is None or self.monitoring_task.done():
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("Started quantum monitoring system")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            self.logger.info("Stopped quantum monitoring system")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        
        while True:
            try:
                if self.auto_anomaly_detection:
                    await self._check_anomalies()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {str(e)}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _check_anomalies(self):
        """Check all metrics for anomalies"""
        
        for metric_name, metric in self.metrics.items():
            try:
                points = metric.get_recent_points(100)
                
                if len(points) >= 20:  # Minimum points for detection
                    anomalies = await self.anomaly_detector.detect_anomalies(
                        metric_name, points
                    )
                    
                    for anomaly in anomalies:
                        await self._handle_anomaly(anomaly)
                        
            except Exception as e:
                self.logger.error(f"Anomaly detection failed for {metric_name}: {str(e)}")
    
    async def _handle_anomaly(self, anomaly: Anomaly):
        """Handle detected anomaly"""
        
        # Create alert
        alert_id = f"anomaly_{anomaly.metric_name}_{int(anomaly.timestamp)}"
        
        alert = Alert(
            alert_id=alert_id,
            title=f"Anomaly detected in {anomaly.metric_name}",
            description=f"{anomaly.anomaly_type.value.title()} detected with {anomaly.confidence:.1%} confidence",
            severity=anomaly.severity,
            timestamp=anomaly.timestamp,
            source_metric=anomaly.metric_name,
            anomaly=anomaly,
            metadata={
                "anomaly_context": anomaly.context
            }
        )
        
        async with self._lock:
            self.alerts.append(alert)
            
            # Keep only recent alerts
            if len(self.alerts) > 1000:
                self.alerts = self.alerts[-1000:]
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {str(e)}")
        
        self.logger.warning(
            f"Anomaly Alert [{alert.severity.value.upper()}]: "
            f"{alert.title} - {alert.description}"
        )
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status"""
        
        metric_stats = {}
        for name, metric in self.metrics.items():
            metric_stats[name] = {
                "type": metric.metric_type.value,
                "points_count": len(metric.points),
                "statistics": metric.get_statistics(),
                "quantum_state": metric.quantum_state.value
            }
        
        recent_alerts = [
            {
                "alert_id": alert.alert_id,
                "title": alert.title,
                "severity": alert.severity.value,
                "timestamp": alert.timestamp,
                "resolved": alert.resolved,
                "source_metric": alert.source_metric
            }
            for alert in self.alerts[-20:]  # Last 20 alerts
        ]
        
        return {
            "monitoring_active": (
                self.monitoring_task is not None and 
                not self.monitoring_task.done()
            ),
            "auto_anomaly_detection": self.auto_anomaly_detection,
            "monitoring_interval": self.monitoring_interval,
            "metrics": metric_stats,
            "recent_alerts": recent_alerts,
            "alert_handlers_count": len(self.alert_handlers),
            "total_alerts": len(self.alerts)
        }
    
    def export_monitoring_report(self) -> Dict[str, Any]:
        """Export comprehensive monitoring report"""
        
        return {
            "quantum_monitoring_report": {
                "version": "1.0.0",
                "timestamp": time.time(),
                "system_status": self.get_monitoring_status(),
                "anomaly_detector_config": {
                    "sensitivity": self.anomaly_detector.sensitivity,
                    "window_size": self.anomaly_detector.window_size,
                    "quantum_ensemble": self.anomaly_detector.quantum_ensemble,
                    "algorithm_weights": self.anomaly_detector.algorithm_weights
                },
                "recent_anomalies": [
                    {
                        "metric_name": anomaly.metric_name,
                        "anomaly_type": anomaly.anomaly_type.value,
                        "severity": anomaly.severity.value,
                        "timestamp": anomaly.timestamp,
                        "confidence": anomaly.confidence,
                        "context": anomaly.context
                    }
                    for anomaly in self.anomaly_detector.anomaly_history[-50:]
                ]
            }
        }
    
    def __del__(self):
        """Cleanup resources"""
        self.stop_monitoring()


# Default alert handlers
async def log_alert_handler(alert: Alert):
    """Default alert handler that logs alerts"""
    logger = logging.getLogger("QuantumMonitoring")
    logger.warning(
        f"ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.description}"
    )


def console_alert_handler(alert: Alert):
    """Console alert handler"""
    print(f"🚨 ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.description}")


# Global monitoring instance
quantum_monitoring = QuantumMonitoringSystem()
quantum_monitoring.add_alert_handler(log_alert_handler)