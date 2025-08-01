# Production Monitoring and Observability Enhancement

**Status**: ⚠️ Partial Implementation - Needs Integration  
**Priority**: P0 - Critical for production operations  
**Current State**: Prometheus/Grafana configured but not fully integrated  

## Current Monitoring Infrastructure

### Existing Components
- ✅ **Prometheus server**: Configured in `docker-compose.yml`
- ✅ **Grafana dashboards**: Basic monitoring setup
- ✅ **Security audit logging**: Comprehensive event tracking in `security_audit.py`
- ✅ **Application logging**: Structured logging throughout application
- ⚠️ **Missing**: Application metrics integration with Prometheus
- ⚠️ **Missing**: Automated alerting and incident response

## Enhanced Observability Stack

### 1. Application Metrics Integration

**File**: `sql_synthesizer/monitoring/metrics.py`
```python
"""
Application metrics collection for Prometheus integration.
Integrates with existing monitoring infrastructure.
"""

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
from prometheus_client.exposition import generate_latest
from functools import wraps
import time
import psutil
import asyncio
from typing import Callable, Any

# Metrics registry
METRICS_REGISTRY = CollectorRegistry()

# Application performance metrics
QUERY_DURATION = Histogram(
    'sql_synthesizer_query_duration_seconds',
    'Time spent processing SQL queries',
    ['query_type', 'database_type', 'success'],
    registry=METRICS_REGISTRY
)

QUERY_TOTAL = Counter(
    'sql_synthesizer_queries_total',
    'Total number of SQL queries processed',
    ['query_type', 'database_type', 'success'],
    registry=METRICS_REGISTRY
)

ACTIVE_CONNECTIONS = Gauge(
    'sql_synthesizer_active_connections',
    'Number of active database connections',
    ['database_type'],
    registry=METRICS_REGISTRY
)

CACHE_OPERATIONS = Counter(
    'sql_synthesizer_cache_operations_total',
    'Total cache operations',
    ['operation', 'cache_type', 'hit_miss'],
    registry=METRICS_REGISTRY
)

SECURITY_EVENTS = Counter(
    'sql_synthesizer_security_events_total',
    'Security events detected',
    ['event_type', 'severity'],
    registry=METRICS_REGISTRY
)

# System resource metrics
MEMORY_USAGE = Gauge(
    'sql_synthesizer_memory_usage_bytes',
    'Memory usage in bytes',
    registry=METRICS_REGISTRY
)

CPU_USAGE = Gauge(
    'sql_synthesizer_cpu_usage_percent',
    'CPU usage percentage',
    registry=METRICS_REGISTRY
)

def track_query_metrics(query_type: str, database_type: str):
    """Decorator to track query performance metrics."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            success = "success"
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception:
                success = "error"
                raise
            finally:
                duration = time.time() - start_time
                QUERY_DURATION.labels(
                    query_type=query_type,
                    database_type=database_type,
                    success=success
                ).observe(duration)
                QUERY_TOTAL.labels(
                    query_type=query_type,
                    database_type=database_type,
                    success=success
                ).inc()
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            success = "success"
            try:
                result = func(*args, **kwargs)
                return result
            except Exception:
                success = "error"
                raise
            finally:
                duration = time.time() - start_time
                QUERY_DURATION.labels(
                    query_type=query_type,
                    database_type=database_type,
                    success=success
                ).observe(duration)
                QUERY_TOTAL.labels(
                    query_type=query_type,
                    database_type=database_type,
                    success=success
                ).inc()
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def track_cache_metrics(operation: str, cache_type: str):
    """Decorator to track cache operation metrics."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            hit_miss = "hit" if result is not None else "miss"
            CACHE_OPERATIONS.labels(
                operation=operation,
                cache_type=cache_type,
                hit_miss=hit_miss
            ).inc()
            return result
        return wrapper
    return decorator

def track_security_event(event_type: str, severity: str):
    """Track security events in metrics."""
    SECURITY_EVENTS.labels(
        event_type=event_type,
        severity=severity
    ).inc()

async def update_system_metrics():
    """Update system resource metrics."""
    while True:
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            MEMORY_USAGE.set(memory.used)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            CPU_USAGE.set(cpu_percent)
            
            await asyncio.sleep(30)  # Update every 30 seconds
        except Exception as e:
            # Log error but don't crash the monitoring
            import logging
            logging.error(f"Failed to update system metrics: {e}")
            await asyncio.sleep(60)

def get_metrics() -> str:
    """Get Prometheus metrics in text format."""
    return generate_latest(METRICS_REGISTRY).decode('utf-8')
```

### 2. Enhanced Flask Integration

**Integration with**: `sql_synthesizer/webapp.py`
```python
# Add to existing webapp.py imports
from .monitoring.metrics import (
    get_metrics, track_query_metrics, track_cache_metrics, 
    track_security_event, update_system_metrics
)

# Add metrics endpoint
@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint."""
    return Response(get_metrics(), mimetype='text/plain')

# Add system metrics background task
@app.before_first_request
def start_monitoring():
    """Start background monitoring tasks."""
    import asyncio
    import threading
    
    def run_monitoring():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(update_system_metrics())
    
    monitoring_thread = threading.Thread(target=run_monitoring, daemon=True)
    monitoring_thread.start()
```

### 3. Database Connection Pool Monitoring

**Integration with**: `sql_synthesizer/database.py`
```python
# Add to existing database.py
from .monitoring.metrics import ACTIVE_CONNECTIONS

class DatabaseManager:
    def __init__(self, config):
        # Existing initialization
        self._update_connection_metrics()
    
    def _update_connection_metrics(self):
        """Update connection pool metrics."""
        for db_type, pool in self.connection_pools.items():
            if hasattr(pool, 'size'):
                ACTIVE_CONNECTIONS.labels(database_type=db_type).set(pool.size())
```

### 4. Security Event Integration

**Integration with**: `sql_synthesizer/security_audit.py`
```python
# Add to existing security_audit.py
from .monitoring.metrics import track_security_event

class SecurityAuditLogger:
    def log_event(self, event_type: str, details: dict, severity: str = "info"):
        # Existing logging code
        
        # Add metrics tracking
        track_security_event(event_type, severity)
        
        # Rest of existing method
```

## Grafana Dashboard Configuration

### Application Performance Dashboard

**File**: `monitoring/grafana/dashboards/application-performance.json`
```json
{
  "dashboard": {
    "title": "SQL Synthesizer - Application Performance",
    "panels": [
      {
        "title": "Query Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(sql_synthesizer_query_duration_seconds_sum[5m]) / rate(sql_synthesizer_query_duration_seconds_count[5m])",
            "legendFormat": "Avg Response Time - {{query_type}}"
          }
        ]
      },
      {
        "title": "Query Throughput",
        "type": "graph", 
        "targets": [
          {
            "expr": "rate(sql_synthesizer_queries_total[5m])",
            "legendFormat": "Queries/sec - {{query_type}}"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(sql_synthesizer_queries_total{success=\"error\"}[5m]) / rate(sql_synthesizer_queries_total[5m]) * 100",
            "legendFormat": "Error Rate %"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(sql_synthesizer_cache_operations_total{hit_miss=\"hit\"}[5m]) / rate(sql_synthesizer_cache_operations_total[5m]) * 100",
            "legendFormat": "Cache Hit Rate %"
          }
        ]
      }
    ]
  }
}
```

### Security Monitoring Dashboard

**File**: `monitoring/grafana/dashboards/security-monitoring.json`
```json
{
  "dashboard": {
    "title": "SQL Synthesizer - Security Monitoring",
    "panels": [
      {
        "title": "Security Events",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(sql_synthesizer_security_events_total[5m])",
            "legendFormat": "{{event_type}} - {{severity}}"
          }
        ]
      },
      {
        "title": "Failed Authentication Attempts",
        "type": "singlestat",
        "targets": [
          {
            "expr": "increase(sql_synthesizer_security_events_total{event_type=\"authentication_failure\"}[1h])",
            "legendFormat": "Failed Logins/hour"
          }
        ]
      },
      {
        "title": "SQL Injection Attempts",
        "type": "singlestat",
        "targets": [
          {
            "expr": "increase(sql_synthesizer_security_events_total{event_type=\"sql_injection_attempt\"}[1h])",
            "legendFormat": "Injection Attempts/hour"
          }
        ]
      }
    ]
  }
}
```

## Alerting Configuration

### Prometheus Alert Rules

**File**: `monitoring/prometheus/alerts.yml`
```yaml
groups:
- name: sql_synthesizer_alerts
  rules:
  # Performance Alerts
  - alert: HighQueryLatency
    expr: rate(sql_synthesizer_query_duration_seconds_sum[5m]) / rate(sql_synthesizer_query_duration_seconds_count[5m]) > 2
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High query latency detected"
      description: "Average query response time is {{ $value }}s"
  
  - alert: HighErrorRate
    expr: rate(sql_synthesizer_queries_total{success="error"}[5m]) / rate(sql_synthesizer_queries_total[5m]) > 0.05
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value | humanizePercentage }}"
  
  # Security Alerts
  - alert: SecurityEventSpike
    expr: rate(sql_synthesizer_security_events_total[5m]) > 10
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Security event spike detected"
      description: "{{ $value }} security events per second"
  
  - alert: SQLInjectionAttempts
    expr: increase(sql_synthesizer_security_events_total{event_type="sql_injection_attempt"}[5m]) > 0
    for: 0m
    labels:
      severity: critical
    annotations:
      summary: "SQL injection attempts detected"
      description: "{{ $value }} SQL injection attempts in the last 5 minutes"
  
  # Resource Alerts
  - alert: HighMemoryUsage
    expr: sql_synthesizer_memory_usage_bytes / (1024^3) > 4
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage"
      description: "Memory usage is {{ $value | humanize }}GB"
  
  - alert: HighCPUUsage
    expr: sql_synthesizer_cpu_usage_percent > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage"
      description: "CPU usage is {{ $value }}%"
```

### Alertmanager Configuration

**File**: `monitoring/alertmanager/config.yml`
```yaml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@company.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
  - match:
      severity: critical
    receiver: 'critical-alerts'
  - match:
      severity: warning
    receiver: 'warning-alerts'

receivers:
- name: 'web.hook'
  webhook_configs:
  - url: 'http://127.0.0.1:5001/'

- name: 'critical-alerts'
  email_configs:
  - to: 'oncall@company.com'
    subject: 'CRITICAL: SQL Synthesizer Alert'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}
  
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK_URL'
    channel: '#alerts'
    title: 'CRITICAL: SQL Synthesizer Alert'
    text: |
      {{ range .Alerts }}
      {{ .Annotations.summary }}
      {{ .Annotations.description }}
      {{ end }}

- name: 'warning-alerts'
  email_configs:
  - to: 'team@company.com'
    subject: 'WARNING: SQL Synthesizer Alert'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}
```

## Enhanced Docker Compose Integration

**Update**: `docker-compose.yml` monitoring section
```yaml
# Add to existing docker-compose.yml
services:
  app:
    # Existing app configuration
    ports:
      - "5000:5000"
      - "8080:8080"  # Add metrics port
    environment:
      - PROMETHEUS_METRICS_ENABLED=true
      - METRICS_PORT=8080

  prometheus:
    # Existing prometheus configuration
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/prometheus/alerts.yml:/etc/prometheus/alerts.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'

  alertmanager:
    image: prom/alertmanager:latest
    container_name: alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager/config.yml:/etc/alertmanager/config.yml
    command:
      - '--config.file=/etc/alertmanager/config.yml'
      - '--storage.path=/alertmanager'
      - '--web.external-url=http://localhost:9093'
```

## Health Check Endpoints

**Integration with**: `sql_synthesizer/webapp.py`
```python
@app.route('/health')
def health_check():
    """Comprehensive health check endpoint."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {}
    }
    
    # Database connectivity check
    try:
        # Test database connections
        health_status["checks"]["database"] = "healthy"
    except Exception as e:
        health_status["checks"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Cache connectivity check
    try:
        # Test Redis/Memcached connections
        health_status["checks"]["cache"] = "healthy"
    except Exception as e:
        health_status["checks"]["cache"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Security audit system check
    try:
        # Test security audit logging
        health_status["checks"]["security_audit"] = "healthy"
    except Exception as e:
        health_status["checks"]["security_audit"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    return jsonify(health_status), status_code

@app.route('/health/ready')
def readiness_check():
    """Kubernetes readiness probe endpoint."""
    # Quick checks for service readiness
    return jsonify({"status": "ready"}), 200

@app.route('/health/live')
def liveness_check():
    """Kubernetes liveness probe endpoint."""
    # Basic application liveness check
    return jsonify({"status": "alive"}), 200
```

## Implementation Roadmap

### Phase 1: Core Metrics (Week 1)
- [ ] Implement application metrics collection
- [ ] Add Prometheus metrics endpoint
- [ ] Update existing monitoring infrastructure

### Phase 2: Dashboards (Week 2)
- [ ] Create Grafana application performance dashboard
- [ ] Create security monitoring dashboard
- [ ] Integrate with existing Grafana setup

### Phase 3: Alerting (Week 3)
- [ ] Configure Prometheus alert rules
- [ ] Setup Alertmanager with email/Slack notifications
- [ ] Test alert escalation procedures

### Phase 4: Advanced Monitoring (Week 4)
- [ ] Implement distributed tracing for async operations
- [ ] Add business metrics tracking
- [ ] Performance regression detection automation

## Success Metrics

- **Observability Coverage**: 95%+ of critical paths monitored
- **Alert Response Time**: <2 minutes for critical alerts
- **False Positive Rate**: <5% for security alerts
- **Performance Visibility**: Real-time query performance tracking
- **Security Monitoring**: 100% security event coverage

## Integration Benefits

- **Proactive Issue Detection**: Automated alerting before user impact
- **Performance Optimization**: Data-driven performance improvements
- **Security Posture**: Real-time security event monitoring
- **Operational Efficiency**: Reduced MTTR through better observability
- **Compliance**: Audit trail and monitoring for regulatory requirements