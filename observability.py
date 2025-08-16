"""
OpenTelemetry configuration for SQL Query Synthesizer.

This module provides comprehensive observability setup including:
- Distributed tracing with OTLP export
- Custom metrics for query performance and cache efficiency  
- Structured logging with trace correlation
- Health check instrumentation
"""

import logging
import os
from typing import Optional

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)


def setup_observability(
    service_name: str = "sql-query-synthesizer",
    service_version: str = "1.0.0",
    otlp_endpoint: Optional[str] = None,
    enable_console_export: bool = False
) -> None:
    """
    Configure OpenTelemetry tracing, metrics, and instrumentation.
    
    Args:
        service_name: Name of the service for telemetry
        service_version: Version of the service
        otlp_endpoint: OTLP collector endpoint (if None, uses environment variable)
        enable_console_export: Whether to export traces to console for debugging
    """

    # Configure resource attributes
    resource = Resource.create({
        "service.name": service_name,
        "service.version": service_version,
        "service.instance.id": os.environ.get("HOSTNAME", "localhost"),
        "deployment.environment": os.environ.get("QUERY_AGENT_ENV", "development")
    })

    # Setup tracing
    _setup_tracing(resource, otlp_endpoint, enable_console_export)

    # Setup metrics
    _setup_metrics(resource, otlp_endpoint)

    # Setup auto-instrumentation
    _setup_instrumentation()

    logger.info(f"OpenTelemetry configured for {service_name} v{service_version}")


def _setup_tracing(
    resource: Resource,
    otlp_endpoint: Optional[str],
    enable_console_export: bool
) -> None:
    """Configure distributed tracing with OTLP export."""

    # Create tracer provider
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)

    # Configure OTLP span exporter
    otlp_endpoint = otlp_endpoint or os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "http://localhost:4317"
    )

    if otlp_endpoint:
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        span_processor = BatchSpanProcessor(otlp_exporter)
        tracer_provider.add_span_processor(span_processor)
        logger.info(f"OTLP span exporter configured for {otlp_endpoint}")

    # Optional console exporter for debugging
    if enable_console_export:
        from opentelemetry.exporter.console import ConsoleSpanExporter
        console_exporter = ConsoleSpanExporter()
        console_processor = BatchSpanProcessor(console_exporter)
        tracer_provider.add_span_processor(console_processor)
        logger.info("Console span exporter enabled")


def _setup_metrics(resource: Resource, otlp_endpoint: Optional[str]) -> None:
    """Configure metrics collection and export."""

    otlp_endpoint = otlp_endpoint or os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "http://localhost:4317"
    )

    if otlp_endpoint:
        # Create OTLP metric exporter
        metric_exporter = OTLPMetricExporter(endpoint=otlp_endpoint)
        metric_reader = PeriodicExportingMetricReader(
            exporter=metric_exporter,
            export_interval_millis=30000  # Export every 30 seconds
        )

        # Create meter provider
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader]
        )
        metrics.set_meter_provider(meter_provider)
        logger.info(f"OTLP metric exporter configured for {otlp_endpoint}")


def _setup_instrumentation() -> None:
    """Configure automatic instrumentation for common libraries."""

    # Flask web framework instrumentation
    try:
        FlaskInstrumentor().instrument()
        logger.info("Flask instrumentation enabled")
    except Exception as e:
        logger.warning(f"Failed to instrument Flask: {e}")

    # SQLAlchemy database instrumentation
    try:
        SQLAlchemyInstrumentor().instrument()
        logger.info("SQLAlchemy instrumentation enabled")
    except Exception as e:
        logger.warning(f"Failed to instrument SQLAlchemy: {e}")

    # Redis cache instrumentation
    try:
        RedisInstrumentor().instrument()
        logger.info("Redis instrumentation enabled")
    except Exception as e:
        logger.warning(f"Failed to instrument Redis: {e}")

    # HTTP requests instrumentation
    try:
        RequestsInstrumentor().instrument()
        logger.info("Requests instrumentation enabled")
    except Exception as e:
        logger.warning(f"Failed to instrument Requests: {e}")


def get_tracer(name: str) -> trace.Tracer:
    """Get a tracer instance for creating custom spans."""
    return trace.get_tracer(name, "1.0.0")


def get_meter(name: str) -> metrics.Meter:
    """Get a meter instance for creating custom metrics."""
    return metrics.get_meter(name, "1.0.0")


# Common custom metrics for the SQL Query Synthesizer
def create_custom_metrics() -> dict:
    """Create application-specific metrics."""

    meter = get_meter("sql_synthesizer.metrics")

    custom_metrics = {
        # Query performance metrics
        "sql_generation_duration": meter.create_histogram(
            name="sql_generation_duration_seconds",
            description="Time taken to generate SQL from natural language",
            unit="s"
        ),
        "query_execution_duration": meter.create_histogram(
            name="query_execution_duration_seconds",
            description="Time taken to execute SQL queries",
            unit="s"
        ),

        # Cache metrics
        "cache_hit_ratio": meter.create_gauge(
            name="cache_hit_ratio",
            description="Ratio of cache hits to total cache requests"
        ),
        "cache_size": meter.create_gauge(
            name="cache_size_bytes",
            description="Current size of cache in bytes",
            unit="By"
        ),

        # Application metrics
        "active_connections": meter.create_gauge(
            name="database_connections_active",
            description="Number of active database connections"
        ),
        "query_requests_total": meter.create_counter(
            name="query_requests_total",
            description="Total number of query requests processed"
        ),
        "llm_api_calls_total": meter.create_counter(
            name="llm_api_calls_total",
            description="Total number of LLM API calls made"
        ),
        "errors_total": meter.create_counter(
            name="errors_total",
            description="Total number of errors by type"
        )
    }

    logger.info(f"Created {len(custom_metrics)} custom metrics")
    return custom_metrics


# Environment-based configuration
def configure_for_environment() -> None:
    """Configure observability based on current environment."""

    env = os.environ.get("QUERY_AGENT_ENV", "development")

    if env == "production":
        # Production: Full observability with OTLP export
        setup_observability(
            service_name="sql-query-synthesizer",
            service_version=get_service_version(),
            otlp_endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
            enable_console_export=False
        )
    elif env == "staging":
        # Staging: Full observability with console output
        setup_observability(
            service_name="sql-query-synthesizer-staging",
            service_version=get_service_version(),
            otlp_endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
            enable_console_export=True
        )
    elif env == "development":
        # Development: Console output only
        setup_observability(
            service_name="sql-query-synthesizer-dev",
            service_version=get_service_version(),
            enable_console_export=True
        )

    # Create custom metrics for all environments
    create_custom_metrics()


def get_service_version() -> str:
    """Get service version from package or environment."""
    try:
        from sql_synthesizer import __version__
        return __version__
    except ImportError:
        return os.environ.get("SERVICE_VERSION", "dev")


if __name__ == "__main__":
    # Demo/test the observability setup
    configure_for_environment()

    # Create a test trace
    tracer = get_tracer("test")
    with tracer.start_as_current_span("test_span") as span:
        span.set_attribute("test.key", "test_value")
        logger.info("Test span created successfully")
