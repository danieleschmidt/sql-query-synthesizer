#!/bin/bash
# Production entrypoint script for SQL Query Synthesizer
# Handles initialization, health checks, and graceful shutdown

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] [ENTRYPOINT]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR]${NC} $1" >&2
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] [WARN]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] [SUCCESS]${NC} $1"
}

# Cleanup function for graceful shutdown
cleanup() {
    log "Received shutdown signal, performing cleanup..."
    
    # Kill any background processes
    if [ ! -z "${GUNICORN_PID:-}" ]; then
        log "Stopping Gunicorn (PID: $GUNICORN_PID)..."
        kill -TERM "$GUNICORN_PID" 2>/dev/null || true
        wait "$GUNICORN_PID" 2>/dev/null || true
    fi
    
    # Stop auto-scaling engine if running
    if [ -f "/tmp/autoscaler.pid" ]; then
        log "Stopping auto-scaling engine..."
        kill -TERM "$(cat /tmp/autoscaler.pid)" 2>/dev/null || true
        rm -f "/tmp/autoscaler.pid"
    fi
    
    log_success "Cleanup completed"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT SIGQUIT

# Function to wait for service
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local timeout=${4:-30}
    
    log "Waiting for $service_name at $host:$port..."
    
    local count=0
    while ! nc -z "$host" "$port" >/dev/null 2>&1; do
        if [ $count -ge $timeout ]; then
            log_error "Timeout waiting for $service_name"
            return 1
        fi
        sleep 1
        count=$((count + 1))
    done
    
    log_success "$service_name is available"
    return 0
}

# Function to initialize database
init_database() {
    log "Initializing database..."
    
    # Wait for PostgreSQL
    if [ ! -z "${DATABASE_URL:-}" ]; then
        # Extract host and port from DATABASE_URL
        if [[ $DATABASE_URL =~ postgresql://[^@]+@([^:]+):([0-9]+)/ ]]; then
            local db_host="${BASH_REMATCH[1]}"
            local db_port="${BASH_REMATCH[2]}"
            wait_for_service "$db_host" "$db_port" "PostgreSQL" 60
        fi
    fi
    
    # Run database migrations if needed
    if [ -f "/app/sql_synthesizer/database/migrations.py" ]; then
        log "Running database migrations..."
        python -c "from sql_synthesizer.database.migrations import run_migrations; run_migrations()" || {
            log_error "Database migrations failed"
            return 1
        }
    fi
    
    log_success "Database initialization completed"
}

# Function to initialize cache
init_cache() {
    log "Initializing cache..."
    
    # Wait for Redis if configured
    if [ "${QUERY_AGENT_CACHE_BACKEND:-}" = "redis" ]; then
        local redis_host="${QUERY_AGENT_REDIS_HOST:-redis}"
        local redis_port="${QUERY_AGENT_REDIS_PORT:-6379}"
        wait_for_service "$redis_host" "$redis_port" "Redis" 30
    fi
    
    log_success "Cache initialization completed"
}

# Function to validate configuration
validate_config() {
    log "Validating configuration..."
    
    # Check required environment variables
    local required_vars=(
        "QUERY_AGENT_SECRET_KEY"
        "DATABASE_URL"
    )
    
    local missing_vars=()
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        log_error "Missing required environment variables: ${missing_vars[*]}"
        return 1
    fi
    
    # Validate OpenAI API key if required
    if [ ! -z "${OPENAI_API_KEY:-}" ]; then
        if [[ ! "$OPENAI_API_KEY" =~ ^sk-[a-zA-Z0-9]{48}$ ]]; then
            log_warn "OpenAI API key format appears invalid"
        fi
    fi
    
    # Validate database URL
    if [[ ! "$DATABASE_URL" =~ ^postgresql:// ]]; then
        log_error "DATABASE_URL must be a PostgreSQL URL"
        return 1
    fi
    
    log_success "Configuration validation completed"
}

# Function to setup monitoring
setup_monitoring() {
    log "Setting up monitoring..."
    
    # Create metrics directory
    mkdir -p /app/data/metrics
    
    # Start auto-scaling engine if enabled
    if [ "${QUERY_AGENT_AUTO_SCALING_ENABLED:-false}" = "true" ]; then
        log "Starting auto-scaling engine..."
        python -c "
from sql_synthesizer.auto_scaling_engine import auto_scaling_engine
auto_scaling_engine.start()
import time
import os
with open('/tmp/autoscaler.pid', 'w') as f:
    f.write(str(os.getpid()))
while True:
    time.sleep(60)
" &
        echo $! > /tmp/autoscaler.pid
    fi
    
    log_success "Monitoring setup completed"
}

# Function to run health check
run_health_check() {
    log "Running initial health check..."
    
    python /app/healthcheck.py || {
        log_error "Health check failed"
        return 1
    }
    
    log_success "Health check passed"
}

# Function to optimize performance
optimize_performance() {
    log "Applying performance optimizations..."
    
    # Set Python optimization flags
    export PYTHONOPTIMIZE=2
    
    # Configure memory settings
    if [ ! -z "${GUNICORN_WORKERS:-}" ]; then
        log "Configuring for $GUNICORN_WORKERS workers"
    fi
    
    # Set up connection pooling
    export QUERY_AGENT_DB_POOL_SIZE="${QUERY_AGENT_DB_POOL_SIZE:-20}"
    export QUERY_AGENT_DB_MAX_OVERFLOW="${QUERY_AGENT_DB_MAX_OVERFLOW:-40}"
    
    log_success "Performance optimizations applied"
}

# Main initialization function
initialize() {
    log "Starting SQL Query Synthesizer initialization..."
    
    # Create necessary directories
    mkdir -p /app/logs /app/data /app/tmp
    
    # Validate configuration
    validate_config
    
    # Initialize services
    init_database
    init_cache
    
    # Setup monitoring
    setup_monitoring
    
    # Apply optimizations
    optimize_performance
    
    # Run health check
    run_health_check
    
    log_success "Initialization completed successfully"
}

# Function to start the application
start_application() {
    log "Starting SQL Query Synthesizer application..."
    
    # Set default values if not provided
    export GUNICORN_WORKERS="${GUNICORN_WORKERS:-4}"
    export GUNICORN_THREADS="${GUNICORN_THREADS:-2}"
    export GUNICORN_TIMEOUT="${GUNICORN_TIMEOUT:-120}"
    export GUNICORN_KEEPALIVE="${GUNICORN_KEEPALIVE:-65}"
    export GUNICORN_MAX_REQUESTS="${GUNICORN_MAX_REQUESTS:-1000}"
    export GUNICORN_MAX_REQUESTS_JITTER="${GUNICORN_MAX_REQUESTS_JITTER:-50}"
    export GUNICORN_PRELOAD="${GUNICORN_PRELOAD:-true}"
    
    # Build Gunicorn command
    local gunicorn_cmd=(
        "gunicorn"
        "--config" "/app/gunicorn.conf.py"
        "--bind" "0.0.0.0:5000"
        "--workers" "$GUNICORN_WORKERS"
        "--threads" "$GUNICORN_THREADS"
        "--timeout" "$GUNICORN_TIMEOUT"
        "--keepalive" "$GUNICORN_KEEPALIVE"
        "--max-requests" "$GUNICORN_MAX_REQUESTS"
        "--max-requests-jitter" "$GUNICORN_MAX_REQUESTS_JITTER"
    )
    
    if [ "$GUNICORN_PRELOAD" = "true" ]; then
        gunicorn_cmd+=("--preload")
    fi
    
    # Add worker class for better performance
    gunicorn_cmd+=(
        "--worker-class" "gevent"
        "--worker-connections" "1000"
    )
    
    # Add logging configuration
    gunicorn_cmd+=(
        "--access-logfile" "/app/logs/access.log"
        "--error-logfile" "/app/logs/error.log"
        "--log-level" "info"
    )
    
    # Add application
    gunicorn_cmd+=("sql_synthesizer.webapp:create_app()")
    
    log "Starting with command: ${gunicorn_cmd[*]}"
    
    # Start Gunicorn
    exec "${gunicorn_cmd[@]}" &
    GUNICORN_PID=$!
    
    log "Gunicorn started with PID: $GUNICORN_PID"
    
    # Wait for the process
    wait "$GUNICORN_PID"
}

# Main execution
main() {
    log "SQL Query Synthesizer Production Entrypoint"
    log "Version: ${VERSION:-unknown}"
    log "Build Date: ${BUILD_DATE:-unknown}"
    log "VCS Ref: ${VCS_REF:-unknown}"
    
    # Handle different commands
    case "${1:-start}" in
        "start")
            initialize
            start_application
            ;;
        "init-only")
            initialize
            log_success "Initialization completed, exiting"
            ;;
        "health-check")
            run_health_check
            ;;
        "shell")
            exec /bin/bash
            ;;
        "python")
            shift
            exec python "$@"
            ;;
        *)
            log_error "Unknown command: $1"
            log "Available commands: start, init-only, health-check, shell, python"
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"